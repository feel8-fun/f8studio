"""Connect development entrypoints to one separately running platform daemon."""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
from types import FrameType
from urllib.parse import urlsplit

import msgspec

from f8pysdk.application_package import read_application
from f8pysdk.platform_client import PlatformClient, PlatformConnection
from f8pysdk.platform_errors import ServiceUnavailableError
from f8pysdk.platform_spec import DevelopmentApplication, DevelopmentCatalog
from f8pysdk.service_runtime_tools.inventory.index import index_paths, read_service_index

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / 'build/workspace'
CONNECTION = GENERATED / 'platform-connection.json'
logger = logging.getLogger(__name__)


def run_foreground(command: list[str]) -> None:
    """Let the foreground child handle Ctrl+C and wait for its shutdown."""
    interrupted = False

    def interrupt(_signum: int, _frame: FrameType | None) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            print('Stopping Platform…', flush=True)
        # The terminal sends SIGINT to the entire foreground process group,
        # including the child. Do not interrupt our wait or send it twice.

    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        if interrupted and exc.returncode in {-signal.SIGINT, 130}:
            raise SystemExit(130) from None
        raise
    finally:
        signal.signal(signal.SIGINT, previous)


def expand_workspace_argument(argument: str, substitutions: dict[str, str]) -> str:
    for key, value in substitutions.items():
        argument = argument.replace(key, value)
    if '${' in argument and not any(token in argument for token in ('${F8_ENDPOINT:', '${F8_PORT:')):
        raise ValueError(f'Unresolved development argument: {argument}')
    return argument


def development_config(data: Path, *, connection_file: Path = CONNECTION) -> Path:
    config = GENERATED / 'config'
    sources = msgspec.json.decode((config / 'extension-sources.json').read_bytes(), type=dict[str, str])
    paths = index_paths(config / 'service-index.json', read_service_index(config / 'service-index.json'))
    applications: list[DevelopmentApplication] = []
    for reference in sources.values():
        source = paths.package_path(reference, relative_to=ROOT)
        settings = source / 'config/development-application.json'
        if not settings.is_file():
            continue
        manifest = read_application(source)
        arguments = msgspec.json.decode(settings.read_bytes(), type=tuple[str, ...])
        substitutions = {'${F8_PACKAGE_ROOT}': str(source), '${F8_DATA_ROOT}': str(data)}
        applications.append(DevelopmentApplication(manifest=manifest, runtime_manifest=str(source / 'pixi.toml'), workdir=str(ROOT),
            arguments=tuple(expand_workspace_argument(value, substitutions) for value in arguments),
            environment={**{key:expand_workspace_argument(value, substitutions) for key,value in manifest.launch.env.items()},
                         'F8_SERVICE_INDEX':str(config / 'service-index.json'),
                         'F8_PLATFORM_CONNECTION_FILE':str(connection_file)}))
    output = connection_file.with_name('development-applications.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(msgspec.json.encode(DevelopmentCatalog(applications=tuple(applications))))
    return output


def available(connection_file: Path = CONNECTION) -> bool:
    if not connection_file.is_file():
        return False
    connection = msgspec.json.decode(connection_file.read_bytes(), type=PlatformConnection)
    if not Path(connection.token_file).is_file():
        logger.info('Previous workspace platform token is missing')
        return False
    client = PlatformClient(connection)
    try:
        response = client.request('GET','/api/health')
        return response.is_success and response.json().get('protocolVersion') == 'f8platform-api/1'
    except ServiceUnavailableError:
        logger.info('Previous workspace platform is unavailable', exc_info=True)
        return False
    finally:
        client.close()


def main() -> None:
    if sys.argv[1:2] == ['cli']:
        subprocess.run([sys.executable, '-m', 'f8platform', *sys.argv[2:]], check=True,
            env={**os.environ, 'F8_PLATFORM_CONNECTION_FILE': str(CONNECTION)})
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('ensure','serve','tray','open','stop'))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--connection-file', type=Path, default=CONNECTION)
    args = parser.parse_args()
    data = args.data_dir or Path(os.environ.get('F8_PLATFORM_DATA_ROOT', str(Path.home()/'.feel8/workspaces'/hashlib.sha256(str(ROOT).encode()).hexdigest()[:16])))
    data.mkdir(parents=True,exist_ok=True)
    connection_file = args.connection_file.resolve()
    os.environ['F8_PLATFORM_CONNECTION_FILE'] = str(connection_file)
    if args.action in {'open','stop'}:
        client = PlatformClient.from_environment()
        try:
            if args.action == 'open':
                import webbrowser
                token = Path(client.connection.token_file).read_text().strip()
                if not webbrowser.open(f'{client.connection.url}/bootstrap/{token}'):
                    raise RuntimeError('No browser available')
            else:
                client.request('POST','/api/shutdown').raise_for_status()
        finally:
            client.close()
        return
    if available(connection_file):
        if args.action == 'ensure':
            print('Workspace platform is ready.')
            return
        raise SystemExit('Workspace platform is already running. Use platform_open or platform_stop.')
    definitions = development_config(data, connection_file=connection_file)
    previous = msgspec.json.decode(connection_file.read_bytes(), type=PlatformConnection) if connection_file.is_file() else None
    preferred = urlsplit(previous.url).port if previous else None
    with socket.socket() as listener:
        try:
            listener.bind(('127.0.0.1',preferred or 0))
        except OSError:
            logger.info('Previous platform port is occupied; selecting a free port', exc_info=True)
            listener.bind(('127.0.0.1',0))
        port = listener.getsockname()[1]
    command = [sys.executable,'-m','f8platform','--data-dir',str(data),'--port',str(port),'serve',
        '--source-index',str(GENERATED/'config/service-index.json'),'--development-config',str(definitions)]
    connection_file.write_bytes(msgspec.json.encode(PlatformConnection(url=f'http://127.0.0.1:{port}', token_file=str(data/'platform-token'))))
    if args.action in {'serve','tray'}:
        if args.action == 'tray':
            command.append('--tray')
        run_foreground(command)
        return
    log_path = data/'platform-console.log'
    with log_path.open('ab') as log:
        process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Platform startup failed; see {log_path}')
        if (data/'platform-token').is_file() and available(connection_file):
            print(f'Workspace platform is ready on port {port}. Log: {log_path}')
            return
        time.sleep(0.1)
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError(f'Platform startup timed out; see {log_path}')


if __name__ == '__main__':
    main()
