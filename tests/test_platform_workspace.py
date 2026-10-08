"""Generated indexes must resolve applications against their declared package root."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import msgspec
import pytest

from f8pysdk.platform_spec import DevelopmentCatalog
from scripts import platform_workspace


@pytest.mark.skipif(os.name == 'nt', reason='POSIX terminal process-group interrupt')
def test_foreground_ctrl_c_waits_for_child_cleanup_without_traceback() -> None:
    child = """
import signal
import time
stopping = False
def stop(signum, frame):
    global stopping
    stopping = True
signal.signal(signal.SIGINT, stop)
print('ready', flush=True)
while not stopping:
    time.sleep(0.01)
time.sleep(0.1)
print('child cleanup complete', flush=True)
"""
    wrapper = f"""
import sys
from scripts.platform_workspace import run_foreground
run_foreground([sys.executable, '-u', '-c', {child!r}])
print('wrapper complete', flush=True)
"""
    process = subprocess.Popen([sys.executable, '-u', '-c', wrapper],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == 'ready'
        os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0
        assert 'Stopping Platform' in stdout
        assert stdout.index('child cleanup complete') < stdout.index('wrapper complete')
        assert 'Traceback' not in stderr
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_foreground_preserves_real_failure_and_restores_interrupt_handler() -> None:
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(subprocess.CalledProcessError) as error:
        platform_workspace.run_foreground([sys.executable, '-c', 'raise SystemExit(7)'])
    assert error.value.returncode == 7
    assert signal.getsignal(signal.SIGINT) == previous


def test_development_config_uses_package_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generated = tmp_path / 'build/workspace'
    config = generated / 'config'
    config.mkdir(parents=True)
    package = tmp_path / 'extensions/editor'
    (package / 'config').mkdir(parents=True)
    declaration = {'schemaVersion': 'f8extensionCatalog/1', 'extensions': [{
        'extensionId': 'editor', 'name': 'Editor', 'version': '1.0', 'description': 'Fixture',
        'runtime': {'kind': 'workspace', 'environment': 'runtime'}, 'application': {
            'launch': {'environment': 'runtime', 'module': 'editor', 'distribution': 'editor'},
            'provides': [], 'requires': [], 'endpoints': [{'name': 'http', 'url': 'http://127.0.0.1:19431'}],
            'health': {'endpoint': 'http', 'path': '/health', 'service': 'editor', 'protocolVersion': 'api/1'},
        },
    }]}
    (package / 'config/extensions.json').write_text(json.dumps(declaration))
    (package / 'config/development-application.json').write_text(json.dumps([
        '-m', 'editor', '--assets', '${F8_PACKAGE_ROOT}/assets', '--port', '${F8_PORT:editor.http}',
    ]))
    (config / 'service-index.json').write_text(json.dumps({
        'schemaVersion': 'f8serviceIndex/1', 'services': [], 'modelRoot': '${F8_MODEL_ROOT}',
        'packageRoot': str(tmp_path),
    }))
    (config / 'extension-sources.json').write_text(json.dumps({'editor': '${F8_PACKAGE_ROOT}/extensions/editor'}))
    monkeypatch.setattr(platform_workspace, 'ROOT', tmp_path)
    monkeypatch.setattr(platform_workspace, 'GENERATED', generated)
    connection = tmp_path / 'discovery/platform.json'
    output = platform_workspace.development_config(tmp_path / 'data', connection_file=connection)
    applications = msgspec.json.decode(output.read_bytes(), type=DevelopmentCatalog).applications
    assert len(applications) == 1
    assert applications[0].manifest.extension_id == 'editor'
    assert applications[0].runtime_manifest == str(package / 'pixi.toml')
    assert Path(applications[0].arguments[3]) == package / 'assets'
    # Resolve dependency endpoints at launch time, after release selection/configuration.
    assert applications[0].arguments[-1] == '${F8_PORT:editor.http}'
    assert applications[0].environment['F8_PLATFORM_CONNECTION_FILE'] == str(connection)
