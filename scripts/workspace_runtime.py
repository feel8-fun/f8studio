"""Prepare and verify independent Python runtimes for workspace development."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import msgspec

from f8pysdk.service_runtime_tools.inventory.index import index_paths, indexed_entry, read_service_index
from .install_services import install
from .workspace_inputs import PYTHON_WORKSPACES, ROOT, PythonWorkspace, sync_inputs, tree_hash, workspace_lock


class RuntimeReceipt(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    sdk: str
    workspace: str
    descriptions: dict[str, str]
    tooling: str = ''


def runtime_command(root: Path, workspace: PythonWorkspace, environment: str, *arguments: str) -> list[str]:
    return ['pixi', 'run', '--frozen', '--no-install', '--manifest-path',
            str(root / workspace.path / 'pixi.toml'), '-e', environment, *arguments]


def runtime_environment() -> dict[str, str]:
    # A root integration interpreter's source paths must never mask installed SDKs.
    return {key: value for key, value in os.environ.items()
            if key not in {'PYTHONPATH', 'PYTHONHOME', 'PIXI_PROJECT_MANIFEST', 'PIXI_ENVIRONMENT_NAME'}}


def receipt_path(root: Path, workspace: PythonWorkspace, environment: str) -> Path:
    return root / 'build/workspace/prepared-runtimes' / (workspace.path.replace('/', '_') + '_' + environment + '.json')


def is_installed(root: Path, workspace: PythonWorkspace, environment: str) -> bool:
    return (root / workspace.path / '.pixi/envs' / environment / 'conda-meta/history').is_file()


def service_descriptions(root: Path, workspaces: tuple[PythonWorkspace, ...]) -> dict[str, Path]:
    index_path = root / 'build/workspace/config/service-index.json'
    index = read_service_index(index_path)
    directories = {(root / workspace.path).resolve() for workspace in workspaces}
    result: dict[str, Path] = {}
    for item in index.services:
        entry = indexed_entry(index_path, index, item)
        if entry is not None and entry.launch.command in {'pixi', 'pixi.exe'} and Path(entry.launch.workdir or '.').resolve() in directories:
            result[item.serviceClass] = index_paths(index_path, index, item).package_path(item.describe, relative_to=index_path.parent)
    return result


def entrypoint_checks(workspace: PythonWorkspace) -> tuple[tuple[str, ...], ...]:
    if workspace.path == 'platform':
        return (('python', '-m', 'f8platform', '--help'),)
    if workspace.path == 'extensions/f8webstudio':
        return (('python', '-m', 'f8studio_server', '--help'),)
    if workspace.path == 'extensions/f8mediagateway':
        return (('python', '-m', 'f8media_gateway', '--help'),)
    if workspace.path == 'extensions/f8diagnostics':
        # These are stdin-based tools, not argparse CLIs. Import without running I/O.
        return (('python', '-c', 'import f8diagnostics.verify; import f8diagnostics.simulate'),)
    return ()


def prepare_runtimes(root: Path = ROOT, *, workspaces: tuple[PythonWorkspace, ...] = PYTHON_WORKSPACES,
                     installed_only: bool = False, check_only: bool = False) -> int:
    with workspace_lock(root):
        return _prepare_runtimes(root, workspaces=workspaces, installed_only=installed_only, check_only=check_only)


def _prepare_runtimes(root: Path, *, workspaces: tuple[PythonWorkspace, ...],
                      installed_only: bool, check_only: bool) -> int:
    if installed_only:
        workspaces = tuple(PythonWorkspace(item.path, tuple(environment for environment in item.environments
            if is_installed(root, item, environment))) for item in workspaces)
        workspaces = tuple(item for item in workspaces if item.environments)
    if not check_only:
        sync_inputs(root, workspaces=workspaces)
    sdk_hash = tree_hash(root / 'sdk')
    tooling_hash = tree_hash(root / 'scripts')
    validated: list[tuple[PythonWorkspace, str, RuntimeReceipt]] = []
    changed: list[PythonWorkspace] = []
    for workspace in workspaces:
        directory = root / workspace.path
        workspace_hash = tree_hash(directory)
        descriptions = service_descriptions(root, (workspace,))
        hashes = {name: tree_hash(path.parent) if path.is_file() else '' for name, path in descriptions.items()}
        expected = RuntimeReceipt(sdk_hash, workspace_hash, hashes, tooling_hash)
        needs_check = check_only
        for environment in workspace.environments:
            record = receipt_path(root, workspace, environment)
            previous = msgspec.json.decode(record.read_bytes(), type=RuntimeReceipt) if record.is_file() else None
            if check_only and (previous != expected or not is_installed(root, workspace, environment)):
                raise ValueError(f'Unprepared runtime: {workspace.path} / {environment}. Run pixi run workspace_runtime_prepare.')
            if not check_only:
                subprocess.run(['pixi', 'install', '--locked', '--manifest-path', str(directory / 'pixi.toml'),
                                '-e', environment], cwd=directory, env=runtime_environment(), check=True)
            verification = runtime_command(root, workspace, environment, 'python',
                str(root / 'scripts/verify_workspace_sdk.py'), str(root / 'sdk/python'))
            probe = subprocess.run(verification, cwd=directory, env=runtime_environment(),
                                   capture_output=True, text=True, timeout=30, check=False)
            if probe.returncode != 0:
                if check_only:
                    raise RuntimeError(f'SDK verification failed for {workspace.path} / {environment}. '
                        f'Run pixi run workspace_runtime_prepare.\n{probe.stdout}\n{probe.stderr}')
                print(f'Rebuilding stale SDK in {workspace.path} / {environment}:\n{probe.stderr}', file=sys.stderr, flush=True)
                subprocess.run(['pixi', 'reinstall', '--locked', '--manifest-path', str(directory / 'pixi.toml'),
                                '-e', environment, 'f8pysdk'], cwd=directory, env=runtime_environment(), check=True)
                subprocess.run(verification, cwd=directory, env=runtime_environment(), check=True, timeout=30)
                needs_check = True
            else:
                print(probe.stdout.strip(), flush=True)
            if previous != expected:
                needs_check = True
            validated.append((workspace, environment, expected))
        if needs_check:
            for environment in workspace.environments:
                for arguments in entrypoint_checks(workspace):
                    subprocess.run(runtime_command(root, workspace, environment, *arguments),
                                   cwd=directory, env=runtime_environment(), check=True, timeout=30, stdout=subprocess.DEVNULL)
            changed.append(workspace)
    descriptions = service_descriptions(root, tuple(changed))
    if descriptions:
        install(root / 'build/workspace/config/service-index.json', refresh=True,
                service_classes=set(descriptions), no_install=True, python_only=True, validate_only=check_only)
    if not check_only:
        if tree_hash(root / 'sdk') != sdk_hash:
            raise ValueError('SDK source changed during runtime preparation. Run workspace_runtime_prepare again.')
        # Only record success after all selected real entrypoints have passed.
        for workspace, environment, expected in validated:
            descriptions = service_descriptions(root, (workspace,))
            receipt = msgspec.structs.replace(expected, descriptions={name: tree_hash(path.parent) for name, path in descriptions.items()})
            record = receipt_path(root, workspace, environment)
            record.parent.mkdir(parents=True, exist_ok=True)
            temporary = record.with_suffix('.tmp')
            temporary.write_bytes(msgspec.json.encode(receipt))
            temporary.replace(record)
    print(f'Verified {len(validated)} independent runtime environments.', flush=True)
    return len(validated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check'))
    parser.add_argument('--workspace', action='append', choices=[item.path for item in PYTHON_WORKSPACES])
    parser.add_argument('--installed-only', action='store_true', help='Prepare only environments already installed locally')
    args = parser.parse_args()
    selected = tuple(item for item in PYTHON_WORKSPACES if not args.workspace or item.path in args.workspace)
    prepare_runtimes(workspaces=selected, installed_only=args.installed_only, check_only=args.action == 'check')


if __name__ == '__main__':
    main()
