from __future__ import annotations

import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from scripts.workspace_inputs import PYTHON_WORKSPACES, PythonWorkspace
from scripts.workspace_runtime import prepare_runtimes, receipt_path

WORKSPACE = PythonWorkspace('extensions/fixture', ('runtime',))


@pytest.fixture
def runtime_root(tmp_path: Path) -> Path:
    sdk = tmp_path / 'sdk/python/f8pysdk'
    sdk.mkdir(parents=True)
    (sdk / '__init__.py').write_text('SDK source')
    workspace = tmp_path / WORKSPACE.path
    workspace.mkdir(parents=True)
    (workspace / 'pixi.toml').write_text('[environments]\nruntime=[]\n[tasks]\nfixture="python -m fixture"\n')
    (workspace / 'fixture.py').write_text('service source')
    config = tmp_path / 'build/workspace/config'
    config.mkdir(parents=True)
    manifest = config / 'service.yml'
    manifest.write_text(json.dumps({'serviceClass': 'f8.fixture', 'launch': {
        'command': 'pixi', 'args': ['run', '-e', 'runtime', 'fixture'], 'workdir': str(workspace)}}))
    (config / 'service-index.json').write_text(json.dumps({'schemaVersion': 'f8serviceIndex/1',
        'packageRoot': str(tmp_path), 'modelRoot': '${F8_MODEL_ROOT}', 'services': [{
        'serviceClass': 'f8.fixture', 'manifests': {'any': str(manifest)},
        'describe': str(tmp_path / 'build/workspace/runtime/fixture/describe.json')}]}))
    return tmp_path


def _run(root: Path, calls: list[list[str]], *, stale: bool = False, describe_failure: bool = False):
    needs_rebuild = stale

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal needs_rebuild
        calls.append(command)
        env = kwargs['env']
        assert isinstance(env, dict) and 'PYTHONPATH' not in env and 'PIXI_PROJECT_MANIFEST' not in env
        if command[1] == 'install':
            history = root / WORKSPACE.path / '.pixi/envs/runtime/conda-meta/history'
            history.parent.mkdir(parents=True, exist_ok=True)
            history.touch()
            assert '--locked' in command
        elif command[1] == 'reinstall':
            assert command[-1] == 'f8pysdk' and '--locked' in command
            needs_rebuild = False
        elif any(item.endswith('verify_workspace_sdk.py') for item in command):
            assert '--frozen' in command and '--no-install' in command
            if needs_rebuild:
                return subprocess.CompletedProcess(command, 1, stdout='', stderr='Stale installed SDK')
            return subprocess.CompletedProcess(command, 0, stdout='SDK verified', stderr='')
        elif command[-1] == '--describe':
            if describe_failure:
                raise subprocess.CalledProcessError(1, command, stderr='Unexpected keyword argument persistent')
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps({'service': {
                'schemaVersion': 'f8service/2', 'serviceClass': 'f8.fixture', 'label': 'Fixture'}, 'operators': []}))
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0)

    return run


def test_preparation_repairs_stale_wheel_even_when_description_already_exists(runtime_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PYTHONPATH', '/root/integration/sdk')
    monkeypatch.setenv('PIXI_PROJECT_MANIFEST', '/root/pixi.toml')
    calls: list[list[str]] = []
    with patch('scripts.workspace_runtime.subprocess.run', side_effect=_run(runtime_root, calls, stale=True)):
        assert prepare_runtimes(runtime_root, workspaces=(WORKSPACE,)) == 1
    assert any(command[1] == 'reinstall' for command in calls)
    assert calls[-1][-1] == '--describe'
    record = receipt_path(runtime_root, WORKSPACE, 'runtime')
    before = record.read_bytes()
    calls.clear()
    with patch('scripts.workspace_runtime.subprocess.run', side_effect=_run(runtime_root, calls)):
        prepare_runtimes(runtime_root, workspaces=(WORKSPACE,))
    assert not any(command[1] == 'reinstall' or command[-1] == '--describe' for command in calls)
    assert record.read_bytes() == before
    # The installed SDK can go stale independently of our receipts (e.g. manual reinstall).
    calls.clear()
    with patch('scripts.workspace_runtime.subprocess.run', side_effect=_run(runtime_root, calls, stale=True)):
        prepare_runtimes(runtime_root, workspaces=(WORKSPACE,))
    assert any(command[1] == 'reinstall' for command in calls)
    assert calls[-1][-1] == '--describe'


def test_source_edits_invalidate_receipt_and_failed_entrypoint_does_not_record_success(runtime_root: Path) -> None:
    calls: list[list[str]] = []
    with patch('scripts.workspace_runtime.subprocess.run', side_effect=_run(runtime_root, calls)):
        prepare_runtimes(runtime_root, workspaces=(WORKSPACE,))
    record = receipt_path(runtime_root, WORKSPACE, 'runtime')
    before = record.read_bytes()
    (runtime_root / WORKSPACE.path / 'fixture.py').write_text('new node requiring persistent')
    with patch('scripts.workspace_runtime.subprocess.run') as run:
        with pytest.raises(ValueError, match='Unprepared runtime'):
            prepare_runtimes(runtime_root, workspaces=(WORKSPACE,), check_only=True)
        run.assert_not_called()
    with patch('scripts.workspace_runtime.subprocess.run', side_effect=_run(runtime_root, calls, describe_failure=True)):
        with pytest.raises(RuntimeError, match='Unexpected keyword argument'):
            prepare_runtimes(runtime_root, workspaces=(WORKSPACE,))
    assert record.read_bytes() == before


def test_check_runs_real_entrypoints_without_installing_or_writing_receipt(runtime_root: Path) -> None:
    calls: list[list[str]] = []
    with patch('scripts.workspace_runtime.subprocess.run', side_effect=_run(runtime_root, calls)):
        prepare_runtimes(runtime_root, workspaces=(WORKSPACE,))
    record = receipt_path(runtime_root, WORKSPACE, 'runtime')
    stamp = record.stat().st_mtime_ns
    calls.clear()
    with patch('scripts.workspace_runtime.subprocess.run', side_effect=_run(runtime_root, calls)):
        prepare_runtimes(runtime_root, workspaces=(WORKSPACE,), check_only=True)
    assert all(command[1] == 'run' for command in calls)
    assert calls[-1][-1] == '--describe'
    assert record.stat().st_mtime_ns == stamp
    with patch('scripts.workspace_runtime.subprocess.run', side_effect=_run(runtime_root, [], stale=True)):
        with pytest.raises(RuntimeError, match='SDK verification failed'):
            prepare_runtimes(runtime_root, workspaces=(WORKSPACE,), check_only=True)


def test_automatic_startup_skips_uninstalled_optional_runtimes(runtime_root: Path) -> None:
    with patch('scripts.workspace_runtime.subprocess.run') as run:
        assert prepare_runtimes(runtime_root, workspaces=(WORKSPACE,), installed_only=True) == 0
        run.assert_not_called()
    assert not (runtime_root / WORKSPACE.path / '.sdk').exists()


def test_workspace_runtime_registry_covers_all_python_extension_workspaces() -> None:
    from scripts.extension_workspace import compose_catalog, source_packages
    catalog = compose_catalog()
    owners = {manifest.extension_id: manifest for manifest in catalog.extensions}
    expected = {'platform'}
    for package in source_packages():
        if any(owners[name].runtime.kind == 'workspace' for name in package.extension_ids):
            expected.add('extensions/' + package.package)
    assert {item.path for item in PYTHON_WORKSPACES} == expected


def test_development_startup_prepares_installed_runtimes_and_management_remains_lightweight() -> None:
    import tomllib
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((root / 'pixi.toml').read_text())
    sdk_tasks = manifest['feature']['sdk']['tasks']
    platform_tasks = manifest['tasks']
    assert '--installed-only' in sdk_tasks['workspace_runtime_ensure']['cmd']
    for name in ('platform_ensure', 'platform_dev', 'platform_tray'):
        assert any(item['task'] == 'workspace_runtime_ensure' for item in platform_tasks[name]['depends-on'])
    for name in ('platform_stop', 'platform_open', 'platform_cli'):
        assert all(item['task'] not in {'workspace_prepare', 'workspace_runtime_ensure'}
                   for item in platform_tasks[name]['depends-on'])
