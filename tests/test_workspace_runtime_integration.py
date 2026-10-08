"""Exercise SDK source changes against a real noneditable Pixi installation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.workspace_inputs import PythonWorkspace
from scripts.workspace_runtime import prepare_runtimes, runtime_command, runtime_environment

pytestmark = pytest.mark.skipif(os.environ.get('F8_TEST_RUNTIME_WORKSPACES') != '1',
    reason='Run pixi run workspace_runtime_test for the real extension environment regression')


def test_noneditable_sdk_refreshes_after_source_only_edit_and_module_deletion(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / 'sdk/python', tmp_path / 'sdk/python',
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', '.pytest_cache', '.ruff_cache'))
    shutil.copytree(root / 'extensions/f8pyengine', tmp_path / 'extensions/f8pyengine',
                    ignore=shutil.ignore_patterns('.git', '.sdk', '.pixi', 'build', 'dist', '__pycache__',
                                                 '*.egg-info', '.pytest_cache', '.ruff_cache', '.cache'))
    (tmp_path / 'scripts').mkdir()
    shutil.copy2(root / 'scripts/verify_workspace_sdk.py', tmp_path / 'scripts/verify_workspace_sdk.py')
    workspace = PythonWorkspace('extensions/f8pyengine', ('pyengine',))
    directory = tmp_path / workspace.path
    config = tmp_path / 'build/workspace/config'
    config.mkdir(parents=True)
    entry = config / 'pyengine.yml'
    entry.write_text(json.dumps({'serviceClass': 'f8.pyengine', 'launch': {
        'command': 'pixi', 'args': ['run', '-e', 'pyengine', 'f8pyengine'], 'workdir': str(directory)}}))
    (config / 'service-index.json').write_text(json.dumps({'schemaVersion': 'f8serviceIndex/1',
        'packageRoot': str(tmp_path), 'modelRoot': '${F8_MODEL_ROOT}', 'services': [{
        'serviceClass': 'f8.pyengine', 'manifests': {'any': str(entry)},
        'describe': str(tmp_path / 'build/workspace/runtime/pyengine/describe.json')}]}))
    obsolete = tmp_path / 'sdk/python/f8pysdk/workspace_obsolete.py'
    obsolete.write_text('VALUE = "old"\n')
    prepare_runtimes(tmp_path, workspaces=(workspace,))
    # No pyproject/version/lock metadata changes: this is the original missed case.
    policy = tmp_path / 'sdk/python/f8pysdk/_specs/state_policy.py'
    policy.write_text(policy.read_text() + '\nWORKSPACE_REFRESH_PROBE: str = "new SDK source"\n')
    obsolete.unlink()
    entrypoint = directory / 'f8pyengine/main.py'
    entrypoint.write_text(entrypoint.read_text().replace('from __future__ import annotations\n',
        'from __future__ import annotations\nfrom f8pysdk._specs.state_policy import WORKSPACE_REFRESH_PROBE\n'))
    prepare_runtimes(tmp_path, workspaces=(workspace,))
    assert not (directory / '.sdk/python/f8pysdk/workspace_obsolete.py').exists()
    probe = subprocess.run(runtime_command(tmp_path, workspace, 'pyengine', 'python', '-c',
        'from f8pysdk._specs.state_policy import WORKSPACE_REFRESH_PROBE; print(WORKSPACE_REFRESH_PROBE)'),
        cwd=directory, env=runtime_environment(), check=True, capture_output=True, text=True)
    assert probe.stdout.strip() == 'new SDK source'
    prepare_runtimes(tmp_path, workspaces=(workspace,), check_only=True)
