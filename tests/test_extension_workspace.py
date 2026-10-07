from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
import sys
from unittest import mock
import zipfile

import msgspec
import pytest

from f8pysdk.extension_packaging import _extract_wheel, build_extension, validate_package
from f8pysdk.extension_spec import ExtensionCatalog
from scripts.extension_workspace import REPO_ROOT, check_workspace, source_packages, sync_workspace, workspace_index, workspace_root


def test_all_services_are_owned_by_independent_extension_packages() -> None:
    check_workspace()
    packages = source_packages()
    assert len(packages) == 13
    assert {item.extension_ids for item in packages if item.package == 'f8pyengine'} == {('pyengine',)}
    engine = validate_package(REPO_ROOT / 'extensions/f8pyengine').extensions[0]
    assert engine.service_classes == ('f8.pyengine', 'f8.pyexpr', 'f8.pyscript')
    assert engine.runtime.environment == 'pyengine'
    assert not any(item.package == 'f8pyscript' for item in packages)
    classes: list[str] = []
    for package in packages:
        catalog = validate_package(REPO_ROOT / 'extensions' / package.package)
        classes.extend(name for extension in catalog.extensions for name in extension.service_classes)
    root_index = json.loads(workspace_index().read_bytes())
    assert len(classes) == len(set(classes)) == len(root_index['services']) == 22
    assert set(classes) == {item['serviceClass'] for item in root_index['services']}


@pytest.mark.parametrize('module', ['f8pyengine', 'f8pyscript'])
def test_core_imports_cannot_reintroduce_service_implementation_dependencies(tmp_path: Path, module: str) -> None:
    source = tmp_path / 'extensions/f8webstudio/f8studio_core/f8studio_core'
    source.mkdir(parents=True)
    (source / 'bad.py').write_text(f'from {module}.main import main\n')
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'extension-workspace.toml').write_text((REPO_ROOT / 'config/extension-workspace.toml').read_text().replace('core = []', 'core = [\"f8studio_core\"]'))
    catalog = msgspec.json.decode((workspace_root() / 'config/extensions.json').read_bytes(), type=ExtensionCatalog)
    with mock.patch('scripts.extension_workspace.compose_catalog', return_value=catalog):
        with pytest.raises(ValueError, match='Core must not import an extension implementation'):
            check_workspace(tmp_path)


def test_extension_manifest_schema_is_public_sdk_api() -> None:
    from f8platform.extension_models import ExtensionCatalog as ServerCatalog

    assert ServerCatalog is ExtensionCatalog


def test_clean_workspace_generation_keeps_same_named_environments_independent(tmp_path: Path) -> None:
    from f8platform.extensions import ExtensionManager
    from f8platform.extension_models import ExtensionRecord

    config = tmp_path / 'config'
    config.mkdir()
    (config / 'extension-workspace.toml').write_text(
        'core = []\napplications = []\n'
        '[[extensions]]\npackage = "f8alpha"\nextension_ids = ["alpha"]\n'
        '[[extensions]]\npackage = "f8beta"\nextension_ids = ["beta"]\n'
    )
    for name in ('alpha', 'beta'):
        source = tmp_path / 'extensions' / f'f8{name}'
        source.mkdir(parents=True)
        (source / 'models.md').write_text('Model profiles')
        (source / 'extension.json').write_text(json.dumps({
            'schemaVersion': 'f8extensionCatalog/1', 'extensions': [{
                'extensionId': name, 'name': name, 'version': '1', 'description': name,
                'runtime': {'kind': 'workspace', 'environment': 'default'},
                'tools': [{'toolId': 'run', 'name': 'Run', 'description': '', 'command': 'python',
                           'args': ['-m', f'f8{name}', '${F8_PACKAGE_ROOT}/models.md'], 'workdir': '.'}],
                'resources': [{'resourceId': 'models', 'path': 'models.md'}],
            }],
        }))
        (source / 'pixi.toml').write_text('[workspace]\nname="fixture"\n[environments]\ndefault=[]\nextra=[]\n')
        (source / 'pixi.lock').write_text('fixture lock\n')
    path = sync_workspace(tmp_path)
    before = {file: file.read_bytes() for file in path.parent.rglob('*') if file.is_file()}
    assert sync_workspace(tmp_path) == path
    assert before == {file: file.read_bytes() for file in path.parent.rglob('*') if file.is_file()}
    assert list(config.iterdir()) == [config / 'extension-workspace.toml']
    manager = ExtensionManager(tmp_path / 'data', base_index=path)
    assert manager.install_plan('alpha').environment_id != manager.install_plan('beta').environment_id
    assert sorted(source.name for source in manager.runtime_registry.sources.values()) == ['default', 'default', 'extra', 'extra']
    manager._commit_record('alpha', ExtensionRecord(version='1', installed=True, enabled=True))
    assert manager.capability_file('alpha', manager._manifest('alpha').resources[0].path) == tmp_path / 'extensions/f8alpha/models.md'
    with mock.patch('f8platform.environments.EnvironmentManager.ready', return_value=True):
        with mock.patch('f8platform.environments.EnvironmentManager.python_launch', return_value=('python', [])):
            _, command, cwd, _ = manager.tool_launcher('alpha', 'run')
    assert cwd == tmp_path / 'extensions/f8alpha'
    assert command[-1] == str(cwd / 'models.md')


def test_generated_workspace_installs_models_from_extension_owned_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from f8platform.extensions import ExtensionManager

    destination = tmp_path / 'models'
    monkeypatch.setenv('F8_MODEL_ROOT', str(destination))
    manager = ExtensionManager(tmp_path / 'data', base_index=workspace_index())
    manager._copy_model_metadata(manager._manifest('dl'))
    expected = REPO_ROOT / 'extensions/f8pydl/resources/models/onnx/neuflow_mixed.yaml'
    installed = destination / 'onnx' / expected.name
    assert installed.read_bytes() == expected.read_bytes()
    installed.write_text('user override')
    manager._copy_model_metadata(manager._manifest('dl'))
    assert installed.read_text() == 'user override'


@pytest.mark.parametrize('name', ['../escape.py', '/escape.py', 'pkg/../../escape.py', 'pkg\\escape.py'])
def test_wheel_extraction_rejects_paths_outside_package(tmp_path: Path, name: str) -> None:
    import zipfile

    wheel = tmp_path / 'evil.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        info = zipfile.ZipInfo(name)
        # ZipInfo normalizes Windows separators when constructed. Preserve the
        # malicious archive member exactly as received from an external wheel.
        info.filename = name
        archive.writestr(info, 'malicious')
    with pytest.raises(ValueError, match='Unsafe wheel path'):
        _extract_wheel(wheel, tmp_path / 'python')
    assert not (tmp_path / 'escape.py').exists()


@pytest.mark.skipif(sys.platform != 'linux', reason='POSIX symlink models Windows temp directory aliases')
def test_native_packaging_accepts_a_temporary_directory_alias(tmp_path: Path) -> None:
    real_stage = tmp_path / 'real-stage'
    real_stage.mkdir()
    alias = tmp_path / 'stage-alias'
    alias.symlink_to(real_stage, target_is_directory=True)
    runtime = tmp_path / 'runtime'
    binary = runtime / 'f8.screencap/0.0.1/linux/f8screencap_service'
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'native payload')
    output = tmp_path / 'extension.zip'
    with mock.patch('f8pysdk.extension_packaging.tempfile.TemporaryDirectory', return_value=nullcontext(str(alias))):
        with mock.patch('f8pysdk.extension_packaging._refresh_describes') as refresh:
            build_extension(REPO_ROOT / 'extensions/f8screencap', output, runtime_root=runtime)
            refresh.assert_called_once_with(real_stage, python_package=False)
    with zipfile.ZipFile(output) as archive:
        assert archive.read('runtime/bundles/f8.screencap/0.0.1/linux/f8screencap_service') == b'native payload'
