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
from scripts.extension_workspace import REPO_ROOT, check_workspace, source_packages


def test_all_services_are_owned_by_independent_extension_packages() -> None:
    check_workspace()
    packages = source_packages()
    assert len(packages) == 11
    assert {item.extension_ids for item in packages if item.package == 'f8pyengine'} == {('pyengine',)}
    engine = validate_package(REPO_ROOT / 'extensions/f8pyengine').extensions[0]
    assert engine.service_classes == ('f8.pyengine', 'f8.pyexpr', 'f8.pyscript')
    assert engine.runtime.environment == 'pyengine'
    assert not any(item.package == 'f8pyscript' for item in packages)
    classes: list[str] = []
    for package in packages:
        catalog = validate_package(REPO_ROOT / 'extensions' / package.package)
        classes.extend(name for extension in catalog.extensions for name in extension.service_classes)
    root_index = json.loads((REPO_ROOT / 'config/service-index.json').read_bytes())
    assert len(classes) == len(set(classes)) == len(root_index['services']) == 22
    assert set(classes) == {item['serviceClass'] for item in root_index['services']}


@pytest.mark.parametrize('module', ['f8pyengine', 'f8pyscript'])
def test_core_imports_cannot_reintroduce_service_implementation_dependencies(tmp_path: Path, module: str) -> None:
    source = tmp_path / 'packages/f8studio_core/f8studio_core'
    source.mkdir(parents=True)
    (source / 'bad.py').write_text(f'from {module}.main import main\n')
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'extensions.json').write_bytes((REPO_ROOT / 'config/extensions.json').read_bytes())
    (config / 'extension-workspace.toml').write_bytes((REPO_ROOT / 'config/extension-workspace.toml').read_bytes())
    catalog = msgspec.json.decode((config / 'extensions.json').read_bytes(), type=ExtensionCatalog)
    with mock.patch('scripts.extension_workspace.compose_catalog', return_value=catalog):
        with pytest.raises(ValueError, match='Core must not import an extension implementation'):
            check_workspace(tmp_path)


def test_extension_manifest_schema_is_public_sdk_api() -> None:
    from f8studio_server.extension_models import ExtensionCatalog as ServerCatalog

    assert ServerCatalog is ExtensionCatalog


@pytest.mark.parametrize('name', ['../escape.py', '/escape.py', 'pkg/../../escape.py', 'pkg\\escape.py'])
def test_wheel_extraction_rejects_paths_outside_package(tmp_path: Path, name: str) -> None:
    import zipfile

    wheel = tmp_path / 'evil.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr(name, 'malicious')
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
