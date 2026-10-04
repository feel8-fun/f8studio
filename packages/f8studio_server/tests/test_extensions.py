from __future__ import annotations

from f8pysdk.service_paths import ServicePaths

import asyncio
import json
import hashlib
import io
import os
from pathlib import Path
import sys
import subprocess
from threading import Event
import time
import zipfile
from unittest.mock import patch

import httpx
import msgspec
import pytest
import yaml

from f8pysdk._specs.builtin_fields import normalize_describe_payload_dict
from f8pysdk.service_runtime_tools.inventory.index import IndexedService, ServiceIndex, indexed_entry, read_service_index
from f8studio_server.app import create_app
from f8studio_server.application import StudioApplication
from f8studio_server.catalog import CatalogService
from f8studio_server.environments import EnvironmentManager
from f8studio_server.errors import ConflictError, InvalidRequestError, NotFoundError
from f8studio_server.extension_artifacts import prepare_artifact
from f8studio_server.extension_models import ExtensionCatalog, ExtensionImportRequest, ExtensionManifest
from f8studio_server.extension_operation import ExtensionInstallCancelled, InstallOperation
from f8studio_server.extensions import ExtensionManager
from f8studio_server.shared_dependencies import (
    InstalledDistribution, RuntimeProbe, check_dependencies, validate_shared_package,
)
from f8studio_server.events import EventJournal
from f8studio_server.processes import ManagedServiceProcesses
from f8studio_server.runtime import RuntimeConfig


def _fixture(tmp_path: Path, *, kind: str = 'native', preinstalled: bool = False,
             names: tuple[str, ...] = ('alpha', 'beta')) -> tuple[ExtensionManager, Path]:
    source = tmp_path / 'release'
    config = source / 'config'
    config.mkdir(parents=True)
    services: list[IndexedService] = []
    extensions: list[dict[str, object]] = []
    for name in names:
        service_class = f'test.{name}'
        description = normalize_describe_payload_dict({
            'service': {'schemaVersion': 'f8service/1', 'serviceClass': service_class, 'label': name},
            'operators': [],
        })
        describe = source / f'{name}.json'
        describe.write_text(json.dumps(description))
        script = source / f'{name}.py'
        script.write_text(f'import pathlib\nprint(pathlib.Path({str(describe)!r}).read_text())\n')
        entry = {'schemaVersion': 'f8serviceEntry/1', 'serviceClass': service_class, 'label': name,
                 'version': '1.0.0', 'launch': {'command': sys.executable, 'args': [str(script)], 'workdir': '..'}}
        if kind == 'pixi':
            entry['launch'] = {'command': 'pixi', 'args': ['run', '-e', 'shared', name], 'workdir': '..'}
        (config / f'{name}.yml').write_text(yaml.safe_dump(entry))
        services.append(IndexedService(serviceClass=service_class, manifests={'any': f'{name}.yml'},
                                       describe=f'../{name}.json'))
        extensions.append({'extensionId': name, 'name': name.title(), 'version': '1.0.0',
                           'description': f'{name} services', 'serviceClasses': [service_class],
                           'runtime': {'kind': kind, **({'environment': 'shared'} if kind == 'pixi' else {})}})
    index = ServiceIndex(schemaVersion='f8serviceIndex/1', services=tuple(services), modelRoot='../resources/models')
    (config / 'service-index.json').write_bytes(msgspec.json.encode(index))
    (config / 'extensions.json').write_text(json.dumps({
        'schemaVersion': 'f8extensionCatalog/1', 'extensions': extensions,
        'preinstalled': names if preinstalled else [],
    }))
    (source / 'pixi.toml').write_text('[workspace]\nname = "fixture"\n[environments]\nshared = { features = [] }\n')
    (source / 'pixi.lock').write_text('locked release version one\n')
    (source / 'wheels').mkdir()
    (source / 'wheels' / 'fixture.whl').write_bytes(b'wheel')
    manager = ExtensionManager(tmp_path / 'data', base_index=config / 'service-index.json')
    return manager, source


async def _finish(manager: ExtensionManager, extension_id: str, catalog: CatalogService) -> None:
    assert (await manager.install(extension_id, catalog.refresh)).state == 'installing'
    assert manager._task is not None
    await manager._task
    assert manager.status(extension_id).state == 'installed', manager.status(extension_id).detail


@pytest.mark.parametrize('variable', ['F8_MODEL_ROOT', 'F8_RESOURCE_ROOT'])
def test_model_metadata_and_registration_use_configured_writable_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str,
) -> None:
    monkeypatch.delenv('F8_MODEL_ROOT', raising=False)
    monkeypatch.delenv('F8_RESOURCE_ROOT', raising=False)
    configured = tmp_path / 'external resources'
    monkeypatch.setenv(variable, str(configured))
    _manager, source = _fixture(tmp_path, names=('alpha',))
    metadata = source / 'resources/models/example/model.yaml'
    metadata.parent.mkdir(parents=True)
    metadata.write_text('model: example\n')
    catalog_path = source / 'config/extensions.json'
    raw = json.loads(catalog_path.read_text())
    raw['extensions'][0]['modelDirectories'] = ['example']
    catalog_path.write_text(json.dumps(raw))
    manager = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
    catalog = CatalogService(extension_indexes=manager.active_indexes)
    asyncio.run(_finish(manager, 'alpha', catalog))
    models = configured if variable == 'F8_MODEL_ROOT' else configured / 'models'
    assert (models / 'example/model.yaml').read_text() == metadata.read_text()
    assert not (tmp_path / 'data/models').exists()
    registration = manager._registration('alpha')
    index = read_service_index(registration)
    entry = indexed_entry(registration, index, index.services[0])
    assert entry is not None
    assert entry.launch.env['F8_MODEL_ROOT'] == str(models)
    assert entry.launch.env['F8_PACKAGE_ROOT'] == str(source)


def _runtime_probe() -> RuntimeProbe:
    from f8studio_server import _runtime_probe as probe
    output = subprocess.run([sys.executable, '-I', str(Path(probe.__file__))],
                            capture_output=True, text=True, check=True, timeout=30).stdout
    return msgspec.json.decode(output.encode(), type=RuntimeProbe)


def test_official_extension_environment_names_match_all_platform_service_launches() -> None:
    config = Path(__file__).resolve().parents[3] / 'config'
    extensions = msgspec.json.decode((config / 'extensions.json').read_bytes(), type=ExtensionCatalog)
    index = read_service_index(config / 'service-index.json')
    services = {item.serviceClass: item for item in index.services}
    for manifest in extensions.extensions:
        if manifest.runtime.kind == 'workspace':
            for name in manifest.service_classes:
                for relative in services[name].manifests.values():
                    launch = yaml.safe_load(ServicePaths.for_index(config / "service-index.json").package_path(relative, relative_to=config).read_text())['launch']
                    assert launch['command'] == 'pixi'
                    assert launch['args'][:3] == ['run', '-e', manifest.runtime.environment], name


def test_native_extension_lifecycle_persists_and_controls_catalog(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path)
    catalog = CatalogService(extension_indexes=manager.active_indexes)
    assert catalog.snapshot().services == ()
    data = tmp_path / 'data'
    (data / 'models').mkdir()
    (data / 'models' / 'user.model').write_bytes(b'user model')

    async def exercise() -> None:
        await _finish(manager, 'alpha', catalog)
        await _finish(manager, 'beta', catalog)
        assert [spec.serviceClass for spec in catalog.snapshot().services] == ['test.alpha', 'test.beta']
        await manager.set_enabled('alpha', False, catalog.refresh, lambda _name: False)
        assert [spec.serviceClass for spec in catalog.snapshot().services] == ['test.beta']
        reloaded = ExtensionManager(data, base_index=source / 'config/service-index.json')
        assert reloaded.status('alpha').state == 'disabled'
        await manager.uninstall('alpha', catalog.refresh, lambda _name: False)
        assert manager.status('alpha').state == 'available'
        await _finish(manager, 'alpha', catalog)

    asyncio.run(exercise())
    assert (data / 'models' / 'user.model').read_bytes() == b'user model'


def _shared_fixture(tmp_path: Path, *, dependencies: list[str] | None = None,
                    requires_python: str | None = None) -> tuple[ExtensionManager, Path, Path]:
    _base, source = _fixture(tmp_path)
    (source / 'pixi.toml').write_text('[environments]\nstudio-runtime = { features = [] }\nonnx = { features = [] }\n')
    (source / 'env').mkdir()
    manager = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
    _publisher, payload = _fixture(tmp_path / 'publisher', names=('player',))
    catalog = json.loads((payload / 'config/extensions.json').read_text())
    runtime: dict[str, object] = {'kind': 'shared', 'environment': 'studio-runtime'}
    if requires_python is not None:
        runtime['requiresPython'] = requires_python
    if dependencies is not None:
        runtime['dependencies'] = dependencies
    catalog['extensions'][0]['runtime'] = runtime
    (payload / 'config/extensions.json').write_text(json.dumps(catalog))
    entry = yaml.safe_load((payload / 'config/player.yml').read_text())
    entry['launch'] = {'command': 'python', 'args': ['-m', 'thirdparty_player', '--fixture-arg'], 'workdir': '..'}
    (payload / 'config/player.yml').write_text(yaml.safe_dump(entry))
    code = payload / 'python'
    code.mkdir()
    metadata_dir = code / 'thirdparty_player-1.0.dist-info'
    metadata_dir.mkdir()
    (metadata_dir / 'METADATA').write_text('Metadata-Version: 2.3\nName: thirdparty-player\nVersion: 1.0\n'
                                         'Requires-Python: >=3.14,<3.15\nRequires-Dist: msgspec>=0.1\n')
    (code / 'thirdparty_player.json').write_bytes((payload / 'player.json').read_bytes())
    (code / 'thirdparty_player.py').write_text(
        'import json, pathlib, sys\n'
        'if "--describe" in sys.argv:\n'
        '    print(pathlib.Path(__file__).with_suffix(".json").read_text())\n'
        'else:\n'
        '    print(json.dumps({"prefix": sys.prefix, "args": sys.argv[1:], "paths": sys.path}))\n'
    )
    return manager, source, payload


def test_published_shared_extension_uses_official_python_without_installing_dependencies(tmp_path: Path) -> None:
    manager, source, payload = _shared_fixture(tmp_path)
    archive = _archive({str(path.relative_to(payload)): path.read_bytes() for path in payload.rglob('*') if path.is_file()})
    request = ExtensionImportRequest(url=DownloadResponse.url, sha256=hashlib.sha256(archive).hexdigest())
    catalog = CatalogService(extension_indexes=manager.active_indexes)
    original_lock = (source / 'pixi.lock').read_bytes()

    async def exercise() -> None:
        with patch('f8studio_server.extension_artifacts.urllib.request.urlopen', return_value=DownloadResponse(archive)):
            await manager.import_package(request)
        plan = manager.install_plan('player')
        assert plan.action == 'shared'
        assert not plan.requires_network
        with patch.object(EnvironmentManager, '_pixi', side_effect=AssertionError('must not install Pixi')):
            await _finish(manager, 'player', catalog)
        registration = manager._registration('player')
        entry = yaml.safe_load((registration.parent / 'test.player.yml').read_text())['launch']
        assert Path(entry['command']) == Path(sys.executable)
        result = subprocess.run([entry['command'], *entry['args']], cwd=entry['workdir'],
                                env={**os.environ, 'PYTHONPATH': '/ambient/path/must/be/ignored'},
                                capture_output=True, text=True, check=True, timeout=30)
        identity = json.loads(result.stdout)
        assert identity['prefix'] == sys.prefix
        assert identity['args'] == ['--fixture-arg']
        assert '/ambient/path/must/be/ignored' not in identity['paths']
        assert len(catalog.snapshot().services) == 1
        assert manager.environment_statuses()[0].ready
        assert manager.preset_environments()[0].environment == 'studio-runtime'
        assert (source / 'pixi.lock').read_bytes() == original_lock
        assert not manager.environments.root.exists()
        reloaded = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
        assert reloaded.status('player').state == 'installed'
        restored_catalog = CatalogService(extension_indexes=reloaded.active_indexes)
        assert len(restored_catalog.snapshot().services) == 1
        await reloaded.set_enabled('player', False, restored_catalog.refresh, lambda _name: False)
        assert restored_catalog.snapshot().services == ()
        await reloaded.set_enabled('player', True, restored_catalog.refresh, lambda _name: False)
        await reloaded.uninstall('player', restored_catalog.refresh, lambda _name: False)
        assert not registration.parent.exists()
        assert (source / 'env').is_dir()
        await _finish(reloaded, 'player', restored_catalog)

    # Use the real test interpreter and real child processes; only its fixture location is substituted.
    with patch.object(EnvironmentManager, '_python', return_value=Path(sys.executable)):
        asyncio.run(exercise())


@pytest.mark.parametrize(('dependencies', 'requires_python', 'message'), [
    (['f8-missing-dependency-for-tests>=1'], '>=3.14', 'not installed'),
    (['msgspec>=999'], '>=3.14', 'cannot satisfy'),
    ([], '<3.10', 'does not satisfy'),
])
def test_shared_extension_rejects_unsatisfied_dependencies_without_registration(
    tmp_path: Path, dependencies: list[str], requires_python: str, message: str,
) -> None:
    manager, _source, payload = _shared_fixture(tmp_path, dependencies=dependencies, requires_python=requires_python)
    manager._add_catalog(payload, preinstalled=False)
    catalog = CatalogService(extension_indexes=manager.active_indexes)

    async def exercise() -> None:
        await manager.install('player', catalog.refresh)
        assert manager._task is not None
        await manager._task
        assert manager.status('player').state == 'failed'
        assert message in manager.status('player').detail
        assert catalog.snapshot().services == ()
        assert not manager._registration('player').exists()
        assert not manager.environments.root.exists()

    with patch.object(EnvironmentManager, '_python', return_value=Path(sys.executable)):
        asyncio.run(exercise())


@pytest.mark.parametrize('metadata_requirement', ['Requires-Python: <3.10', 'Requires-Dist: msgspec>=999'])
def test_shared_extension_checks_published_distribution_metadata(tmp_path: Path, metadata_requirement: str) -> None:
    manager, _source, payload = _shared_fixture(tmp_path, dependencies=[])
    metadata_dir = payload / 'python/thirdparty_player-1.0.dist-info'
    (metadata_dir / 'METADATA').write_text(
        f'Metadata-Version: 2.3\nName: thirdparty-player\nVersion: 1.0\n{metadata_requirement}\n',
    )
    manager._add_catalog(payload, preinstalled=False)
    with pytest.raises(InvalidRequestError):
        validate_shared_package(manager._manifest('player'), payload / 'python', _runtime_probe())


@pytest.mark.parametrize('module_name', ['json.py', 'JSON.py'])
def test_shared_extension_cannot_shadow_official_or_standard_modules(tmp_path: Path, module_name: str) -> None:
    manager, _source, payload = _shared_fixture(tmp_path)
    (payload / 'python' / module_name).write_text('raise RuntimeError("must never import this")\n')
    manager._add_catalog(payload, preinstalled=False)
    with pytest.raises(InvalidRequestError, match='would shadow'):
        validate_shared_package(manager._manifest('player'), payload / 'python', _runtime_probe())


def test_imported_catalog_cannot_replace_official_environment_definitions(tmp_path: Path) -> None:
    manager, _source, payload = _shared_fixture(tmp_path)
    (payload / 'pixi.toml').write_text('[environments]\nmalicious = { features = [] }\n')
    manager._add_catalog(payload, preinstalled=False)
    assert manager._payloads['player'].environments.official is manager.environments.official
    assert [preset.environment for preset in manager.preset_environments()] == ['studio-runtime', 'onnx']


def test_shared_extension_requires_an_available_official_runtime(tmp_path: Path) -> None:
    manager, _source, payload = _shared_fixture(tmp_path)
    manager._add_catalog(payload, preinstalled=False)
    assert not manager.preset_environments()[0].ready
    with pytest.raises(InvalidRequestError, match='not installed'):
        manager.install_plan('player')
    (payload / 'config/extensions.json').write_text((payload / 'config/extensions.json').read_text().replace(
        'studio-runtime', 'undeclared-environment',
    ))
    new_manager = ExtensionManager(tmp_path / 'other-data', base_index=manager._base_index)
    new_manager._add_catalog(payload, preinstalled=False)
    with pytest.raises(InvalidRequestError, match='unavailable'):
        new_manager.install_plan('player')


def test_shared_runtime_is_never_removed_and_lock_changes_invalidate_its_identity(tmp_path: Path) -> None:
    manager, source, payload = _shared_fixture(tmp_path)
    manager._add_catalog(payload, preinstalled=False)
    with patch.object(EnvironmentManager, '_python', return_value=Path(sys.executable)):
        plan = manager.install_plan('player')
        assert plan.environment_id is not None
        manager.environments.remove_unused(plan.environment_id, set())
        assert manager.environments.ready(plan.environment_id)
        (source / 'pixi.lock').write_text('new official dependencies')
        assert manager.install_plan('player').environment_id != plan.environment_id
        assert not manager.environments.ready(plan.environment_id)


def test_shared_onnx_environment_is_reused_and_kept_until_last_user_uninstalls(tmp_path: Path) -> None:
    _initial, source, payload = _shared_fixture(tmp_path)
    path = source / 'config/extensions.json'
    document = json.loads(path.read_text())
    document['extensions'][0]['runtime'] = {'kind': 'pixi', 'environment': 'onnx'}
    path.write_text(json.dumps(document))
    entry_path = source / 'config/alpha.yml'
    entry = yaml.safe_load(entry_path.read_text())
    entry['launch'] = {'command': 'pixi', 'args': ['run', '-e', 'onnx', 'alpha'], 'workdir': '..'}
    entry_path.write_text(yaml.safe_dump(entry))
    path = payload / 'config/extensions.json'
    document = json.loads(path.read_text())
    document['extensions'][0]['runtime'] = {'kind': 'shared', 'environment': 'onnx'}
    path.write_text(json.dumps(document))
    archive = _archive({str(path.relative_to(payload)): path.read_bytes() for path in payload.rglob('*') if path.is_file()})
    request = ExtensionImportRequest(url=DownloadResponse.url, sha256=hashlib.sha256(archive).hexdigest())
    manager = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
    catalog = CatalogService(extension_indexes=manager.active_indexes)
    pixi = source / 'pixi'
    pixi.touch()
    probe_output = msgspec.json.encode(_runtime_probe()).decode()

    def run(operation: InstallOperation, command: list[str], **_kwargs: object) -> str:
        if command[1] == 'install':
            return 'installed'
        if any(Path(argument).name == '_runtime_probe.py' for argument in command):
            return probe_output
        return (source / 'alpha.json' if operation.extension_id == 'alpha' else payload / 'player.json').read_text()

    async def exercise() -> None:
        with patch('f8studio_server.extension_artifacts.urllib.request.urlopen', return_value=DownloadResponse(archive)):
            await manager.import_package(request)
        with pytest.raises(InvalidRequestError, match='not installed'):
            manager.install_plan('player')
        await _finish(manager, 'alpha', catalog)
        official_plan = manager.install_plan('alpha')
        shared_plan = manager.install_plan('player')
        assert shared_plan.environment_id == official_plan.environment_id
        assert not shared_plan.requires_network
        await _finish(manager, 'player', catalog)
        assert manager.environment_statuses()[0].extension_ids == ('alpha', 'player')
        registration = manager._registration('player')
        launch = yaml.safe_load((registration.parent / 'test.player.yml').read_text())['launch']
        assert str(manager.environments.workspace(official_plan) / 'pixi.toml') in launch['args']
        assert '--no-install' in launch['args']
        await manager.uninstall('alpha', catalog.refresh, lambda _name: False)
        assert manager.environments.ready(shared_plan.environment_id)
        reloaded = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
        assert reloaded.status('player').state == 'installed'
        restored_catalog = CatalogService(extension_indexes=reloaded.active_indexes)
        await reloaded.uninstall('player', restored_catalog.refresh, lambda _name: False)
        assert not manager.environments.ready(shared_plan.environment_id)

    with patch.object(EnvironmentManager, '_pixi', return_value=pixi), \
         patch.object(EnvironmentManager, '_find_pixi', return_value=pixi), \
         patch.object(InstallOperation, 'run', autospec=True, side_effect=run) as commands:
        asyncio.run(exercise())
        assert sum(call.args[1][1] == 'install' for call in commands.call_args_list) == 1


def test_shared_environment_uses_official_workspace_and_never_the_publisher_workspace(tmp_path: Path) -> None:
    manager, source, payload = _shared_fixture(tmp_path)
    path = payload / 'config/extensions.json'
    path.write_text(path.read_text().replace('studio-runtime', 'onnx'))
    python = EnvironmentManager._python(source / '.pixi/envs/onnx')
    python.parent.mkdir(parents=True)
    python.touch()
    manager._add_catalog(payload, preinstalled=False)
    plan = manager.install_plan('player')
    assert plan.environment_id is not None and plan.environment_id.startswith('workspace-')
    assert manager._payloads['player'].environments.workspace(plan) == source
    assert manager.environments.ready(plan.environment_id)
    (source / 'pixi.lock').write_text('changed workspace lock')
    assert not manager.environments.ready(plan.environment_id)


def test_environment_fingerprint_reuses_file_hashes_and_invalidates_changed_inputs(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path, kind='pixi')
    original = hashlib.file_digest
    with patch('f8studio_server.environments.hashlib.file_digest', wraps=original) as hashing:
        initial = manager.install_plan('alpha').environment_id
        count = hashing.call_count
        assert manager.install_plan('beta').environment_id == initial
        assert hashing.call_count == count
        (source / 'wheels/fixture.whl').write_bytes(b'updated wheel')
        assert manager.install_plan('alpha').environment_id != initial
        assert hashing.call_count == count + 1
        (source / 'pixi.toml').write_text('[environments]\nother = { features = [] }\n')
        with pytest.raises(InvalidRequestError, match='undeclared'):
            manager.install_plan('alpha')


def test_shared_dependency_check_handles_extras_markers_and_cycles() -> None:
    probe = RuntimeProbe(python_version='3.14.3', markers={'sys_platform': 'linux', 'python_version': '3.14'},
                         modules=(), distributions={
                             'Example_Tool': InstalledDistribution(version='1.0', requires=(
                                 'helper>=2; extra == "fast"', 'missing; sys_platform == "win32"',
                             ), extras=('fast',)),
                             'helper': InstalledDistribution(version='2.0', requires=('example-tool>=1',), extras=()),
                         })
    check_dependencies(('example-tool[fast]>=1',), probe)
    check_dependencies(('missing; sys_platform == "win32"',), probe)
    with pytest.raises(InvalidRequestError, match='does not provide extras'):
        check_dependencies(('example-tool[unknown]',), probe)
    with pytest.raises(InvalidRequestError, match='URL dependencies'):
        check_dependencies(('tool @ https://publisher.example/tool.whl',), probe)


def test_preinstalled_uninstall_does_not_reappear_on_restart(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path, preinstalled=True)
    catalog = CatalogService(extension_indexes=manager.active_indexes)

    async def exercise() -> None:
        await manager.uninstall('beta', catalog.refresh, lambda _name: False)

    asyncio.run(exercise())
    reloaded = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
    assert reloaded.status('beta').state == 'available'
    assert reloaded.status('alpha').state == 'installed'
    assert (source / 'beta.py').is_file()


def test_running_services_prevent_disable_and_uninstall(tmp_path: Path) -> None:
    manager, _source = _fixture(tmp_path, preinstalled=True)

    async def exercise() -> None:
        with pytest.raises(ConflictError, match='Stop running services'):
            await manager.set_enabled('alpha', False, lambda: None, lambda _name: True)
        with pytest.raises(ConflictError, match='Stop running services'):
            await manager.uninstall('alpha', lambda: None, lambda _name: True)

    asyncio.run(exercise())
    assert manager.status('alpha').state == 'installed'


def test_refresh_failure_rolls_back_enable_and_uninstall(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path, preinstalled=True)

    def fail() -> None:
        raise ValueError('catalog failed')

    async def exercise() -> None:
        with pytest.raises(ValueError, match='catalog failed'):
            await manager.set_enabled('alpha', False, fail, lambda _name: False)
        with pytest.raises(ValueError, match='catalog failed'):
            await manager.uninstall('alpha', fail, lambda _name: False)

    asyncio.run(exercise())
    assert ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json').status('alpha').state == 'installed'


def test_install_failure_keeps_services_inactive_and_can_retry(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path)
    catalog = CatalogService(extension_indexes=manager.active_indexes)
    (source / 'alpha.py').write_text('raise RuntimeError("broken extension")\n')

    async def exercise() -> None:
        await manager.install('alpha', catalog.refresh)
        assert manager._task is not None
        await manager._task
        assert manager.status('alpha').state == 'failed'
        assert 'broken extension' in manager.status('alpha').detail
        assert catalog.snapshot().services == ()
        (source / 'alpha.py').write_text(f'import pathlib\nprint(pathlib.Path({str(source / "alpha.json")!r}).read_text())\n')
        await _finish(manager, 'alpha', catalog)

    asyncio.run(exercise())
    assert 'broken extension' in (tmp_path / 'data/extensions/logs/alpha.log').read_text()


def test_shared_locked_environment_is_reused_until_last_extension_is_removed(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path, kind='pixi')
    catalog = CatalogService(extension_indexes=manager.active_indexes)
    environment = manager.install_plan('alpha').environment_id
    assert environment == manager.install_plan('beta').environment_id
    assert environment is not None
    pixi = source / 'pixi'
    pixi.touch()

    def run(operation: InstallOperation, command: list[str], **_kwargs: object) -> str:
        if command[1] == 'install':
            return 'installed'
        return (source / f'{operation.extension_id}.json').read_text()

    async def exercise() -> None:
        with patch.object(EnvironmentManager, '_pixi', return_value=pixi), \
             patch.object(EnvironmentManager, '_find_pixi', return_value=pixi), \
             patch.object(InstallOperation, 'run', autospec=True, side_effect=run) as commands:
            await _finish(manager, 'alpha', catalog)
            assert manager.install_plan('beta').action == 'reuse'
            await _finish(manager, 'beta', catalog)
            assert sum(call.args[1][1] == 'install' for call in commands.call_args_list) == 1
            registered = read_service_index(manager.active_indexes()[0])
            launch = yaml.safe_load((manager.active_indexes()[0].parent / registered.services[0].manifests['any']).read_text())['launch']
            assert '--no-install' in launch['args']
            assert str(tmp_path / 'data/runtimes' / environment / 'pixi.toml') in launch['args']
            assert manager.environment_statuses()[0].extension_ids == ('alpha', 'beta')
            await manager.uninstall('alpha', catalog.refresh, lambda _name: False)
            assert (tmp_path / 'data/runtimes' / environment / '.ready').is_file()
            await manager.uninstall('beta', catalog.refresh, lambda _name: False)
            assert not (tmp_path / 'data/runtimes' / environment).exists()

    asyncio.run(exercise())


def test_different_locks_produce_separate_environments(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path, kind='pixi')
    first = manager.install_plan('alpha').environment_id
    (source / 'pixi.lock').write_text('incompatible version two\n')
    reloaded = ExtensionManager(tmp_path / 'other-data', base_index=source / 'config/service-index.json')
    assert first != reloaded.install_plan('alpha').environment_id


def test_cancelled_install_never_activates_and_other_operations_are_rejected(tmp_path: Path) -> None:
    manager, _source = _fixture(tmp_path)
    started = Event()

    def wait(_manifest: ExtensionManifest, operation: InstallOperation) -> None:
        started.set()
        operation.cancel_requested.wait(5)
        raise ExtensionInstallCancelled

    async def exercise() -> None:
        with patch.object(manager.environments, 'ensure', side_effect=wait):
            await manager.install('alpha', lambda: None)
            await asyncio.to_thread(started.wait, 2)
            with pytest.raises(ConflictError):
                await manager.install('beta', lambda: None)
            assert (await manager.cancel('alpha')).state == 'available'
        assert manager.active_indexes() == ()

    asyncio.run(exercise())


@pytest.mark.skipif(os.name == 'nt', reason='POSIX process group')
def test_cancel_terminates_describe_process(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path)
    marker = source / 'started'
    (source / 'alpha.py').write_text(f'import pathlib, time\npathlib.Path({str(marker)!r}).touch()\ntime.sleep(120)\n')

    async def exercise() -> None:
        await manager.install('alpha', lambda: None)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert marker.exists()
        assert (await manager.cancel('alpha')).state == 'available'

    asyncio.run(exercise())


def test_manifest_rejects_duplicate_service_owners_and_unknown_extensions(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path)
    with pytest.raises(NotFoundError):
        manager.status('../escape')
    path = source / 'config/extensions.json'
    catalog = json.loads(path.read_text())
    catalog['extensions'][1]['serviceClasses'] = ['test.alpha']
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='multiple extension owners'):
        ExtensionManager(tmp_path / 'bad-data', base_index=source / 'config/service-index.json')


def test_generic_extension_api_and_plans(tmp_path: Path) -> None:
    manager, _source = _fixture(tmp_path, preinstalled=True)
    with patch('f8studio_server.application.ExtensionManager', return_value=manager):
        studio = StudioApplication(data_dir=tmp_path / 'studio')
    app = create_app(web_dist=tmp_path, application=studio)

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            assert len((await client.get('/api/extensions')).json()) == 2
            assert (await client.post('/api/projects', json={'projectId': 'keep', 'name': 'Keep'})).status_code == 201
            node = (await client.post('/api/catalog/nodes', json={
                'kind': 'service', 'nodeId': 'beta', 'serviceClass': 'test.beta',
            })).json()
            patched = await client.post('/api/projects/keep/patch', json={
                'requestId': 'add-beta', 'expectedGraphRevision': 0, 'expectedLayoutRevision': 0,
                'operations': [{'op': 'createNode', 'node': node}],
            })
            assert patched.status_code == 200, patched.text
            saved_project = (await client.get('/api/projects/keep')).json()
            assert (await client.get('/api/extensions/alpha/plan')).json()['action'] == 'none'
            detail = (await client.get('/api/extensions/alpha/detail')).json()
            assert detail['extensionId'] == 'alpha'
            assert detail['services'][0]['describe']['service']['serviceClass'] == 'test.alpha'
            assert (await client.get('/api/environments')).json() == []
            assert (await client.get('/api/environments/presets')).json() == [{'environment': 'shared', 'ready': False}]
            assert (await client.put('/api/extensions/beta/enabled', json={'enabled': False})).json()['state'] == 'disabled'
            assert (await client.delete('/api/extensions/beta')).json()['state'] == 'available'
            assert (await client.get('/api/projects/keep')).json() == saved_project
            assert (await client.get('/api/extensions/missing/plan')).status_code == 404
            assert (await client.get('/api/extensions/missing/detail')).status_code == 404
            assert (await client.post('/api/extensions/missing/install')).status_code == 404
            assert (await client.put('/api/extensions/alpha/enabled', json={})).status_code == 422
            assert (await client.post('/api/extensions/import', json={'url': 'http://insecure.example/ext.zip', 'sha256': 'a' * 64})).status_code == 422

    try:
        asyncio.run(exercise())
    finally:
        studio.editor.close()


def test_repository_catalog_owns_every_registered_service() -> None:
    root = Path(__file__).resolve().parents[3]
    catalog = msgspec.json.decode((root / 'config/extensions.json').read_bytes(), type=ExtensionCatalog)
    index = read_service_index(root / 'config/service-index.json')
    classes = [name for extension in catalog.extensions for name in extension.service_classes]
    assert len(classes) == len(set(classes))
    assert set(classes) == {item.serviceClass for item in index.services}


class DownloadResponse(io.BytesIO):
    url = 'https://publisher.example/extension.zip'

    def geturl(self) -> str:
        return self.url


def _archive(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as package:
        for name, content in files.items():
            package.writestr(name, content)
    return output.getvalue()


def test_published_extension_is_imported_installed_and_restored_from_its_own_payload(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path)
    _publisher, payload = _fixture(tmp_path / 'publisher', preinstalled=True, names=('player',))
    entry = yaml.safe_load((payload / 'config/player.yml').read_text())
    entry['launch']['args'] = ['player.py']
    (payload / 'config/player.yml').write_text(yaml.safe_dump(entry))
    (payload / 'player.py').write_text('import pathlib\nprint(pathlib.Path(__file__).with_suffix(".json").read_text())\n')
    archive = _archive({str(path.relative_to(payload)): path.read_bytes() for path in payload.rglob('*') if path.is_file()})
    request = ExtensionImportRequest(url=DownloadResponse.url, sha256=hashlib.sha256(archive).hexdigest())
    catalog = CatalogService(extension_indexes=manager.active_indexes)

    async def exercise() -> None:
        with patch('f8studio_server.extension_artifacts.urllib.request.urlopen', return_value=DownloadResponse(archive)):
            statuses = await manager.import_package(request)
        assert len(statuses) == 3
        assert manager.status('player').state == 'available'
        assert not manager.status('player').preinstalled
        assert catalog.snapshot().services == ()
        await _finish(manager, 'player', catalog)
        assert [item.serviceClass for item in catalog.snapshot().services] == ['test.player']

    asyncio.run(exercise())
    reloaded = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
    assert reloaded.status('player').state == 'installed'
    assert len(reloaded.statuses()) == 3


@pytest.mark.parametrize('filename', ['../escape', '/absolute', 'folder\\escape', 'C:/escape'])
def test_archive_paths_cannot_escape_payload(tmp_path: Path, filename: str) -> None:
    archive = _archive({filename: b'bad'})
    request = ExtensionImportRequest(url=DownloadResponse.url, sha256=hashlib.sha256(archive).hexdigest())
    with patch('f8studio_server.extension_artifacts.urllib.request.urlopen', return_value=DownloadResponse(archive)):
        with pytest.raises(InvalidRequestError, match='archive path'):
            prepare_artifact(request, tmp_path)
    assert not (tmp_path / 'escape').exists()


def test_archive_symlinks_are_rejected(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as package:
        entry = zipfile.ZipInfo('link')
        entry.external_attr = (0o120777 << 16)
        package.writestr(entry, '../escape')
    archive = stream.getvalue()
    request = ExtensionImportRequest(url=DownloadResponse.url, sha256=hashlib.sha256(archive).hexdigest())
    with patch('f8studio_server.extension_artifacts.urllib.request.urlopen', return_value=DownloadResponse(archive)):
        with pytest.raises(InvalidRequestError, match='Unsupported extension archive file'):
            prepare_artifact(request, tmp_path)


def test_archive_checksum_failure_does_not_register_extensions(tmp_path: Path) -> None:
    manager, _source = _fixture(tmp_path)
    request = ExtensionImportRequest(url=DownloadResponse.url, sha256='0' * 64)

    async def exercise() -> None:
        with patch('f8studio_server.extension_artifacts.urllib.request.urlopen', return_value=DownloadResponse(b'bad')):
            with pytest.raises(InvalidRequestError, match='SHA-256 does not match'):
                await manager.import_package(request)

    asyncio.run(exercise())
    assert len(manager.statuses()) == 2
    assert not (tmp_path / 'data/extensions/sources.json').exists()


def test_describe_stderr_does_not_corrupt_json_output(tmp_path: Path) -> None:
    operation = InstallOperation('test', tmp_path / 'test.log')
    result = operation.run([sys.executable, '-c',
                            'import sys; print("WARNING", file=sys.stderr); print("{}")'],
                           cwd=tmp_path, timeout=5)
    assert json.loads(result) == {}
    assert 'WARNING' in (tmp_path / 'test.log').read_text()


def test_broken_restored_metadata_is_disabled_without_crashing_studio(tmp_path: Path) -> None:
    _manager, source = _fixture(tmp_path, preinstalled=True)
    (source / 'alpha.json').write_text('{broken')
    restored = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
    assert restored.status('alpha').state == 'failed'
    catalog = CatalogService(extension_indexes=restored.active_indexes)
    assert [item.serviceClass for item in catalog.snapshot().services] == ['test.beta']


def test_undeclared_environment_is_rejected_before_installation(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path, kind='pixi')
    (source / 'pixi.toml').write_text('[workspace]\nname = "fixture"\n')
    with pytest.raises(InvalidRequestError, match='undeclared environment'):
        manager.install_plan('alpha')


def test_starting_service_blocks_uninstall_and_disabled_service_cannot_start(tmp_path: Path) -> None:
    manager, _source = _fixture(tmp_path, preinstalled=True)
    catalog = CatalogService(extension_indexes=manager.active_indexes)
    processes = ManagedServiceProcesses(catalog=catalog, runtime_config=RuntimeConfig(),
                                        events=EventJournal(server_epoch='test'), service_enabled=manager.service_enabled)
    started, release = Event(), Event()

    def start(*_args: object, **_kwargs: object) -> None:
        started.set()
        release.wait(5)

    async def exercise() -> None:
        with patch.object(processes._manager, 'start', side_effect=start), \
             patch.object(processes._manager, 'is_running', return_value=False):
            task = asyncio.create_task(processes.start('alpha', service_class='test.alpha'))
            try:
                assert await asyncio.to_thread(started.wait, 2)
                with pytest.raises(ConflictError, match='Stop running services'):
                    await manager.uninstall('alpha', catalog.refresh, processes.is_class_running)
            finally:
                release.set()
                await task
        await manager.set_enabled('alpha', False, catalog.refresh, processes.is_class_running)
        with patch.object(processes._manager, 'start') as launch:
            with pytest.raises(ConflictError, match='disabled or uninstalled'):
                await processes.start('alpha', service_class='test.alpha')
            launch.assert_not_called()

    asyncio.run(exercise())


def test_cancelled_service_start_waits_for_thread_and_stops_spawned_process(tmp_path: Path) -> None:
    manager, _source = _fixture(tmp_path, preinstalled=True)
    processes = ManagedServiceProcesses(catalog=CatalogService(extension_indexes=manager.active_indexes),
                                        runtime_config=RuntimeConfig(), events=EventJournal(server_epoch='test'))
    started, release = Event(), Event()

    def start(*_args: object, **_kwargs: object) -> None:
        started.set()
        release.wait(5)

    async def exercise() -> None:
        with patch.object(processes._manager, 'start', side_effect=start), \
             patch.object(processes._manager, 'stop', return_value=True) as stop:
            task = asyncio.create_task(processes.start('alpha', service_class='test.alpha'))
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            stop.assert_called_once_with('alpha')
            assert not processes._starting_classes

    asyncio.run(exercise())


def test_state_write_failure_does_not_change_enabled_state(tmp_path: Path) -> None:
    manager, _source = _fixture(tmp_path, preinstalled=True)

    async def exercise() -> None:
        with patch.object(manager, '_save_records', side_effect=OSError('disk full')):
            with pytest.raises(OSError, match='disk full'):
                await manager.set_enabled('alpha', False, lambda: None, lambda _name: False)

    asyncio.run(exercise())
    assert manager.status('alpha').state == 'installed'
    assert manager.service_enabled('test.alpha')


def test_corrupt_user_state_warns_without_crashing_or_rewriting_files(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _manager, source = _fixture(tmp_path, preinstalled=True)
    state = tmp_path / 'data/extensions/state.json'
    sources = tmp_path / 'data/extensions/sources.json'
    state.write_text('{broken')
    sources.write_text('{broken')
    reloaded = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
    assert reloaded.status('alpha').state == 'installed'
    assert state.read_text() == '{broken'
    assert sources.read_text() == '{broken'
    assert 'Cannot read extension installation state' in caplog.text
    assert 'Cannot read extension sources' in caplog.text


def test_wrong_launch_environment_is_rejected_before_expensive_install(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path, kind='pixi')
    path = source / 'config/alpha.yml'
    entry = yaml.safe_load(path.read_text())
    entry['launch']['args'][2] = 'wrong'
    path.write_text(yaml.safe_dump(entry))

    async def exercise() -> None:
        with patch.object(manager.environments, 'ensure') as prepare:
            with pytest.raises(InvalidRequestError, match='declared extension environment'):
                await manager.install('alpha', lambda: None)
            prepare.assert_not_called()

    asyncio.run(exercise())
    assert manager.status('alpha').state == 'available'


def test_extension_detail_previews_without_installing_and_reports_missing_descriptions(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path)
    detail = manager.detail('alpha')
    assert detail.services[0].describe is not None
    assert detail.services[0].describe.service.serviceClass == 'test.alpha'
    assert manager.status('alpha').state == 'available'
    assert manager.active_indexes() == ()
    (source / 'alpha.json').unlink()
    assert manager.detail('alpha').services[0].describe is None
    (source / 'alpha.json').write_text('{"service":{"serviceClass":"wrong.class","label":"Wrong"}}')
    with pytest.raises(InvalidRequestError, match='class mismatch'):
        manager.detail('alpha')


def test_workspace_inventory_groups_stale_and_current_records_by_environment(tmp_path: Path) -> None:
    manager, source = _fixture(tmp_path, kind='pixi')
    path = source / 'config/extensions.json'
    path.write_text(path.read_text().replace('"pixi"', '"workspace"'))
    records = {name: {'version': '1.0.0', 'installed': True, 'enabled': True,
                      'environmentId': 'workspace-legacy-shared-old'} for name in ('alpha', 'beta')}
    manager._state_path.write_text(json.dumps(records))
    with patch.object(EnvironmentManager, '_python', return_value=Path(sys.executable)):
        restored = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
        inventory = restored.environment_statuses()
        assert len(inventory) == 1
        assert inventory[0].name == 'shared'
        assert inventory[0].state == 'changed'
        assert inventory[0].ready
        assert inventory[0].extension_ids == ('alpha', 'beta')
        assert inventory[0].service_classes == ('test.alpha', 'test.beta')


def test_explicit_runtime_selection_survives_restart_and_checks_dependencies(tmp_path: Path) -> None:
    manager, source, payload = _shared_fixture(tmp_path)
    archive = _archive({str(path.relative_to(payload)): path.read_bytes() for path in payload.rglob('*') if path.is_file()})
    request = ExtensionImportRequest(url=DownloadResponse.url, sha256=hashlib.sha256(archive).hexdigest())
    catalog = CatalogService(extension_indexes=manager.active_indexes)
    async def exercise() -> None:
        with patch('f8studio_server.extension_artifacts.urllib.request.urlopen', return_value=DownloadResponse(archive)):
            await manager.import_package(request)
        environment = manager.environment_statuses()[0]
        await manager.select_runtime('player', environment.environment_id)
        await _finish(manager, 'player', catalog)
        assert manager.status('player').runtime_selectable
        with pytest.raises(ConflictError, match='Uninstall'):
            await manager.select_runtime('player', None)
        reloaded = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
        assert reloaded.status('player').state == 'installed'
        assert reloaded.status('player').runtime_environment == environment.environment_id
        await reloaded.close()
        await manager.close()
    with patch.object(EnvironmentManager, '_python', return_value=Path(sys.executable)):
        asyncio.run(exercise())


def test_independent_runtime_migrates_matching_records_and_reconciles_changed_definition(tmp_path: Path) -> None:
    from f8studio_server.environment_definitions import selected_manifest, write_manifest
    _unused, root = _fixture(tmp_path, kind='pixi', names=('alpha',))
    (root / 'pixi.lock').write_text(yaml.safe_dump({'version': 6,
        'environments': {'shared': {'packages': {'linux-64': []}}}, 'packages': []}))
    catalog_path = root / 'config/extensions.json'
    catalog = json.loads(catalog_path.read_text())
    catalog['extensions'][0]['runtime']['kind'] = 'workspace'
    catalog['preinstalled'] = ['alpha']
    catalog_path.write_text(json.dumps(catalog))
    python = root / '.pixi/envs/shared' / ('python.exe' if os.name == 'nt' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.write_bytes(b'fixture')
    data = tmp_path / 'data'
    old = ExtensionManager(data, base_index=root / 'config/service-index.json')
    old_identifier = old.status('alpha').environment_id
    definition = root / 'runtimes/shared'
    definition.mkdir(parents=True)
    write_manifest(definition / 'pixi.toml', selected_manifest(root, 'shared'))
    lock_bytes = (root / 'pixi.lock').read_bytes()
    (definition / 'pixi.lock').write_bytes(lock_bytes)
    (root / 'config/runtime-environments.json').write_text(json.dumps({'schemaVersion': 'f8runtimeCatalog/1',
        'runtimes': [{'runtimeId': 'shared', 'manifest': '${F8_PACKAGE_ROOT}/runtimes/shared/pixi.toml', 'developmentEnvironment': 'shared'}]}))
    migrated = ExtensionManager(data, base_index=root / 'config/service-index.json')
    identifier = migrated.status('alpha').environment_id
    assert identifier != old_identifier
    assert migrated.status('alpha').state == 'installed'
    assert migrated.environment_statuses()[0].extension_ids == ('alpha',)
    assert migrated.environment_statuses()[0].ready
    registration = migrated._registration('alpha')
    previous = registration.read_bytes()
    (definition / 'pixi.toml').write_text((definition / 'pixi.toml').read_text() + '\n[dependencies]\npython="3.12.*"\n')
    changed = ExtensionManager(data, base_index=root / 'config/service-index.json')
    assert changed.status('alpha').state == 'installed'
    assert changed.status('alpha').environment_id == identifier
    assert registration.read_bytes() == previous
    status = changed.environment_statuses()[0]
    assert status.state == 'changed'
    assert status.extension_ids == ('alpha',)

    def run(_operation: InstallOperation, command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
        assert Path(command[command.index('--manifest-path') + 1]) == definition / 'pixi.toml'
        assert cwd == definition
        installed_python = definition / '.pixi/envs/shared' / ('python.exe' if os.name == 'nt' else 'bin/python')
        installed_python.parent.mkdir(parents=True)
        installed_python.write_bytes(b'prepared')
        return ''

    async def prepare() -> None:
        await changed.prepare_environment(status.environment_id, lambda: None, lambda _name: False)
        assert changed._environment_refresh_task is not None
        await changed._environment_refresh_task
        assert changed.status('alpha').state == 'installed'
        assert changed.status('alpha').environment_id == status.environment_id
        assert changed.environment_statuses()[0].state == 'ready'
        assert str(definition) in registration.with_name('test.alpha.yml').read_text()

    with patch.object(EnvironmentManager, '_pixi', return_value=Path('/fixture/pixi')), patch.object(InstallOperation, 'run', autospec=True, side_effect=run):
        asyncio.run(prepare())


def test_independent_bundled_runtime_keeps_offline_prefix_and_lists_consumers(tmp_path: Path) -> None:
    from f8studio_server.environment_definitions import selected_manifest, write_manifest
    _unused, root = _fixture(tmp_path, names=('alpha',))
    (root / 'pixi.toml').write_text('[workspace]\nname="offline"\n[pypi-dependencies]\nfixture={path="wheels/fixture.whl"}\n[environments]\nstudio-runtime={features=[]}\n')
    locked = {'version': 6, 'environments': {'studio-runtime': {'packages': {'linux-64': [{'pypi': 'wheels/fixture.whl'}]}}},
              'packages': [{'pypi': 'wheels/fixture.whl', 'name': 'fixture'}]}
    (root / 'pixi.lock').write_text(yaml.safe_dump(locked))
    python = root / 'env' / ('python.exe' if os.name == 'nt' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.write_bytes(b'offline interpreter')
    catalog_path = root / 'config/extensions.json'
    catalog = json.loads(catalog_path.read_text())
    catalog['extensions'][0]['runtime'] = {'kind': 'bundled'}
    catalog['preinstalled'] = ['alpha']
    catalog_path.write_text(json.dumps(catalog))
    data = tmp_path / 'data'
    old = ExtensionManager(data, base_index=root / 'config/service-index.json')
    previous = old.status('alpha').environment_id
    independent = root / 'runtimes/studio-runtime'
    independent.mkdir(parents=True)
    definition = selected_manifest(root, 'studio-runtime')
    definition['pypi-dependencies']['fixture']['path'] = '../../wheels/fixture.whl'
    write_manifest(independent / 'pixi.toml', definition)
    locked['environments']['studio-runtime']['packages']['linux-64'][0]['pypi'] = '../../wheels/fixture.whl'
    locked['packages'][0]['pypi'] = '../../wheels/fixture.whl'
    (independent / 'pixi.lock').write_text(yaml.safe_dump(locked))
    (root / 'config/runtime-environments.json').write_text(json.dumps({'schemaVersion': 'f8runtimeCatalog/1',
        'runtimes': [{'runtimeId': 'studio-runtime', 'manifest': '${F8_PACKAGE_ROOT}/runtimes/studio-runtime/pixi.toml'}]}))
    manager = ExtensionManager(data, base_index=root / 'config/service-index.json')
    current = manager.status('alpha')
    assert current.state == 'installed'
    assert current.environment_id != previous
    environment = manager.environment_statuses()[0]
    assert environment.runtime_kind == 'bundled'
    assert environment.state == 'ready'
    assert environment.extension_ids == ('alpha',)
    detail = manager.runtime_registry.detail(environment.environment_id)
    assert detail.storage_path == str(root / 'env')
    assert detail.definition_path == str(independent / 'pixi.toml')


def test_imported_pixi_extension_can_declare_multiple_environments(tmp_path: Path) -> None:
    manager, _source = _fixture(tmp_path)
    _publisher, payload = _fixture(tmp_path / 'publisher', kind='pixi', names=('player',))
    catalog_path = payload / 'config/extensions.json'
    catalog_path.write_text(catalog_path.read_text().replace('shared', 'player'))
    manifest_path = payload / 'pixi.toml'
    manifest_path.write_text(manifest_path.read_text().replace('shared', 'player'))
    with manifest_path.open('a') as destination:
        destination.write('second = { features = [] }\n')
    manager._add_catalog(payload, preinstalled=False)
    assert 'player' in {item.extension_id for item in manager.statuses()}
    assert manager._payloads['player'].environments.preset_names() == ('player', 'second')
    assert {source.name for source in manager.runtime_registry.sources.values()
            if source.source == 'package'} == {'player', 'second'}


def test_identically_named_package_environments_keep_independent_prefixes_and_share_cache(tmp_path: Path) -> None:
    manager, _source = _fixture(tmp_path)
    _first, first = _fixture(tmp_path / 'first', kind='pixi', names=('one',))
    _second, second = _fixture(tmp_path / 'second', kind='pixi', names=('two',))
    manager._add_catalog(first, preinstalled=False)
    manager._add_catalog(second, preinstalled=False)
    first_plan = manager.install_plan('one')
    second_plan = manager.install_plan('two')
    assert first_plan.environment_id != second_plan.environment_id
    first_manager = manager._payloads['one'].environments
    second_manager = manager._payloads['two'].environments
    assert first_manager.workspace(first_plan) != second_manager.workspace(second_plan)
    assert first_manager.install_environment()['PIXI_CACHE_DIR'] == second_manager.install_environment()['PIXI_CACHE_DIR']
    assert first_manager.install_environment()['UV_CACHE_DIR'] == second_manager.install_environment()['UV_CACHE_DIR']
