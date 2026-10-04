from __future__ import annotations

import json
import os
from pathlib import Path
import tomllib

import pytest
import yaml

from f8studio_server.environment_definitions import (
    materialize_locked_environment, read_manifest, selected_lock, selected_manifest, write_manifest,
)
from f8studio_server.environments import EnvironmentManager
from f8studio_server.errors import InvalidRequestError
from f8studio_server.extension_models import EnvironmentCreateRequest
from f8studio_server.runtime_registry import RuntimeRegistry
from f8studio_server.runtime_sources import equivalent_environment, read_runtime_sources


def split_workspace(root: Path) -> Path:
    root.mkdir()
    (root / 'config').mkdir()
    (root / 'package').mkdir()
    (root / 'package/module.py').write_text('VALUE = 42')
    (root / 'package/pyproject.toml').write_text('[project]\nname="local-package"\nversion="1.0"\n')
    (root / 'pixi.toml').write_text(
        '[workspace]\nname="development"\nchannels=["conda-forge"]\nplatforms=["linux-64"]\n'
        '[feature.base.dependencies]\npython="3.12.*"\n'
        '[feature.base.pypi-dependencies]\nlocal-package={path="package",editable=true}\n'
        '[feature.other.dependencies]\npython="3.13.*"\n'
        '[environments]\nbase={features=["base"],no-default-feature=true}\n'
        'other={features=["other"],no-default-feature=true}\n'
    )
    (root / 'pixi.lock').write_text(yaml.safe_dump({'version': 6, 'environments': {
        'base': {'packages': {'linux-64': [{'conda': 'https://example/python-3.12.9-build_0.conda'}, {'pypi': 'package'}]}},
        'other': {'packages': {'linux-64': [{'conda': 'https://example/python-3.13.1-build_0.conda'}]}},
    }, 'packages': [{'conda': 'https://example/python-3.12.9-build_0.conda'}, {'pypi': 'package', 'name': 'local-package'},
                    {'conda': 'https://example/python-3.13.1-build_0.conda'}]}, sort_keys=False))
    catalog = []
    for name in ('base', 'other'):
        destination = root / 'runtimes' / name
        destination.mkdir(parents=True)
        manifest = selected_manifest(root, name)
        if name == 'base':
            manifest['feature']['base']['pypi-dependencies']['local-package']['path'] = '../../package'
        write_manifest(destination / 'pixi.toml', manifest)
        lock = selected_lock(root, name)
        if name == 'base':
            lock['packages'][1]['pypi'] = '../../package'
            lock['environments']['base']['packages']['linux-64'][1]['pypi'] = '../../package'
        (destination / 'pixi.lock').write_text(yaml.safe_dump(lock, sort_keys=False))
        catalog.append({'runtimeId': name, 'manifest': '${F8_PACKAGE_ROOT}/runtimes/' + name + '/pixi.toml',
                        'developmentEnvironment': name})
    (root / 'config/runtime-environments.json').write_text(json.dumps({'schemaVersion': 'f8runtimeCatalog/1', 'runtimes': catalog}))
    return root


def test_official_definition_identity_and_detail_are_independent_of_development_workspace(tmp_path: Path) -> None:
    root = split_workspace(tmp_path / 'repo')
    manager = EnvironmentManager(tmp_path / 'data', root)
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    registry.add_preset('base')
    identifier = next(iter(registry.sources))
    detail = registry.detail(identifier)
    assert detail.definition_path == str(root / 'runtimes/base/pixi.toml')
    assert detail.source_environment == 'base'
    definition = tomllib.loads(detail.manifest)
    assert set(definition['environments']) == {'base'}
    assert set(definition['feature']) == {'base'}
    assert manager.preset_names() == ('base', 'other')
    original = manager.identity('base')
    (root / 'pixi.toml').write_text((root / 'pixi.toml').read_text().replace('3.13.*', '3.14.*'))
    assert manager.identity('base') == original
    source = root / 'runtimes/base/pixi.toml'
    source.write_text(source.read_text().replace('3.12.*', '3.11.*'))
    assert manager.identity('base') != original


def test_matching_development_prefix_is_reused_and_legacy_selection_resolves(tmp_path: Path) -> None:
    root = split_workspace(tmp_path / 'repo')
    python = root / '.pixi/envs/base' / ('python.exe' if os.name == 'nt' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.write_bytes(b'fixture')
    manager = EnvironmentManager(tmp_path / 'data', root)
    owner = manager.for_environment('base')
    assert equivalent_environment(root, owner.source_root, 'base')
    legacy = owner.legacy_environment_id('base')
    assert legacy is not None
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    registry.add_preset('base')
    source = registry.source(legacy)
    assert source.target.manager is owner
    assert registry.status(legacy).ready
    assert manager.workspace(source.target.plan) == root
    command, args = manager.launch(source.target.plan, 'base')
    assert Path(command).is_file()
    assert args[args.index('--manifest-path') + 1] == str(root / 'pixi.toml')
    assert source.target.plan.environment_id != legacy
    (root / 'runtimes/base/pixi.toml').write_text((root / 'runtimes/base/pixi.toml').read_text().replace('3.12.*', '3.11.*'))
    assert not owner.can_reuse_development_environment('base')
    assert not owner.workspace_python_exists('base')


def test_derived_environment_snapshots_repo_dependencies_outside_runtime_directory(tmp_path: Path) -> None:
    root = split_workspace(tmp_path / 'repo')
    manager = EnvironmentManager(tmp_path / 'data', root)
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    registry.add_preset('base')
    item = registry.create(EnvironmentCreateRequest(name='derived', base_environment_id=next(iter(registry.sources))))
    definition = read_manifest(registry.definitions / item.environment_id)
    dependency = definition['feature']['base']['pypi-dependencies']['local-package']
    path = registry.definitions / item.environment_id / dependency['path']
    assert path.is_relative_to(registry.definitions / item.environment_id)
    assert (path / 'module.py').read_text() == 'VALUE = 42'
    assert dependency['editable'] is False


def test_managed_materialization_retargets_manifest_and_lock_local_inputs(tmp_path: Path) -> None:
    root = split_workspace(tmp_path / 'repo')
    workspace = tmp_path / 'installed'
    workspace.mkdir()
    source = root / 'runtimes/base'
    original_lock = yaml.safe_load((source / 'pixi.lock').read_text())
    original_lock['environments']['default'] = {'packages': {'linux-64': []}}
    (source / 'pixi.lock').write_text(yaml.safe_dump(original_lock, sort_keys=False))
    original = (source / 'pixi.lock').read_bytes()
    materialize_locked_environment(source, 'base', workspace, root)
    definition = read_manifest(workspace)
    dependency = definition['feature']['base']['pypi-dependencies']['local-package']
    assert dependency['editable'] is True
    assert (workspace / dependency['path'] / 'module.py').is_file()
    lock = yaml.safe_load((workspace / 'pixi.lock').read_text())
    reference = dependency['path']
    assert 'default' in lock['environments']
    assert lock['packages'][1]['pypi'] == reference
    assert lock['environments']['base']['packages']['linux-64'][1]['pypi'] == reference
    assert (source / 'pixi.lock').read_bytes() == original


@pytest.mark.parametrize('reference', ['../outside/pixi.toml', '${F8_PACKAGE_ROOT}/../outside/pixi.toml'])
def test_runtime_catalog_rejects_implicit_and_escaping_references(tmp_path: Path, reference: str) -> None:
    root = split_workspace(tmp_path / 'repo')
    path = root / 'config/runtime-environments.json'
    catalog = json.loads(path.read_text())
    catalog['runtimes'][0]['manifest'] = reference
    path.write_text(json.dumps(catalog))
    with pytest.raises(InvalidRequestError, match='explicit|unsafe'):
        read_runtime_sources(root)


def test_runtime_catalog_accepts_multiple_environments_and_rejects_missing_locks(tmp_path: Path) -> None:
    root = split_workspace(tmp_path / 'repo')
    manifest = root / 'runtimes/base/pixi.toml'
    original = manifest.read_text()
    manifest.write_text(original + '\nextra={features=[]}\n')
    assert read_runtime_sources(root)['base'][0] == manifest.parent
    manifest.write_text(original)
    manifest.with_name('pixi.lock').unlink()
    with pytest.raises(InvalidRequestError, match='independent pixi.lock'):
        read_runtime_sources(root)


def test_official_revision_history_survives_restart_and_routes_new_workspace_plans(tmp_path: Path) -> None:
    root = split_workspace(tmp_path / 'repo')
    data = tmp_path / 'data'
    manager = EnvironmentManager(data, root)
    registry = RuntimeRegistry(data, manager)
    original = next(item for item in registry.sources if registry.sources[item].name == 'base')
    definition = root / 'runtimes/base/pixi.toml'
    definition.write_text(definition.read_text().replace('3.12.*', '3.11.*'))
    new_plan = manager.workspace_plan('tool', 'base')
    assert manager.workspace(new_plan) == root / 'runtimes/base'
    restarted = RuntimeRegistry(data, EnvironmentManager(data, root))
    assert restarted.canonical_id(original) != original
    assert restarted.source(original).target.plan.environment_id == new_plan.environment_id
    assert restarted.detail(original).environment_id == new_plan.environment_id
    assert set(read_manifest(root)['environments']) == {'base', 'other'}


def test_preparation_updates_legacy_aliases_and_persists_official_revision_history(tmp_path: Path) -> None:
    import asyncio
    from unittest.mock import patch
    from f8studio_server.extension_operation import InstallOperation
    root = split_workspace(tmp_path / 'repo')
    data = tmp_path / 'data'
    manager = EnvironmentManager(data, root)
    registry = RuntimeRegistry(data, manager)
    original = next(key for key, value in registry.sources.items() if value.name == 'base')
    legacy = manager.for_environment('base').legacy_environment_id('base')
    assert legacy is not None
    definition = root / 'runtimes/base/pixi.toml'
    definition.write_text(definition.read_text().replace('3.12.*', '3.11.*'))

    def run(_operation: InstallOperation, _command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
        python = cwd / '.pixi/envs/base' / ('python.exe' if os.name == 'nt' else 'bin/python')
        python.parent.mkdir(parents=True)
        python.write_bytes(b'fixture')
        return ''

    async def prepare() -> None:
        await registry.prepare(original)
        assert registry.preparation_task is not None
        await registry.preparation_task

    with patch.object(EnvironmentManager, '_pixi', return_value=Path('/fixture/pixi')), patch.object(InstallOperation, 'run', autospec=True, side_effect=run):
        asyncio.run(prepare())
    prepared = registry.canonical_id(original)
    assert prepared != original
    assert registry.source(legacy).target.plan.environment_id == prepared
    history = json.loads((registry.definitions / 'official-sources.json').read_text())
    assert history[prepared] == 'base'
    restarted = RuntimeRegistry(data, EnvironmentManager(data, root))
    assert restarted.canonical_id(original) == prepared
    assert restarted.status(prepared).ready
