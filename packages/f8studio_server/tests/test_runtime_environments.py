from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from f8studio_server.environment_definitions import environment_identity, read_manifest
from f8studio_server.environments import EnvironmentManager
from f8studio_server.errors import ConflictError, InvalidRequestError
from f8studio_server.extension_models import EnvironmentCreateRequest
from f8studio_server.extension_operation import InstallOperation
from f8studio_server.runtime_registry import RuntimeRegistry, directory_usage


def source(root: Path) -> EnvironmentManager:
    root.mkdir(parents=True)
    (root / 'pixi.toml').write_text('[workspace]\nname="base"\nchannels=["conda-forge"]\nplatforms=["linux-64"]\n'
        '[feature.base.dependencies]\npython="3.12.*"\nnumpy=">=1.26,<3"\n'
        '[feature.other.dependencies]\npython="3.12.*"\n'
        '[environments]\nbase={features=["base"],no-default-feature=true}\nother={features=["other"],no-default-feature=true}\n')
    write_lock(root, 'base', numpy='1.26.4', other=True)
    return EnvironmentManager(root.parent / 'data', root)


def write_lock(root: Path, environment: str, *, numpy: str = '1.26.4', other: bool = False) -> None:
    python = 'https://conda.anaconda.org/conda-forge/linux-64/python-3.12.9-build_0.conda'
    array = f'https://conda.anaconda.org/conda-forge/linux-64/numpy-{numpy}-build_0.conda'
    envs = {environment: {'channels': [{'url': 'https://conda.anaconda.org/conda-forge/'}],
                          'packages': {'linux-64': [{'conda': python}, {'conda': array}]}}}
    if other:
        envs['other'] = {'packages': {'linux-64': [{'conda': python}]}}
    (root / 'pixi.lock').write_text(yaml.safe_dump({'version': 6, 'environments': envs, 'packages': [
        {'conda': python, 'sha256': 'a' * 64}, {'conda': array, 'sha256': 'b' * 64},
    ]}))


def test_environment_identity_ignores_unrelated_features_and_locked_packages(tmp_path: Path) -> None:
    manager = source(tmp_path / 'base')
    initial = environment_identity(manager.source_root, 'base')
    manifest = manager.source_root / 'pixi.toml'
    manifest.write_text(manifest.read_text().replace('[feature.other.dependencies]\npython="3.12.*"', '[feature.other.dependencies]\npython="3.13.*"'))
    lock = yaml.safe_load((manager.source_root / 'pixi.lock').read_text())
    lock['environments']['other']['packages']['linux-64'].append({'conda': 'other-package'})
    lock['packages'].append({'conda': 'other-package', 'sha256': 'new'})
    (manager.source_root / 'pixi.lock').write_text(yaml.safe_dump(lock))
    assert environment_identity(manager.source_root, 'base') == initial
    write_lock(manager.source_root, 'base', numpy='2.0.0', other=True)
    assert environment_identity(manager.source_root, 'base') != initial


def test_create_revision_is_lazy_and_preserves_or_adjusts_base_constraints(tmp_path: Path) -> None:
    manager = source(tmp_path / 'base')
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    registry.add_preset('base')
    base = next(iter(registry.sources))
    original = (manager.source_root / 'pixi.toml').read_bytes()
    preserved = registry.create(EnvironmentCreateRequest(name='new-tool', base_environment_id=base,
                                                        pypi_dependencies=('requests>=2,<3',)))
    assert preserved.state == 'declared' and not preserved.ready
    snapshot = read_manifest(registry.definitions / preserved.environment_id)
    assert 'f8-base-lock' in snapshot['feature']
    assert 'solve-group' not in snapshot['environments']['runtime']
    assert not (registry.definitions / preserved.environment_id / '.pixi').exists()
    adjusted = registry.create(EnvironmentCreateRequest(name='new-tool', base_environment_id=base, policy='adjust',
                                                       conda_dependencies=('numpy >=2,<3',)))
    changed = read_manifest(registry.definitions / adjusted.environment_id)
    assert 'numpy' not in changed['feature']['base']['dependencies']
    assert changed['feature']['f8-additions']['dependencies']['numpy'] == '>=2,<3'
    assert (manager.source_root / 'pixi.toml').read_bytes() == original
    assert registry.create(EnvironmentCreateRequest(name='new-tool', base_environment_id=base,
                                                   pypi_dependencies=('requests>=2,<3',))).environment_id == preserved.environment_id
    restored = RuntimeRegistry(tmp_path / 'data', manager)
    assert restored.status(preserved.environment_id).state == 'declared'


def test_revisions_with_identical_resolved_packages_share_storage_and_retention(tmp_path: Path) -> None:
    manager = source(tmp_path / 'base')
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    first = registry.create(EnvironmentCreateRequest(name='tool-a'))
    second = registry.create(EnvironmentCreateRequest(name='tool-b'))
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    def run(operation: InstallOperation, command: list[str], *, cwd: Path, env: dict[str, str] | None = None, timeout: float | None = None) -> str:
        calls.append((command, env))
        if command[1] == 'lock':
            write_lock(cwd, 'runtime')
        else:
            python = EnvironmentManager._python(cwd / '.pixi/envs/runtime')
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()
        return ''
    async def exercise() -> None:
        for identifier in (first.environment_id, second.environment_id):
            await registry.prepare(identifier)
            assert registry.preparation_task is not None
            await registry.preparation_task
            assert registry.status(identifier).ready
        revisions = registry.revisions()
        assert revisions[0].resolved_id == revisions[1].resolved_id
        workspace = registry.source(first.environment_id).target.manager.source_root
        assert len(list((registry.storage / 'runtimes').iterdir())) == 1
        assert all(env is not None and Path(env['PIXI_CACHE_DIR']).parent == registry.storage for _, env in calls)
        registry.retain(first.environment_id, True)
        with pytest.raises(ConflictError, match='Unpin'):
            registry.remove(first.environment_id, set())
        registry.retain(first.environment_id, False)
        with pytest.raises(ConflictError, match='rebind'):
            registry.remove(first.environment_id, {first.environment_id})
        registry.remove(first.environment_id, set())
        assert workspace.is_dir()
        registry.remove(second.environment_id, set())
        assert not workspace.exists()
        assert (registry.storage / 'package-cache').is_dir()
    with patch.object(EnvironmentManager, '_pixi', return_value=Path('/fixture/pixi')), patch.object(InstallOperation, 'run', autospec=True, side_effect=run):
        asyncio.run(exercise())


def test_failed_preparation_is_actionable_and_can_be_retried(tmp_path: Path) -> None:
    manager = source(tmp_path / 'base')
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    item = registry.create(EnvironmentCreateRequest(name='conflicting'))
    async def exercise() -> None:
        await registry.prepare(item.environment_id)
        assert registry.preparation_task is not None
        await registry.preparation_task
        assert registry.status(item.environment_id).state == 'failed'
        assert 'Incompatible numpy constraints' in registry.status(item.environment_id).detail
        assert not registry.status(item.environment_id).ready
        registry.remove(item.environment_id, set())
    with patch.object(EnvironmentManager, '_pixi', return_value=Path('/fixture/pixi')), patch.object(InstallOperation, 'run', side_effect=RuntimeError('Incompatible numpy constraints')):
        asyncio.run(exercise())


def test_child_snapshot_and_storage_guards(tmp_path: Path) -> None:
    manager = source(tmp_path / 'base')
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    registry.add_preset('base')
    base = next(iter(registry.sources))
    created = registry.create(EnvironmentCreateRequest(name='tool', base_environment_id=base))
    registry.set_storage(str(tmp_path / 'new-volume'))
    assert registry.storage == tmp_path / 'new-volume'
    assert manager.root == tmp_path / 'new-volume/runtimes'
    restored = EnvironmentManager(tmp_path / 'data', manager.source_root)
    assert restored.root == manager.root
    with pytest.raises(InvalidRequestError, match='absolute'):
        registry.set_storage('relative')
    with pytest.raises(InvalidRequestError, match='start'):
        registry.create(EnvironmentCreateRequest(name='../escape'))
    registry.remove(created.environment_id, set())


def test_hardlink_usage_does_not_sum_shared_files_as_exclusive(tmp_path: Path) -> None:
    root = tmp_path / 'env'
    root.mkdir()
    cached = tmp_path / 'cached'
    cached.write_bytes(b'x' * 128)
    os.link(cached, root / 'package')
    os.link(cached, root / 'second-name')
    (root / 'private').write_bytes(b'x' * 32)
    usage = directory_usage(root)
    assert usage.logical_bytes == 288
    assert usage.unique_file_bytes == 160
    assert usage.shared_link_bytes == 128
    assert usage.exclusive_file_bytes == 32


def test_generated_manifest_is_accepted_by_pixi_without_solving(tmp_path: Path) -> None:
    import shutil
    import subprocess
    pixi = shutil.which('pixi')
    if pixi is None:
        pytest.skip('Pixi CLI is unavailable')
    manager = source(tmp_path / 'base')
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    registry.add_preset('base')
    base = next(iter(registry.sources))
    for request in (EnvironmentCreateRequest(name='independent'),
                    EnvironmentCreateRequest(name='preserve', base_environment_id=base),
                    EnvironmentCreateRequest(name='adjust', base_environment_id=base, policy='adjust', conda_dependencies=('numpy >=2',))):
        created = registry.create(request)
        result = subprocess.run([pixi, 'workspace', 'environment', 'list', '--manifest-path',
                                 str(registry.definitions / created.environment_id / 'pixi.toml')],
                                capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        assert 'runtime' in result.stdout


def test_snapshot_materializes_local_dependencies_and_excludes_toolchain_caches(tmp_path: Path) -> None:
    manager = source(tmp_path / 'base')
    project = manager.source_root / 'local-package'
    (project / '.pixi').mkdir(parents=True)
    (project / '.sdk').mkdir()
    (project / '.pixi/large-cache').write_bytes(b'x' * 128)
    (project / '.sdk/large-sdk').write_bytes(b'x' * 128)
    (project / 'module.py').write_text('ANSWER = 42')
    manifest = manager.source_root / 'pixi.toml'
    manifest.write_text(manifest.read_text().replace('[environments]', '[feature.base.pypi-dependencies]\nlocal-package={path="local-package",editable=true}\n[environments]'))
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    registry.add_preset('base')
    created = registry.create(EnvironmentCreateRequest(name='snapshot', base_environment_id=next(iter(registry.sources))))
    workspace = registry.definitions / created.environment_id
    definition = read_manifest(workspace)
    local = definition['feature']['base']['pypi-dependencies']['local-package']
    copied = workspace / local['path']
    assert (copied / 'module.py').read_text() == 'ANSWER = 42'
    assert local['editable'] is False
    assert not (copied / '.pixi').exists()
    assert not (copied / '.sdk').exists()
    (project / 'module.py').write_text('ANSWER = 99')
    assert (copied / 'module.py').read_text() == 'ANSWER = 42'


def test_deriving_from_developer_revision_retains_parent_additions_and_blocks_base_removal(tmp_path: Path) -> None:
    manager = source(tmp_path / 'base')
    registry = RuntimeRegistry(tmp_path / 'data', manager)
    parent = registry.create(EnvironmentCreateRequest(name='parent', pypi_dependencies=('requests>=2,<3',)))
    def run(operation: InstallOperation, command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
        if command[1] == 'lock':
            write_lock(cwd, 'runtime')
        else:
            python = EnvironmentManager._python(cwd / '.pixi/envs/runtime')
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()
        return ''
    async def exercise() -> None:
        await registry.prepare(parent.environment_id)
        assert registry.preparation_task is not None
        await registry.preparation_task
        child = registry.create(EnvironmentCreateRequest(name='child', base_environment_id=parent.environment_id,
                                                        conda_dependencies=('ffmpeg >=7',)))
        definition = read_manifest(registry.definitions / child.environment_id)
        features = definition['feature']
        assert features['f8-additions']['pypi-dependencies']['requests']['version'] == '<3,>=2'
        assert any('ffmpeg' in feature.get('dependencies', {}) for feature in features.values())
        with pytest.raises(ConflictError, match='base'):
            registry.remove(parent.environment_id, set())
        registry.remove(child.environment_id, set())
        registry.remove(parent.environment_id, set())
    with patch.object(EnvironmentManager, '_pixi', return_value=Path('/fixture/pixi')), patch.object(InstallOperation, 'run', autospec=True, side_effect=run):
        asyncio.run(exercise())
