from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import msgspec
import pytest
import yaml

# Release tooling is not part of the installed server package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from assemble_release import assemble_release, validate_runtime_artifact
from f8pysdk.release_spec import PublishedArtifact, ReleaseArtifact, StudioReleaseLock
from f8studio_server.environments import EnvironmentManager
from f8studio_server.errors import InvalidRequestError
from f8studio_server.extensions import ExtensionManager
from f8studio_server.extension_models import EnvironmentCreateRequest, ExtensionManifest, ExtensionRuntime
from f8studio_server.runtime_registry import RuntimeRegistry


def archive_artifact(root: Path, name: str, *, version: str = '1.0', runtime: bool = True,
                     platform: str = 'linux-x86_64') -> ReleaseArtifact:
    payload = root / f'{name}-{version}'
    (payload / 'config').mkdir(parents=True)
    descriptor = PublishedArtifact(schema_version='f8artifact/1', artifact_id=name, version=version,
                                   kind='runtime' if runtime else 'extension', platform=platform)
    (payload / 'config/artifact.json').write_bytes(msgspec.json.encode(descriptor))
    if runtime:
        workspace = payload / 'runtimes' / name
        workspace.mkdir(parents=True)
        (workspace / 'pixi.toml').write_text('[workspace]\nname="test"\nchannels=["conda-forge"]\nplatforms=["linux-64"]\n'
            '[dependencies]\npython="3.12.*"\n[environments]\n' + name + '=[]\n')
        package = {'conda': 'https://example.invalid/python-3.12.9-build_0.conda', 'sha256': 'a' * 64}
        (workspace / 'pixi.lock').write_text(yaml.safe_dump({'version': 6, 'environments': {
            name: {'packages': {'linux-64': [{'conda': package['conda']}]}}}, 'packages': [package]}))
        (payload / 'config/runtime-environments.json').write_text(json.dumps({'schemaVersion': 'f8runtimeCatalog/1', 'runtimes': [{
            'runtimeId': name, 'providerId': 'feel8.python', 'version': version, 'abi': 'cpython312',
            'manifest': '${F8_PACKAGE_ROOT}/runtimes/' + name + '/pixi.toml',
        }]}))
    else:
        (payload / 'config/extensions.json').write_text(json.dumps({'schemaVersion': 'f8extensionCatalog/1', 'extensions': [{
            'extensionId': name, 'name': name, 'version': version, 'description': 'Independent extension', 'runtime': {'kind': 'native'},
            'skills': [{'skillId': 'recipe', 'path': '${F8_PACKAGE_ROOT}/skills/recipe/SKILL.md'}],
        }]}))
    if not runtime:
        (payload / 'skills/recipe').mkdir(parents=True)
        (payload / 'skills/recipe/SKILL.md').write_text('A recipe')
    output = root / f'{name}-{version}.zip'
    with zipfile.ZipFile(output, 'w') as archive:
        for path in payload.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(payload).as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return ReleaseArtifact(artifact_id=name, version=version, kind=descriptor.kind, location=output.name, sha256=digest)


def release_lock(root: Path, artifacts: tuple[ReleaseArtifact, ...]) -> Path:
    path = root / 'release.json'
    path.write_bytes(msgspec.json.encode(StudioReleaseLock(schema_version='f8studioRelease/1', platform='linux-x86_64', artifacts=artifacts)))
    return path


def test_assembly_registers_independent_packages_and_parallel_runtime_versions(tmp_path: Path) -> None:
    artifacts = (archive_artifact(tmp_path, 'studio-runtime'), archive_artifact(tmp_path, 'python-services-v1'),
                 archive_artifact(tmp_path, 'python-services-v2', version='2.0'), archive_artifact(tmp_path, 'debug', runtime=False))
    output = tmp_path / 'release'
    assert assemble_release(release_lock(tmp_path, artifacts), output, cache=tmp_path / 'cache', platform='linux-x86_64') == (
        'studio-runtime', 'python-services-v1', 'python-services-v2')
    manager = ExtensionManager(tmp_path / 'data', base_index=output / 'config/service-index.json')
    assert manager.status('debug').state == 'available'
    assert manager._payloads['debug'].root == output / 'extension-packages' / artifacts[-1].sha256
    registry = manager.runtime_registry
    assert len(registry.sources) == 3
    manifest = ExtensionManifest(extension_id='consumer', name='Consumer', version='1', description='',
        runtime=ExtensionRuntime(kind='shared', environment='python-services-v1', provider_id='feel8.python',
                                 provider_version='>=1,<2', abi='cpython312'))
    registry.validate_compatibility(manifest)
    v2 = next(key for key, source in registry.sources.items() if source.name == 'python-services-v2')
    with pytest.raises(InvalidRequestError, match='does not satisfy'):
        registry.validate_compatibility(manifest, v2)
    detail = registry.detail(v2)
    assert detail.provider_id == 'feel8.python' and detail.provider_version == '2.0' and detail.abi == 'cpython312'
    restarted = RuntimeRegistry(tmp_path / 'data', EnvironmentManager(tmp_path / 'data', output))
    assert set(registry.sources) == set(restarted.sources)


@pytest.mark.parametrize('failure', ['checksum', 'platform', 'identity', 'traversal'])
def test_assembly_failures_do_not_publish_partial_output(tmp_path: Path, failure: str) -> None:
    artifact = archive_artifact(tmp_path, 'studio-runtime', platform='windows-x86_64' if failure == 'platform' else 'linux-x86_64')
    if failure == 'checksum':
        artifact = msgspec.structs.replace(artifact, sha256='b' * 64)
    elif failure == 'identity':
        artifact = msgspec.structs.replace(artifact, version='9.0')
    elif failure == 'traversal':
        archive = tmp_path / artifact.location
        with zipfile.ZipFile(archive, 'a') as package:
            package.writestr('../../escape.txt', 'escape')
        artifact = msgspec.structs.replace(artifact, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    output = tmp_path / 'release'
    with pytest.raises((ValueError, InvalidRequestError)):
        assemble_release(release_lock(tmp_path, (artifact,)), output, cache=tmp_path / 'cache', platform='linux-x86_64')
    assert not output.exists()
    assert not (tmp_path / 'escape.txt').exists()


def test_runtime_publisher_rejects_source_dependencies(tmp_path: Path) -> None:
    archive_artifact(tmp_path, 'studio-runtime')
    root = tmp_path / 'studio-runtime-1.0'
    workspace = root / 'runtimes/studio-runtime'
    (workspace / 'pixi.toml').write_text((workspace / 'pixi.toml').read_text() +
        '[pypi-dependencies]\nexample={path="../../source",editable=true}\n')
    with pytest.raises(ValueError, match='contained wheels'):
        validate_runtime_artifact(root)


def test_derived_preserve_retains_provider_but_adjust_does_not_claim_abi(tmp_path: Path) -> None:
    artifact = archive_artifact(tmp_path, 'studio-runtime')
    output = tmp_path / 'release'
    assemble_release(release_lock(tmp_path, (artifact,)), output, cache=tmp_path / 'cache', platform='linux-x86_64')
    registry = RuntimeRegistry(tmp_path / 'data', EnvironmentManager(tmp_path / 'data', output))
    base = next(iter(registry.sources))
    preserved = registry.create(EnvironmentCreateRequest(name='preserved', base_environment_id=base, policy='preserve'))
    adjusted = registry.create(EnvironmentCreateRequest(name='adjusted', base_environment_id=base, policy='adjust'))
    assert registry.release_definition(preserved.environment_id).abi == 'cpython312'
    assert registry.release_definition(adjusted.environment_id).abi is None


def test_distribution_parser_requires_explicit_artifact_or_workspace_mode() -> None:
    spec = importlib.util.spec_from_file_location('release_dist_ci', Path('scripts/dist_ci.py'))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module._build_parser().parse_args([])
    assert module._build_parser().parse_args(['--release-lock', 'release.json']).release_lock == Path('release.json')


def test_package_runtime_provider_is_reusable_and_exposes_its_release_identity(tmp_path: Path) -> None:
    base = archive_artifact(tmp_path, 'studio-runtime')
    extension = archive_artifact(tmp_path, 'debug', runtime=False)
    archive_artifact(tmp_path, 'private-v1', version='2.0')
    package = tmp_path / 'debug-1.0'
    private = tmp_path / 'private-v1-2.0'
    import shutil
    shutil.copytree(private / 'runtimes', package / 'runtimes')
    shutil.copy2(private / 'config/runtime-environments.json', package / 'config/runtime-environments.json')
    catalog_path = package / 'config/extensions.json'
    catalog = json.loads(catalog_path.read_text())
    catalog['extensions'][0]['runtime'] = {'kind': 'pixi', 'environment': 'private-v1'}
    catalog_path.write_text(json.dumps(catalog))
    path = tmp_path / extension.location
    with zipfile.ZipFile(path, 'w') as archive:
        for item in package.rglob('*'):
            if item.is_file():
                archive.write(item, item.relative_to(package).as_posix())
    extension = msgspec.structs.replace(extension, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    output = tmp_path / 'release'
    assemble_release(release_lock(tmp_path, (base, extension)), output, cache=tmp_path / 'cache', platform='linux-x86_64')
    manager = ExtensionManager(tmp_path / 'data', base_index=output / 'config/service-index.json')
    identifier = next(key for key, source in manager.runtime_registry.sources.items() if source.name == 'private-v1')
    assert manager.runtime_registry.source(identifier).source == 'package'
    assert manager.runtime_registry.detail(identifier).provider_version == '2.0'
    manifest = ExtensionManifest(extension_id='another', name='Another', version='1', description='',
        runtime=ExtensionRuntime(kind='shared', environment=identifier, provider_id='feel8.python', provider_version='>=2,<3'))
    manager.runtime_registry.validate_compatibility(manifest)


def test_shared_wheel_tags_are_checked_against_the_selected_interpreter(tmp_path: Path) -> None:
    from f8studio_server.shared_dependencies import RuntimeProbe, validate_shared_package
    python = tmp_path / 'python'
    info = python / 'compiled-1.0.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: compiled\nVersion: 1.0\n')
    (info / 'WHEEL').write_text('Wheel-Version: 1.0\nTag: cp312-cp312-win_amd64\n')
    manifest = ExtensionManifest(extension_id='compiled', name='Compiled', version='1', description='',
        runtime=ExtensionRuntime(kind='shared', environment='base'))
    probe = RuntimeProbe(python_version='3.14.3', markers={}, distributions={}, modules=(), wheel_tags=('cp314-cp314-manylinux_2_28_x86_64',))
    with pytest.raises(InvalidRequestError, match='interpreter/ABI/platform'):
        validate_shared_package(manifest, python, probe)


def test_runtime_probe_works_in_a_bare_interpreter_without_reporting_tooling_as_dependencies(tmp_path: Path) -> None:
    import os
    import packaging
    import shutil
    import subprocess
    import venv
    from f8studio_server.shared_dependencies import RuntimeProbe
    prefix = tmp_path / 'bare'
    venv.EnvBuilder(with_pip=False).create(prefix)
    library = tmp_path / 'probe-library'
    shutil.copytree(Path(packaging.__file__).parent, library / 'packaging')
    python = prefix / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    script = Path('packages/f8studio_server/f8studio_server/_runtime_probe.py').resolve()
    output = subprocess.run([str(python), '-I', str(script), str(library)], capture_output=True, text=True, check=True)
    probe = msgspec.json.decode(output.stdout.encode(), type=RuntimeProbe)
    assert probe.wheel_tags
    assert probe.distributions == {}


def test_extension_version_switch_rollback_and_restart_do_not_require_studio_rebuild(tmp_path: Path) -> None:
    import asyncio
    import io
    from unittest.mock import patch
    from f8studio_server.errors import ConflictError
    from f8studio_server.extension_models import ExtensionImportRequest

    class Download(io.BytesIO):
        def geturl(self) -> str:
            return 'https://example.invalid/artifact.zip'

    base = archive_artifact(tmp_path, 'studio-runtime')
    old = archive_artifact(tmp_path, 'debug', version='1.0', runtime=False)
    new = archive_artifact(tmp_path, 'debug', version='2.0', runtime=False)
    output = tmp_path / 'release'
    assemble_release(release_lock(tmp_path, (base,)), output, cache=tmp_path / 'cache', platform='linux-x86_64')
    manager = ExtensionManager(tmp_path / 'data', base_index=output / 'config/service-index.json')

    async def import_version(artifact: ReleaseArtifact) -> None:
        request = ExtensionImportRequest(url='https://example.invalid/artifact.zip', sha256=artifact.sha256)
        with patch('f8studio_server.extension_artifacts.urllib.request.urlopen', return_value=Download((tmp_path / artifact.location).read_bytes())):
            await manager.import_package(request)

    async def install() -> None:
        await manager.install('debug', lambda: None)
        assert manager._task is not None
        await manager._task
        assert manager.status('debug').state == 'installed'

    async def exercise() -> None:
        await import_version(old)
        await install()
        with pytest.raises(ConflictError, match='Uninstall'):
            await import_version(new)
        assert manager.status('debug').version == '1.0'
        assert manager.status('debug').state == 'installed'
        await manager.uninstall('debug', lambda: None, lambda _name: False)
        await import_version(new)
        assert manager.status('debug').version == '2.0'
        assert manager.status('debug').state == 'available'
        await install()
        await manager.uninstall('debug', lambda: None, lambda _name: False)
        await import_version(old)
        assert manager.status('debug').version == '1.0'
        await install()
        await manager.close()

    asyncio.run(exercise())
    assert manager._source_digests == [old.sha256, new.sha256, old.sha256]
    restored = ExtensionManager(tmp_path / 'data', base_index=output / 'config/service-index.json')
    assert restored.status('debug').version == '1.0' and restored.status('debug').state == 'installed'


def test_failed_version_selection_restores_catalog_and_runtime_sources(tmp_path: Path) -> None:
    import asyncio
    import io
    from unittest.mock import patch
    from f8studio_server.extension_models import ExtensionImportRequest

    class Download(io.BytesIO):
        def geturl(self) -> str:
            return 'https://example.invalid/artifact.zip'

    base = archive_artifact(tmp_path, 'studio-runtime')
    old = archive_artifact(tmp_path, 'debug', version='1.0', runtime=False)
    new = archive_artifact(tmp_path, 'debug', version='2.0', runtime=False)
    output = tmp_path / 'release'
    assemble_release(release_lock(tmp_path, (base, old)), output, cache=tmp_path / 'cache', platform='linux-x86_64')
    manager = ExtensionManager(tmp_path / 'data', base_index=output / 'config/service-index.json')
    sources = dict(manager.runtime_registry.sources)
    original_replace = Path.replace

    def reject_persistence(source: Path, target: Path | str) -> Path:
        if Path(target) == manager._sources_path:
            raise OSError('Cannot save extension version selection')
        return original_replace(source, target)

    async def exercise() -> None:
        request = ExtensionImportRequest(url='https://example.invalid/artifact.zip', sha256=new.sha256)
        with patch('f8studio_server.extension_artifacts.urllib.request.urlopen', return_value=Download((tmp_path / new.location).read_bytes())), \
             patch.object(Path, 'replace', autospec=True, side_effect=reject_persistence):
            with pytest.raises(InvalidRequestError, match='Cannot save'):
                await manager.import_package(request)
        assert manager.status('debug').version == '1.0'
        assert manager._source_digests == []
        assert manager.runtime_registry.sources == sources
        assert manager._payloads['debug'].root == output / 'extension-packages' / old.sha256
        await manager.close()

    asyncio.run(exercise())
