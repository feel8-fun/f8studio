"""Assemble publisher artifacts without compiling services or resolving dependencies."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import cast
import urllib.parse
import urllib.request

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sdk/python'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packages/f8studio_server'))

import msgspec
import yaml

from f8pysdk.runtime_package import validate_runtime_package
from f8pysdk.release_spec import (
    BundledExtensionCatalog, BundledExtensionPackage, PublishedArtifact,
    ReleaseArtifact, RuntimeCatalog, RuntimeDefinition, StudioReleaseLock,
)
from f8studio_server.environment_definitions import JsonObject, read_lock, read_manifest, write_manifest
from f8studio_server.extension_artifacts import MAX_ARCHIVE_BYTES, extract_archive
from f8studio_server.runtime_sources import read_runtime_sources


PLATFORMS = {'linux-x86_64', 'windows-x86_64'}


def _fetch_artifact(artifact: ReleaseArtifact, root: Path, cache: Path) -> Path:
    if not re.fullmatch(r'[0-9a-f]{64}', artifact.sha256):
        raise ValueError(f'Invalid SHA-256 for {artifact.artifact_id}')
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f'{artifact.sha256}.zip'
    if not archive.is_file():
        temporary = archive.with_suffix('.download')
        parsed = urllib.parse.urlsplit(artifact.location)
        try:
            if parsed.scheme and not Path(artifact.location).is_absolute():
                if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError('Release artifacts require HTTPS without embedded credentials')
                with urllib.request.urlopen(artifact.location, timeout=30) as source, temporary.open('wb') as target:
                    if urllib.parse.urlsplit(str(source.geturl())).scheme != 'https':
                        raise ValueError('Artifact download redirected away from HTTPS')
                    total = 0
                    while chunk := source.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_ARCHIVE_BYTES:
                            raise ValueError('Release archive exceeds the download limit')
                        target.write(chunk)
            else:
                source_path = (root / artifact.location).resolve()
                if source_path.stat().st_size > MAX_ARCHIVE_BYTES:
                    raise ValueError('Release archive exceeds the download limit')
                shutil.copyfile(source_path, temporary)
            with temporary.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != artifact.sha256:
                    raise ValueError(f'Artifact checksum mismatch: {artifact.artifact_id}')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != artifact.sha256:
            raise ValueError(f'Cached artifact checksum mismatch: {artifact.artifact_id}')
    return archive


def _base_manifest(source: Path, output: Path) -> None:
    """Point the launcher at exactly the provider's inputs, preserving the full lock."""
    manifest = read_manifest(source)
    lock = read_lock(source)
    if lock is None:
        raise ValueError('Base runtime requires a structured lock')

    def relocate(table: JsonObject, keys: set[str]) -> None:
        for key, value in table.items():
            if isinstance(value, dict):
                relocate(value, keys)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        relocate(cast(JsonObject, item), keys)
            elif key in keys and isinstance(value, str) and '://' not in value:
                target = (source / value).resolve()
                if not target.is_relative_to(output.resolve()) or not target.is_file():
                    raise ValueError(f'Runtime input escapes release: {value}')
                table[key] = Path(os.path.relpath(target, output)).as_posix()

    relocate(manifest, {'path'})
    relocate(lock, {'pypi', 'conda'})
    write_manifest(output / 'pixi.toml', manifest)
    (output / 'pixi.lock').write_text(yaml.safe_dump(lock, sort_keys=False), encoding='utf-8')


def validate_runtime_artifact(root: Path) -> RuntimeCatalog:
    return validate_runtime_package(root)


def assemble_release(lock_path: Path, destination: Path, *, cache: Path, platform: str) -> tuple[str, ...]:
    lock = msgspec.json.decode(lock_path.read_bytes(), type=StudioReleaseLock)
    if platform not in PLATFORMS or lock.platform != platform:
        raise ValueError(f'Release platform {lock.platform} does not match {platform}')
    identities = [item.artifact_id for item in lock.artifacts]
    if len(identities) != len(set(identities)):
        raise ValueError('Release artifact IDs must be unique')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Release destination must be empty')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # All downloads, metadata validation and extraction finish before publication.
    with tempfile.TemporaryDirectory(prefix='f8-release-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'release'
        staging.mkdir()
        definitions: list[RuntimeDefinition] = []
        packages: list[BundledExtensionPackage] = []
        owners: set[str] = set()
        services: set[str] = set()
        for artifact in lock.artifacts:
            archive = _fetch_artifact(artifact, lock_path.parent, cache)
            kind_root = 'runtime-providers' if artifact.kind == 'runtime' else 'extension-packages'
            relative = Path(kind_root) / artifact.sha256
            payload = staging / relative
            payload.parent.mkdir(exist_ok=True)
            extract_archive(archive, payload, required=('config/artifact.json',))
            descriptor = msgspec.json.decode((payload / 'config/artifact.json').read_bytes(), type=PublishedArtifact)
            if (descriptor.artifact_id, descriptor.version, descriptor.kind) != (artifact.artifact_id, artifact.version, artifact.kind):
                raise ValueError(f'Artifact identity does not match release lock: {artifact.artifact_id}')
            if descriptor.platform not in {'any', platform}:
                raise ValueError(f'Artifact platform mismatch: {artifact.artifact_id}')
            if artifact.kind == 'runtime':
                catalog = validate_runtime_artifact(payload)
                if len(catalog.runtimes) != 1 or catalog.runtimes[0].runtime_id != artifact.artifact_id or catalog.runtimes[0].version != artifact.version:
                    raise ValueError('Runtime artifact must publish exactly the locked runtime identity/version')
                definition = catalog.runtimes[0]
                definitions.append(msgspec.structs.replace(definition, manifest='${F8_PACKAGE_ROOT}/' +
                    relative.as_posix() + '/' + definition.manifest.removeprefix('${F8_PACKAGE_ROOT}/')))
            else:
                from f8pysdk.extension_spec import ExtensionCatalog
                catalog_extensions = msgspec.json.decode((payload / 'config/extensions.json').read_bytes(), type=ExtensionCatalog)
                if len(catalog_extensions.extensions) != 1:
                    raise ValueError('A release artifact must publish exactly one extension')
                extension = catalog_extensions.extensions[0]
                if (extension.extension_id, extension.version) != (artifact.artifact_id, artifact.version):
                    raise ValueError('Extension identity does not match artifact descriptor')
                if extension.extension_id in owners or services.intersection(extension.service_classes):
                    raise ValueError('Extension artifacts have conflicting ownership')
                if catalog_extensions.preinstalled or extension.runtime.kind not in {'native', 'shared', 'pixi'}:
                    raise ValueError('Published extension must be independently installable')
                owners.add(extension.extension_id)
                services.update(extension.service_classes)
                packages.append(BundledExtensionPackage(path='${F8_PACKAGE_ROOT}/' + relative.as_posix(), sha256=artifact.sha256))
        runtime_ids = [definition.runtime_id for definition in definitions]
        if lock.base_runtime != 'studio-runtime' or lock.base_runtime not in runtime_ids or len(runtime_ids) != len(set(runtime_ids)):
            raise ValueError('Release needs exactly one studio-runtime base and unique runtime IDs')
        config = staging / 'config'
        config.mkdir()
        (config / 'runtime-environments.json').write_bytes(msgspec.json.encode(RuntimeCatalog(schema_version='f8runtimeCatalog/1', runtimes=tuple(definitions))))
        (config / 'extension-packages.json').write_bytes(msgspec.json.encode(BundledExtensionCatalog(schema_version='f8extensionPackages/1', packages=tuple(packages))))
        (config / 'extensions.json').write_text(json.dumps({'schemaVersion': 'f8extensionCatalog/1', 'extensions': [], 'preinstalled': []}) + '\n')
        (config / 'service-index.json').write_text(json.dumps({'schemaVersion': 'f8serviceIndex/1', 'services': [], 'modelRoot': '${F8_MODEL_ROOT}'}) + '\n')
        sources = read_runtime_sources(staging)
        _base_manifest(sources[lock.base_runtime][0], staging)
        (config / 'release-lock.json').write_bytes(msgspec.json.encode(lock))
        from f8studio_server.extensions import ExtensionManager
        ExtensionManager(Path(temporary) / 'validation-data', base_index=config / 'service-index.json')
        if destination.exists():
            destination.rmdir()
        staging.replace(destination)
    return tuple(runtime_ids)
