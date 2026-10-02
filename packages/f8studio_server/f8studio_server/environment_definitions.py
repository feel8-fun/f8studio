"""Explicit Pixi environment snapshots and identities, independent of installation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import tomllib
from typing import cast
from urllib.parse import urlparse

import msgspec
import yaml

from f8pysdk.specs import F8JsonValue
from .errors import InvalidRequestError

JsonObject = dict[str, F8JsonValue]


def object_value(value: F8JsonValue, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise InvalidRequestError(f'{context} must be an object')
    return value


def read_manifest(root: Path) -> JsonObject:
    return msgspec.convert(tomllib.loads((root / 'pixi.toml').read_text(encoding='utf-8')), type=JsonObject)


def selected_manifest(root: Path, environment: str) -> JsonObject:
    manifest = read_manifest(root)
    definitions = object_value(manifest.get('environments', {}), 'Pixi environments')
    if environment not in definitions:
        raise InvalidRequestError(f'Unknown Pixi environment: {environment}')
    definition = definitions[environment]
    if isinstance(definition, list):
        feature_names = definition
        use_default = True
    else:
        settings = object_value(definition, 'Environment definition')
        feature_names = settings.get('features', [])
        use_default = not settings.get('no-default-feature', False)
    if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
        raise InvalidRequestError('Environment features must be names')
    result: JsonObject = {}
    workspace = object_value(manifest.get('workspace', manifest.get('project', {})), 'Pixi workspace')
    result['workspace'] = {key: value for key, value in workspace.items()
                           if key in {'channels', 'platforms', 'channel-priority', 'requires-pixi'}}
    if use_default:
        for key in ('dependencies', 'host-dependencies', 'build-dependencies', 'pypi-dependencies',
                    'system-requirements', 'target', 'tasks', 'pypi-options'):
            if key in manifest:
                result[key] = manifest[key]
    features = object_value(manifest.get('feature', {}), 'Pixi features')
    result['feature'] = {str(name): features[str(name)] for name in feature_names}
    result['environments'] = {environment: {'features': feature_names, 'no-default-feature': not use_default}}
    return result


def read_lock(root: Path) -> JsonObject | None:
    raw: object = yaml.safe_load((root / 'pixi.lock').read_text(encoding='utf-8'))
    # Older test/build fixtures may only carry an opaque lock identifier.
    if not isinstance(raw, dict):
        return None
    return msgspec.convert(raw, type=JsonObject)


def selected_lock(root: Path, environment: str) -> JsonObject | None:
    lock = read_lock(root)
    if lock is None:
        return None
    environments = object_value(lock.get('environments', {}), 'Lock environments')
    selected = object_value(environments.get(environment), f'Locked environment {environment}')
    platform_packages = object_value(selected.get('packages', {}), 'Locked platform packages')
    references: set[str] = set()
    for packages in platform_packages.values():
        if not isinstance(packages, list):
            raise InvalidRequestError('Invalid locked package references')
        for package in packages:
            entry = object_value(package, 'Locked package reference')
            for kind in ('conda', 'pypi'):
                reference = entry.get(kind)
                if isinstance(reference, str):
                    references.add(reference)
    packages = lock.get('packages', [])
    if not isinstance(packages, list):
        raise InvalidRequestError('Invalid lock package metadata')
    metadata: list[F8JsonValue] = []
    for package in packages:
        entry = object_value(package, 'Locked package metadata')
        if entry.get('conda') in references or entry.get('pypi') in references:
            metadata.append(entry)
    result: JsonObject = {'version': lock.get('version'), 'environments': {environment: selected}, 'packages': metadata}
    if 'platforms' in lock:
        result['platforms'] = lock['platforms']
    return result


def environment_identity(root: Path, environment: str) -> str:
    snapshot = selected_manifest(root, environment)
    lock = selected_lock(root, environment)
    digest = hashlib.sha256(f'{sys.platform}:{platform.machine()}:{environment}'.encode())
    digest.update(json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode())
    digest.update(json.dumps(lock, sort_keys=True, separators=(',', ':')).encode() if lock is not None
                  else (root / 'pixi.lock').read_bytes())
    referenced_files: set[Path] = set()
    if lock is not None:
        packages = lock.get('packages', [])
        if isinstance(packages, list):
            for value in packages:
                entry = object_value(value, 'Locked package')
                for kind in ('pypi', 'conda'):
                    reference = entry.get(kind)
                    if isinstance(reference, str):
                        location = urlparse(reference)
                        if not location.scheme:
                            referenced_files.add((root / reference).resolve())
                        elif location.scheme == 'file':
                            referenced_files.add(Path(location.path).resolve())
    for wheel in sorted((root / 'wheels').glob('*.whl')):
        if lock is not None and wheel.resolve() not in referenced_files:
            continue
        with wheel.open('rb') as source:
            digest.update(wheel.name.encode())
            digest.update(hashlib.file_digest(source, 'sha256').digest())
    return digest.hexdigest()


def toml_value(value: F8JsonValue) -> str:
    """Serialize the finite JSON-shaped subset used by Pixi manifests."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return '[' + ', '.join(toml_value(item) for item in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(f'{toml_key(key)} = {toml_value(item)}' for key, item in value.items()) + '}'
    raise InvalidRequestError('Null is not valid in a Pixi manifest')


def toml_key(key: str) -> str:
    return key if re.fullmatch(r'[A-Za-z0-9_-]+', key) else json.dumps(key)


def write_manifest(path: Path, manifest: JsonObject) -> None:
    lines: list[str] = []
    for name, table in manifest.items():
        values = object_value(table, f'Pixi table {name}')
        lines.append(f'[{toml_key(name)}]')
        lines.extend(f'{toml_key(key)} = {toml_value(value)}' for key, value in values.items())
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def relocate_dependencies(manifest: JsonObject, source_root: Path, destination: Path) -> None:
    """Snapshot local dependencies once; never leave mutable source paths in a revision."""
    import shutil
    for key, value in manifest.items():
        if not isinstance(value, dict):
            continue
        if key == 'pypi-dependencies':
            for spec in value.values():
                if not isinstance(spec, dict):
                    continue
                dependency = cast(JsonObject, spec)
                reference = dependency.get('path')
                if not isinstance(reference, str):
                    continue
                source = (source_root / reference).resolve()
                if not source.is_relative_to(source_root.resolve()):
                    raise InvalidRequestError(f'Local dependency escapes package: {source}')
                relative = Path('sources') / hashlib.sha256(str(source).encode()).hexdigest()[:16] / source.name
                target = destination / relative
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.is_dir():
                        shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', '.pixi', 'build', '__pycache__', '.pytest_cache', 'node_modules', '.sdk', '.venv', 'env'))
                    else:
                        shutil.copy2(source, target)
                spec['path'] = relative.as_posix()
                spec['editable'] = False
        else:
            relocate_dependencies(value, source_root, destination)
