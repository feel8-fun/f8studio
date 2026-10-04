"""Explicit Pixi environment snapshots and identities, independent of installation."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
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


@lru_cache(maxsize=32)
def _read_lock_snapshot(path: Path, modified_ns: int, size: int) -> JsonObject | None:
    raw: object = yaml.safe_load(path.read_text(encoding='utf-8'))
    # Older test/build fixtures may only carry an opaque lock identifier.
    if not isinstance(raw, dict):
        return None
    return msgspec.convert(raw, type=JsonObject)


def read_lock(root: Path) -> JsonObject | None:
    path = (root / 'pixi.lock').resolve()
    stat = path.stat()
    return deepcopy(_read_lock_snapshot(path, stat.st_mtime_ns, stat.st_size))


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
    for wheel in local_wheels(root, environment):
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


def toml_text(manifest: JsonObject) -> str:
    lines: list[str] = []
    for name, table in manifest.items():
        values = object_value(table, f'Pixi table {name}')
        lines.append(f'[{toml_key(name)}]')
        lines.extend(f'{toml_key(key)} = {toml_value(value)}' for key, value in values.items())
        lines.append('')
    return '\n'.join(lines)


def write_manifest(path: Path, manifest: JsonObject) -> None:
    path.write_text(toml_text(manifest), encoding='utf-8')


def relocate_dependencies(manifest: JsonObject, source_root: Path, destination: Path, *,
                          allowed_root: Path | None = None, preserve_editable: bool = False) -> None:
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
                if not source.is_relative_to((allowed_root or source_root).resolve()):
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
                if not preserve_editable:
                    spec['editable'] = False
        else:
            relocate_dependencies(value, source_root, destination, allowed_root=allowed_root, preserve_editable=preserve_editable)


def local_wheels(root: Path, environment: str) -> tuple[Path, ...]:
    lock = selected_lock(root, environment)
    if lock is None:
        return tuple(sorted((root / 'wheels').glob('*.whl')))
    packages = lock.get('packages', [])
    result: set[Path] = set()
    if isinstance(packages, list):
        for value in packages:
            entry = object_value(value, 'Locked package')
            reference = entry.get('pypi')
            if isinstance(reference, str) and not urlparse(reference).scheme:
                path = (root / reference).resolve()
                if path.suffix == '.whl':
                    result.add(path)
    return tuple(sorted(result))


def materialize_locked_environment(root: Path, environment: str, destination: Path, allowed_root: Path) -> None:
    """Copy local inputs and retarget an existing lock without resolving versions."""
    manifest = selected_manifest(root, environment)
    # Pixi also locks its implicit default environment. Keep that metadata
    # when retargeting a single-environment workspace for --locked install.
    lock = read_lock(root)
    if lock is None:
        raise InvalidRequestError('A managed runtime requires a structured Pixi lock file')
    replacements: dict[str, str] = {}

    def collect(table: JsonObject) -> None:
        for key, value in table.items():
            if not isinstance(value, dict):
                continue
            if key == 'pypi-dependencies':
                for entry in value.values():
                    if isinstance(entry, dict) and isinstance(cast(JsonObject, entry).get('path'), str):
                        reference = cast(str, entry['path'])
                        source = (root / reference).resolve()
                        relative = Path('sources') / hashlib.sha256(str(source).encode()).hexdigest()[:16] / source.name
                        replacements[reference] = relative.as_posix()
            else:
                collect(value)

    def retarget(table: JsonObject) -> None:
        for key, value in table.items():
            if isinstance(value, dict):
                retarget(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        retarget(cast(JsonObject, item))
            elif key in {'pypi', 'conda'} and isinstance(value, str) and value in replacements:
                table[key] = replacements[value]

    collect(manifest)
    relocate_dependencies(manifest, root, destination, allowed_root=allowed_root, preserve_editable=True)
    retarget(lock)
    write_manifest(destination / 'pixi.toml', manifest)
    (destination / 'pixi.lock').write_text(yaml.safe_dump(lock, sort_keys=False), encoding='utf-8')
