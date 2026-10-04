"""Explicit references to independently versioned official Pixi workspaces."""
from __future__ import annotations

from pathlib import Path
import re
from typing import cast

import msgspec

from f8pysdk.release_spec import RuntimeCatalog, RuntimeDefinition as RuntimeDefinition
from packaging.version import Version

from .environment_definitions import JsonObject, read_manifest, selected_lock, selected_manifest
from .errors import InvalidRequestError


def read_runtime_catalog(root: Path) -> RuntimeCatalog:
    path = root / 'config/runtime-environments.json'
    if not path.is_file():
        return RuntimeCatalog(schema_version='f8runtimeCatalog/1', runtimes=())
    catalog = msgspec.json.decode(path.read_bytes(), type=RuntimeCatalog)
    for definition in catalog.runtimes:
        if (definition.provider_id is None) != (definition.version is None):
            raise InvalidRequestError('Runtime providerId and version must be declared together')
        if definition.version is not None:
            Version(definition.version)
        if definition.provider_id is not None and not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*', definition.provider_id):
            raise InvalidRequestError(f'Invalid runtime provider: {definition.provider_id}')
        if definition.abi is not None and not definition.abi.strip():
            raise InvalidRequestError('Runtime ABI must not be empty')
    return catalog


def read_runtime_sources(root: Path) -> dict[str, tuple[Path, str | None]]:
    path = root / 'config/runtime-environments.json'
    if not path.is_file():
        return {}
    catalog = read_runtime_catalog(root)
    if catalog.schema_version != 'f8runtimeCatalog/1':
        raise InvalidRequestError(f'Unsupported runtime catalog: {catalog.schema_version}')
    sources: dict[str, tuple[Path, str | None]] = {}
    for definition in catalog.runtimes:
        name = definition.runtime_id
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', name) or name in sources:
            raise InvalidRequestError(f'Invalid or duplicate official runtime ID: {name}')
        if definition.development_environment not in {None, name}:
            raise InvalidRequestError(f'Runtime {name} migration must reference the same development environment name')
        prefix = '${F8_PACKAGE_ROOT}/'
        if not definition.manifest.startswith(prefix):
            raise InvalidRequestError(f'Runtime {name} must use an explicit F8_PACKAGE_ROOT manifest reference')
        manifest = (root / definition.manifest.removeprefix(prefix)).resolve()
        if not manifest.is_relative_to(root.resolve()) or manifest.name != 'pixi.toml' or not manifest.is_file():
            raise InvalidRequestError(f'Missing or unsafe manifest for runtime {name}: {definition.manifest}')
        if manifest.parent == root.resolve():
            raise InvalidRequestError('Runtime catalog must reference a separate workspace directory')
        environments = read_manifest(manifest.parent).get('environments')
        if not isinstance(environments, dict) or name not in environments:
            raise InvalidRequestError(f'Runtime {name} must declare its selected environment')
        if not manifest.with_name('pixi.lock').is_file():
            raise InvalidRequestError(f'Runtime {name} is missing its independent pixi.lock')
        sources[name] = manifest.parent, definition.development_environment
    return sources


def _absolute_paths(value: JsonObject, root: Path) -> JsonObject:
    """Compare source references relative to their owning manifests, not spelling."""
    result: JsonObject = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[key] = _absolute_paths(item, root)
        elif isinstance(item, list):
            result[key] = [_absolute_paths(cast(JsonObject, entry), root) if isinstance(entry, dict) else entry for entry in item]
        elif key in {'path', 'pypi', 'conda'} and isinstance(item, str) and '://' not in item:
            result[key] = str((root / item).resolve())
        else:
            result[key] = item
    return result


def equivalent_environment(first: Path, second: Path, environment: str) -> bool:
    if not (first / 'pixi.toml').is_file() or not (first / 'pixi.lock').is_file():
        return False
    definitions = read_manifest(first).get('environments', {})
    if not isinstance(definitions, dict) or environment not in definitions:
        return False
    first_manifest = _absolute_paths(selected_manifest(first, environment), first)
    second_manifest = _absolute_paths(selected_manifest(second, environment), second)
    # Runtime workspaces omit development tasks. Dependency compatibility is
    # checked independently of launcher tasks, which still use the source tree.
    for manifest in (first_manifest, second_manifest):
        manifest.pop('tasks', None)
        features = manifest.get('feature', {})
        if isinstance(features, dict):
            for feature in features.values():
                if isinstance(feature, dict):
                    cast(JsonObject, feature).pop('tasks', None)
    first_lock = selected_lock(first, environment)
    second_lock = selected_lock(second, environment)
    return (first_manifest == second_manifest and first_lock is not None and second_lock is not None
            and _absolute_paths(first_lock, first) == _absolute_paths(second_lock, second))
