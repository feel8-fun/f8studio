"""Developer runtime revisions, preparation, retention and shared-runtime references."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from threading import RLock
from typing import Literal, cast
from urllib.parse import urlparse

import msgspec
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from f8pysdk.codec import copy_model
from f8pysdk.specs import F8JsonValue
from .environment_definitions import (
    JsonObject, object_value, relocate_dependencies, selected_lock,
    selected_manifest, toml_text, write_manifest,
)
from .environments import EnvironmentManager, SharedRuntimeTarget
from f8pysdk.release_spec import RuntimeDefinition
from packaging.specifiers import SpecifierSet

from .errors import ConflictError, InvalidRequestError, NotFoundError
from .extension_models import (
    EnvironmentCreateRequest, EnvironmentDetail, EnvironmentRevision, EnvironmentStatus,
    EnvironmentUsage, ExtensionManifest, ExtensionRuntime, RuntimeStorageStatus,
)
from .extension_operation import ExtensionInstallCancelled, InstallOperation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeSource:
    name: str
    source: Literal['official', 'package', 'developer']
    target: SharedRuntimeTarget
    manifest: ExtensionManifest | None = None
    release: RuntimeDefinition | None = None


@dataclass(frozen=True)
class RuntimeSourcesSnapshot:
    sources: dict[str, RuntimeSource]
    aliases: dict[str, str]
    shared_targets: dict[str, SharedRuntimeTarget]


def directory_usage(root: Path) -> EnvironmentUsage:
    logical = unique = shared = exclusive = 0
    seen: set[tuple[int, int]] = set()
    if root.is_dir():
        for directory, _directories, files in os.walk(root, followlinks=False):
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                stat = path.stat()
                logical += stat.st_size
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                unique += stat.st_size
                if stat.st_nlink > 1:
                    shared += stat.st_size
                else:
                    exclusive += stat.st_size
    return EnvironmentUsage(logical_bytes=logical, unique_file_bytes=unique,
                            shared_link_bytes=shared, exclusive_file_bytes=exclusive)


class RuntimeRegistry:
    def __init__(self, data_dir: Path, official: EnvironmentManager) -> None:
        self.data_dir = data_dir
        self.official = official
        self.storage = official.root.parent
        self.definitions = data_dir / 'runtime-definitions'
        self.state_file = self.definitions / 'revisions.json'
        self._source_history_file = self.definitions / 'official-sources.json'
        self._source_history = (msgspec.json.decode(self._source_history_file.read_bytes(), type=dict[str, str])
                                if self._source_history_file.is_file() else {})
        self._lock = RLock()
        self.sources: dict[str, RuntimeSource] = {}
        self._aliases: dict[str, str] = {}
        self._revisions = (msgspec.json.decode(self.state_file.read_bytes(), type=dict[str, EnvironmentRevision])
                           if self.state_file.is_file() else {})
        self._operation: InstallOperation | None = None
        self._task: asyncio.Task[None] | None = None
        self._progress: dict[str, str] = {}
        self._failures: dict[str, str] = {}
        for identifier, revision in tuple(self._revisions.items()):
            if revision.state == 'preparing':
                self._revisions[identifier] = copy_model(revision, update={
                    'state': 'failed', 'detail': 'Preparation was interrupted by Studio shutdown. Retry preparation.',
                })
            self._register_revision(identifier)
        if official.has_runtime_catalog:
            for name in official.preset_names():
                self.add_preset(name)

    @property
    def preparation_task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def busy(self) -> bool:
        return self._operation is not None

    def _save(self) -> None:
        self.definitions.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix('.tmp')
        temporary.write_bytes(msgspec.json.encode(self._revisions))
        temporary.replace(self.state_file)

    def snapshot_sources(self) -> RuntimeSourcesSnapshot:
        return RuntimeSourcesSnapshot(sources=dict(self.sources), aliases=dict(self._aliases),
                                      shared_targets=dict(self.official.shared_targets))

    def restore_sources(self, snapshot: RuntimeSourcesSnapshot) -> None:
        self.sources = snapshot.sources
        self._aliases = snapshot.aliases
        self.official.shared_targets = snapshot.shared_targets

    def add_source(self, manager: EnvironmentManager, manifest: ExtensionManifest, *, official: bool) -> None:
        environment = manifest.runtime.environment
        if manifest.runtime.kind == 'shared':
            if environment is not None:
                self.add_preset(environment)
            return
        if environment is None or manifest.runtime.kind == 'native':
            return
        owners = (tuple(manager.for_environment(name) for name in manager.preset_names())
                  if not official and manager.has_runtime_catalog
                  else (manager.for_environment(environment),))
        seen: set[tuple[Path, str]] = set()
        for owner in owners:
            names = (environment,) if official and not manager.has_runtime_catalog else owner.preset_names()
            for name in names:
                key = (owner.source_root, name)
                if key in seen:
                    continue
                seen.add(key)
                selected = copy_model(manifest, update={
                    'runtime': copy_model(manifest.runtime, update={'environment': name}),
                })
                plan = owner.plan(selected)
                assert plan.environment_id is not None
                release = manager.runtime_releases.get(name)
                source = RuntimeSource(name=name, source='official' if official else 'package',
                                       target=SharedRuntimeTarget(owner, plan, name), manifest=selected, release=release)
                self.sources[plan.environment_id] = source
                self.official.shared_targets[plan.environment_id] = source.target
                self._register_source_alias(source)

    def add_preset(self, environment: str) -> None:
        if environment not in self.official.preset_names():
            return
        manager = self.official.for_environment(environment)
        plan = manager.preset_plan(environment)
        assert plan.environment_id is not None
        manifest = ExtensionManifest(extension_id='runtime-provider', name=environment, version='1', description='',
                                     runtime=ExtensionRuntime(kind=plan.runtime_kind, environment=environment))
        target = SharedRuntimeTarget(manager, plan, environment)
        self.sources[plan.environment_id] = RuntimeSource(name=environment, source='official', target=target, manifest=manifest,
            release=self.official.runtime_releases.get(environment))
        self.official.shared_targets[plan.environment_id] = target
        self._register_source_alias(self.sources[plan.environment_id])

    def _register_source_alias(self, source: RuntimeSource) -> None:
        previous = source.target.manager.legacy_environment_id(source.target.environment)
        identifier = source.target.plan.environment_id
        if identifier is None:
            return
        if previous is not None:
            self._aliases[previous] = identifier
            self.official.shared_targets[previous] = source.target
        if (source.source == 'official' and self.official.has_runtime_catalog
                and self.official.runtime_releases[source.name].version is None
                and source.target.manager is self.official.for_environment(source.name)):
            for old_id, name in self._source_history.items():
                if name == source.name and old_id != identifier:
                    self._aliases[old_id] = identifier
                    self.official.shared_targets[old_id] = source.target
            if identifier not in self._source_history:
                self._source_history[identifier] = source.name
                self.definitions.mkdir(parents=True, exist_ok=True)
                temporary = self._source_history_file.with_suffix('.tmp')
                temporary.write_bytes(msgspec.json.encode(self._source_history))
                temporary.replace(self._source_history_file)

    def release_definition(self, identifier: str) -> RuntimeDefinition | None:
        identifier = self.canonical_id(identifier)
        revision = self._revisions.get(identifier)
        if revision is not None:
            base = revision.request.base_environment_id
            definition = self.release_definition(base) if base is not None else None
            if definition is not None and revision.request.policy == 'adjust':
                return copy_model(definition, update={'abi': None})
            return definition
        source = self.sources.get(identifier)
        name = source.name if source is not None else identifier
        return source.release if source is not None else self.official.runtime_releases.get(name)

    def validate_compatibility(self, manifest: ExtensionManifest, identifier: str | None = None) -> None:
        runtime = manifest.runtime
        if runtime.provider_id is None and runtime.provider_version is None and runtime.abi is None:
            return
        definition = self.release_definition(identifier or runtime.environment or '')
        if definition is None:
            raise InvalidRequestError('Selected runtime has no publisher identity; choose a versioned runtime')
        if runtime.provider_id is not None and definition.provider_id != runtime.provider_id:
            raise InvalidRequestError(f'Extension requires runtime provider {runtime.provider_id}')
        if runtime.provider_version is not None and (definition.version is None or not
                SpecifierSet(runtime.provider_version).contains(definition.version, prereleases=True)):
            raise InvalidRequestError(f'Runtime version {definition.version} does not satisfy {runtime.provider_version}')
        if runtime.abi is not None and runtime.abi != definition.abi:
            raise InvalidRequestError(f'Extension requires runtime ABI {runtime.abi}; selected ABI is {definition.abi}')

    def canonical_id(self, identifier: str) -> str:
        return self._aliases.get(identifier, identifier)

    def _revision_workspace(self, revision: EnvironmentRevision) -> Path:
        if revision.resolved_id is not None:
            return self.storage / 'runtimes' / revision.resolved_id
        return self.definitions / revision.environment_id

    def _register_revision(self, identifier: str) -> None:
        revision = self._revisions[identifier]
        root = self._revision_workspace(revision)
        if not (root / 'pixi.lock').is_file():
            return
        manager = EnvironmentManager(self.data_dir, root)
        plan = manager.workspace_plan(identifier, 'runtime')
        target = SharedRuntimeTarget(manager, plan, 'runtime', available=revision.state == 'ready')
        self.sources[identifier] = RuntimeSource(name=revision.request.name, source='developer', target=target)
        self.official.shared_targets[identifier] = target

    def source(self, identifier: str) -> RuntimeSource:
        identifier = self._aliases.get(identifier, identifier)
        source = self.sources.get(identifier)
        if source is None:
            raise NotFoundError(f'Environment is not prepared or does not exist: {identifier}')
        return source

    def source_snapshot(self) -> dict[str, RuntimeSource]:
        with self._lock:
            return dict(self.sources)

    def revisions(self) -> tuple[EnvironmentRevision, ...]:
        with self._lock:
            return tuple(self._revisions.values())

    def status(self, identifier: str) -> EnvironmentStatus:
        identifier = self._aliases.get(identifier, identifier)
        revision = self._revisions.get(identifier)
        if revision is not None:
            source = self.sources.get(identifier)
            ready = (revision.state == 'ready' and source is not None
                     and source.target.manager.ready(source.target.plan.environment_id))
            state = revision.state if revision.state != 'ready' or ready else 'missing'
            detail = (self._operation.detail if self._operation is not None and self._operation.extension_id == identifier
                      else revision.detail)
            return EnvironmentStatus(environment_id=identifier, runtime_kind='pixi', extension_ids=(), ready=ready,
                                     name=revision.request.name, source='developer', revision=identifier.removeprefix('dev-')[:12],
                                     state=state, detail=detail, base_environment_id=revision.request.base_environment_id,
                                     pinned=revision.pinned)
        source = self.source(identifier)
        ready = source.target.manager.ready(source.target.plan.environment_id)
        prefix_exists = (source.target.manager.workspace_python_exists(source.target.environment)
                         or source.target.manager.development_python_exists(source.target.environment))
        state = 'ready' if ready else 'changed' if source.target.plan.runtime_kind == 'workspace' and prefix_exists else 'missing'
        return EnvironmentStatus(environment_id=identifier, runtime_kind=source.target.plan.runtime_kind,
                                 extension_ids=(), ready=ready, name=source.name, source=source.source,
                                 revision=identifier.rsplit('-', 1)[-1][:12],
                                 state='preparing' if self._operation is not None and self._operation.extension_id == identifier
                                 else 'failed' if identifier in self._failures else state,
                                 detail=self._operation.detail if self._operation is not None and self._operation.extension_id == identifier
                                 else self._progress.get(identifier, ''))

    def create(self, request: EnvironmentCreateRequest) -> EnvironmentStatus:
        with self._lock:
            if self.busy:
                raise ConflictError('Wait for environment preparation to finish')
            if not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', request.name):
                raise InvalidRequestError('Environment name must start with a lowercase letter and use letters, numbers, _ or -')
            base = self.source(request.base_environment_id) if request.base_environment_id else None
            if base is not None and base.target.plan.runtime_kind == 'bundled':
                raise InvalidRequestError('This bundled interpreter has no reproducible Pixi snapshot to derive from')
            base_revision = base.target.manager.identity(base.target.environment) if base else None
            identity = hashlib.sha256(msgspec.json.encode((request, base_revision))).hexdigest()
            identifier = f'dev-{identity}'
            if identifier in self._revisions:
                return self.status(identifier)
            workspace = self.definitions / identifier
            workspace.mkdir(parents=True, exist_ok=False)
            try:
                manifest: JsonObject = (selected_manifest(base.target.manager.source_root, base.target.environment) if base else {
                    'workspace': {'channels': ['conda-forge'], 'platforms': [self._platform()]},
                    'dependencies': {'python': request.python}, 'feature': {},
                    'environments': {'runtime': {'features': [], 'no-default-feature': False}},
                })
                old_environments = object_value(manifest['environments'], 'Environment definitions')
                env_settings = next(iter(old_environments.values()))
                manifest['environments'] = {'runtime': env_settings}
                workspace_settings = object_value(manifest['workspace'], 'Workspace')
                workspace_settings['name'] = 'f8-developer-runtime'
                features = object_value(manifest.get('feature', {}), 'Features')
                additions: JsonObject = {'dependencies': {}, 'pypi-dependencies': {}}
                for dependency in request.conda_dependencies:
                    match = re.fullmatch(r'([a-zA-Z0-9_][a-zA-Z0-9_.-]*)\s*(.*)', dependency.strip())
                    if match is None:
                        raise InvalidRequestError(f'Invalid Conda requirement: {dependency}')
                    conda_requirements = object_value(additions['dependencies'], 'Conda dependencies')
                    if match[1] in conda_requirements:
                        raise InvalidRequestError(f'Duplicate Conda requirement: {match[1]}')
                    conda_requirements[match[1]] = match[2] or '*'
                for value in request.pypi_dependencies:
                    requirement = Requirement(value)
                    if requirement.url is not None or requirement.marker is not None:
                        raise InvalidRequestError('Developer requirements must use package versions; URL and conditional requirements are unsupported')
                    pypi_requirements = object_value(additions['pypi-dependencies'], 'PyPI dependencies')
                    if any(canonicalize_name(name) == canonicalize_name(requirement.name) for name in pypi_requirements):
                        raise InvalidRequestError(f'Duplicate PyPI requirement: {requirement.name}')
                    pypi_requirements[requirement.name] = {
                        'version': str(requirement.specifier) or '*', 'extras': sorted(requirement.extras),
                    }
                if request.policy == 'adjust':
                    self._remove_overridden(manifest, additions)
                elif base is not None:
                    pins = self._base_pins(base)
                    pins_name = 'f8-base-lock' if 'f8-base-lock' not in features else f'f8-base-lock-{identity[:12]}'
                    features[pins_name] = pins
                    self._append_feature(env_settings, pins_name)
                additions_name = 'f8-additions' if 'f8-additions' not in features else f'f8-additions-{identity[:12]}'
                features[additions_name] = additions
                manifest['feature'] = features
                self._append_feature(env_settings, additions_name)
                if base is not None:
                    baseline = selected_lock(base.target.manager.source_root, base.target.environment)
                    (workspace / 'base-lock.json').write_bytes(msgspec.json.encode(baseline))
                    relocate_dependencies(manifest, base.target.manager.source_root, workspace,
                                          allowed_root=base.target.manager.dependency_root)
                write_manifest(workspace / 'pixi.toml', manifest)
                self._revisions[identifier] = EnvironmentRevision(environment_id=identifier, request=request,
                                                                  base_revision=base_revision)
                self._save()
            except (OSError, ValueError, msgspec.DecodeError) as exc:
                shutil.rmtree(workspace)
                self._revisions.pop(identifier, None)
                logger.exception('Cannot create environment revision %s', request.name)
                if isinstance(exc, InvalidRequestError):
                    raise
                raise InvalidRequestError(f'Cannot create environment revision: {exc}') from exc
            return self.status(identifier)

    @staticmethod
    def _append_feature(settings: F8JsonValue, name: str) -> None:
        config = object_value(settings, 'Environment settings')
        names = config.get('features', [])
        if not isinstance(names, list):
            raise InvalidRequestError('Invalid feature names')
        if name not in names:
            names.append(name)
        config['features'] = names

    @staticmethod
    def _remove_overridden(manifest: JsonObject, additions: JsonObject) -> None:
        for key, table in tuple(manifest.items()):
            if not isinstance(table, dict):
                continue
            if key in {'dependencies', 'pypi-dependencies'}:
                names = object_value(additions.get(key, {}), 'Additional requirements')
                for name in tuple(table):
                    if any(canonicalize_name(name) == canonicalize_name(added) for added in names):
                        table.pop(name)
            else:
                RuntimeRegistry._remove_overridden(table, additions)

    @staticmethod
    def _base_pins(base: RuntimeSource) -> JsonObject:
        lock = selected_lock(base.target.manager.source_root, base.target.environment)
        if lock is None:
            raise InvalidRequestError('The base needs a valid Pixi lock file before its package versions can be preserved')
        pins: JsonObject = {'dependencies': {}, 'pypi-dependencies': {}}
        environment = object_value(object_value(lock['environments'], 'Lock environments')[base.target.environment], 'Locked environment')
        platforms = object_value(environment.get('packages', {}), 'Locked packages')
        metadata = lock.get('packages', [])
        if not isinstance(metadata, list):
            raise InvalidRequestError('Invalid package metadata')
        target_pins: JsonObject = {}
        platform_definitions = lock.get('platforms', [])
        for platform_name, references in platforms.items():
            if not isinstance(references, list):
                raise InvalidRequestError('Invalid package references')
            subdir = platform_name
            if isinstance(platform_definitions, list):
                for entry in platform_definitions:
                    definition = object_value(entry, 'Platform definition')
                    if definition.get('name') == platform_name:
                        subdir = str(definition.get('subdir', platform_name))
            platform_pins: JsonObject = {'dependencies': {}, 'pypi-dependencies': {}}
            for reference in references:
                package = object_value(reference, 'Package reference')
                conda = package.get('conda')
                pypi = package.get('pypi')
                if isinstance(conda, str):
                    filename = Path(urlparse(conda).path).name.removesuffix('.conda').removesuffix('.tar.bz2')
                    name, version, build = filename.rsplit('-', 2)
                    object_value(platform_pins['dependencies'], 'Conda pins')[name] = {'version': f'=={version}', 'build': build}
                elif isinstance(pypi, str) and not pypi.startswith(('./', '../')):
                    found = next((object_value(cast(F8JsonValue, entry), 'Package metadata') for entry in metadata
                                  if isinstance(entry, dict) and cast(JsonObject, entry).get('pypi') == pypi), None)
                    if found is not None and isinstance(found.get('name'), str) and isinstance(found.get('version'), str):
                        object_value(platform_pins['pypi-dependencies'], 'PyPI pins')[str(found['name'])] = f"=={found['version']}"
            target_pins[subdir] = platform_pins
        pins['target'] = target_pins
        return pins

    @staticmethod
    def _platform() -> str:
        import platform
        import sys
        if sys.platform == 'win32':
            return 'win-64'
        if sys.platform == 'darwin':
            return 'osx-arm64' if platform.machine() == 'arm64' else 'osx-64'
        return 'linux-aarch64' if platform.machine() == 'aarch64' else 'linux-64'

    async def prepare(self, identifier: str) -> EnvironmentStatus:
        with self._lock:
            if self.busy:
                raise ConflictError('An environment is already being prepared')
            if identifier not in self._revisions:
                self.source(identifier)
            self._failures.pop(identifier, None)
            operation = InstallOperation(identifier, self.definitions / 'logs' / f'{identifier}.log')
            self._operation = operation
            self._task = asyncio.create_task(self._prepare(operation), name=f'prepare-runtime-{identifier}')
        return self.status(identifier)

    async def _prepare(self, operation: InstallOperation) -> None:
        identifier = operation.extension_id
        try:
            revision = self._revisions.get(identifier)
            if revision is not None:
                self._revisions[identifier] = copy_model(revision, update={'state': 'preparing', 'detail': ''})
                self._save()
                await asyncio.to_thread(self._prepare_revision, revision, operation)
                with self._lock:
                    current = self._revisions[identifier]
                    self._revisions[identifier] = copy_model(current, update={'state': 'ready', 'detail': ''})
                    self._register_revision(identifier)
                    self._save()
            else:
                source = self.source(identifier)
                if source.manifest is None:
                    raise InvalidRequestError('Environment source has no install declaration')
                if source.manifest.runtime.kind == 'workspace':
                    pixi = await asyncio.to_thread(source.target.manager.pixi_executable, operation)
                    await asyncio.to_thread(operation.run, [str(pixi), 'install', '--locked', '-e', source.target.environment,
                                           '--manifest-path', str(source.target.manager.source_root / 'pixi.toml')],
                                           cwd=source.target.manager.source_root, env=source.target.manager.install_environment())
                else:
                    await asyncio.to_thread(source.target.manager.ensure, source.manifest, operation)
                current_plan = source.target.manager.plan(source.manifest)
                assert current_plan.environment_id is not None
                target = SharedRuntimeTarget(source.target.manager, current_plan, source.target.environment)
                with self._lock:
                    self.sources.pop(identifier)
                    self.sources[current_plan.environment_id] = RuntimeSource(name=source.name, source=source.source,
                                                                             target=target, manifest=source.manifest)
                    self.official.shared_targets[identifier] = target
                    self.official.shared_targets[current_plan.environment_id] = target
                    self._aliases[identifier] = current_plan.environment_id
                    self._register_source_alias(self.sources[current_plan.environment_id])
                    self._progress[current_plan.environment_id] = 'Environment prepared successfully'
        except ExtensionInstallCancelled:
            self._progress[identifier] = 'Preparation cancelled'
            if identifier in self._revisions:
                self._revisions[identifier] = copy_model(self._revisions[identifier], update={'state': 'declared', 'detail': 'Preparation cancelled'})
                self._save()
        except Exception as exc:
            logger.exception('Cannot prepare runtime %s', identifier)
            message = f'{type(exc).__name__}: {exc}'
            self._progress[identifier] = message
            self._failures[identifier] = message
            if identifier in self._revisions:
                self._revisions[identifier] = copy_model(self._revisions[identifier], update={'state': 'failed', 'detail': message})
                self._save()
        finally:
            with self._lock:
                self._operation = None

    def _prepare_revision(self, revision: EnvironmentRevision, operation: InstallOperation) -> None:
        workspace = self.definitions / revision.environment_id
        pixi = self.official.pixi_executable(operation)
        operation.report('Resolving the independent runtime revision')
        operation.run([str(pixi), 'lock', '--manifest-path', str(workspace / 'pixi.toml')],
                      cwd=workspace, env=self.official.install_environment())
        resolved = selected_lock(workspace, 'runtime')
        if resolved is None:
            raise InvalidRequestError('Pixi did not produce a valid lock file')
        digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(',', ':')).encode())
        sources = workspace / 'sources'
        if sources.is_dir():
            for path in sorted(sources.rglob('*')):
                if path.is_file() and not path.is_symlink():
                    digest.update(path.relative_to(workspace).as_posix().encode())
                    with path.open('rb') as stream:
                        digest.update(hashlib.file_digest(stream, 'sha256').digest())
        identity = digest.hexdigest()
        resolved_id = f'user-{identity}'
        destination = self.storage / 'runtimes' / resolved_id
        if not destination.exists():
            temporary = destination.with_name(destination.name + '.tmp')
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(workspace, temporary)
            temporary.rename(destination)
        operation.check_cancelled()
        with self._lock:
            self._revisions[revision.environment_id] = copy_model(self._revisions[revision.environment_id], update={'resolved_id': resolved_id})
            self._save()
        operation.report('Installing the locked runtime using the shared package cache')
        operation.run([str(pixi), 'install', '--locked', '-e', 'runtime', '--manifest-path', str(destination / 'pixi.toml')],
                      cwd=destination, env=self.official.install_environment())

    async def cancel(self, identifier: str) -> EnvironmentStatus:
        operation, task = self._operation, self._task
        if operation is not None and operation.extension_id == identifier and task is not None:
            await asyncio.to_thread(operation.cancel)
            await task
        return self.status(identifier)

    async def close(self) -> None:
        if self._operation is not None:
            await self.cancel(self._operation.extension_id)

    def retain(self, identifier: str, pinned: bool) -> EnvironmentStatus:
        with self._lock:
            if self.busy:
                raise ConflictError('Wait for runtime preparation to finish')
            revision = self._revisions.get(identifier)
            if revision is None:
                raise InvalidRequestError('Only developer revisions have a retention setting')
            self._revisions[identifier] = copy_model(revision, update={'pinned': pinned})
            self._save()
            return self.status(identifier)

    def remove(self, identifier: str, referenced: set[str]) -> None:
        with self._lock:
            if self.busy:
                raise ConflictError('Wait for runtime preparation to finish')
            if identifier in referenced:
                raise ConflictError('Uninstall or rebind extensions before removing their environment')
            revision = self._revisions.get(identifier)
            if revision is None:
                raise InvalidRequestError('Official and package environments are managed by their extensions')
            if revision.pinned:
                raise ConflictError('Unpin the environment revision before removing it')
            if any(self.canonical_id(item.request.base_environment_id or '') == identifier for item in self._revisions.values()):
                raise ConflictError('Other revisions still reference this environment as their base')
            remaining = {key: value for key, value in self._revisions.items() if key != identifier}
            self._revisions = remaining
            self._save()
            self.sources.pop(identifier, None)
            self.official.shared_targets.pop(identifier, None)
            if (self.definitions / identifier).is_dir():
                shutil.rmtree(self.definitions / identifier)
            if revision.resolved_id and not any(item.resolved_id == revision.resolved_id for item in remaining.values()):
                target = self.storage / 'runtimes' / revision.resolved_id
                if target.is_dir():
                    shutil.rmtree(target)

    def detail(self, identifier: str) -> EnvironmentDetail:
        identifier = self._aliases.get(identifier, identifier)
        revision = self._revisions.get(identifier)
        source = self.sources.get(identifier)
        root = self._revision_workspace(revision) if revision is not None else self.source(identifier).target.manager.source_root
        installed_root = root if revision is not None else self.source(identifier).target.manager.workspace(self.source(identifier).target.plan)
        prefix = (installed_root / 'env' if source is not None and source.target.plan.runtime_kind == 'bundled'
                  else installed_root / '.pixi/envs' / ('runtime' if revision is not None else self.source(identifier).target.environment))
        release = self.release_definition(identifier)
        return EnvironmentDetail(environment_id=identifier, name=revision.request.name if revision else self.source(identifier).name,
                                 revision=identifier.rsplit('-', 1)[-1][:12], manifest=toml_text(selected_manifest(root, 'runtime' if revision else self.source(identifier).target.environment)),
                                 base_environment_id=revision.request.base_environment_id if revision else None,
                                 policy=revision.request.policy if revision else None,
                                 conda_dependencies=revision.request.conda_dependencies if revision else (),
                                 pypi_dependencies=revision.request.pypi_dependencies if revision else (),
                                 storage_path=str(prefix), cache_path=str(self.storage / 'package-cache'),
                                 definition_path=str(root / 'pixi.toml'),
                                 source_environment='runtime' if revision else self.source(identifier).target.environment,
                                 provider_id=release.provider_id if release else None,
                                 provider_version=release.version if release else None,
                                 abi=release.abi if release else None,
                                 usage=directory_usage(prefix), pinned=revision.pinned if revision else False,
                                 changed_packages=self._changes(root) if revision else ())

    def _changes(self, root: Path) -> tuple[str, ...]:
        if not (root / 'pixi.lock').is_file():
            return ()
        snapshot = root / 'base-lock.json'
        original = msgspec.json.decode(snapshot.read_bytes(), type=JsonObject | None) if snapshot.is_file() else None
        current = selected_lock(root, 'runtime')
        if original is None or current is None:
            return ()
        old = self._package_names(original)
        new = self._package_names(current)
        return tuple(f'{name}: {old.get(name, "absent")} → {new.get(name, "removed")}'
                     for name in sorted(old.keys() | new.keys()) if old.get(name) != new.get(name))

    @staticmethod
    def _package_names(lock: JsonObject) -> dict[str, str]:
        result: dict[str, str] = {}
        packages = lock.get('packages', [])
        if not isinstance(packages, list):
            return result
        for value in packages:
            entry = object_value(value, 'Package metadata')
            if isinstance(entry.get('conda'), str):
                filename = Path(urlparse(str(entry['conda'])).path).name.removesuffix('.conda').removesuffix('.tar.bz2')
                name, version, build = filename.rsplit('-', 2)
                result[name] = f'{version}/{build}'
            elif isinstance(entry.get('name'), str):
                result[str(entry['name'])] = str(entry.get('version', entry.get('pypi', 'local')))
        return result

    def storage_status(self) -> RuntimeStorageStatus:
        installed = any(revision.resolved_id for revision in self._revisions.values())
        managed = any(source.target.plan.runtime_kind == 'pixi' and source.target.manager.ready(source.target.plan.environment_id)
                      for source in self.sources.values() if source.source != 'developer')
        return RuntimeStorageStatus(path=str(self.storage), cache_path=str(self.storage / 'package-cache'),
                                    can_change=not self.busy and not installed and not managed)

    def set_storage(self, path: str) -> RuntimeStorageStatus:
        if not self.storage_status().can_change:
            raise ConflictError('Remove prepared managed environments before changing runtime storage')
        destination = Path(path).expanduser()
        if not destination.is_absolute():
            raise InvalidRequestError('Runtime storage must be an absolute directory path')
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
        temporary = self.data_dir / 'runtime-storage.tmp'
        temporary.write_bytes(msgspec.json.encode({'path': str(destination)}))
        temporary.replace(self.data_dir / 'runtime-storage.json')
        self.storage = destination
        self.official.set_runtime_storage(destination)
        for source in self.sources.values():
            source.target.manager.root = destination / 'runtimes'
        return self.storage_status()
