from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import shutil
import sys
from threading import RLock
from typing import Literal
import zipfile

import msgspec
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import yaml

from f8pysdk.codec import copy_model, validate_as
from f8pysdk.monitoring import validate_describe_monitor_contract
from f8pysdk.service_runtime_tools.inventory import ServiceCatalog
from f8pysdk.service_runtime_tools.inventory.index import (
    IndexedService, ServiceIndex, default_service_index, indexed_entry,
    load_index_into_catalog, read_service_index,
)
from f8pysdk.specs import F8ServiceDescribe, F8ServiceEntry

from .environments import EnvironmentManager
from .extension_artifacts import prepare_artifact
from .errors import ConflictError, InvalidRequestError, NotFoundError
from .extension_models import (
    EnvironmentStatus, ExtensionCatalog, ExtensionImportRequest, ExtensionInstallPlan, ExtensionManifest,
    ExtensionRecord, ExtensionStatus, PresetEnvironmentStatus,
)
from .extension_operation import ExtensionInstallCancelled, InstallOperation
from .shared_dependencies import RuntimeProbe, validate_shared_package


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtensionPayload:
    root: Path
    index_path: Path
    index: ServiceIndex
    environments: EnvironmentManager


class ExtensionManager:
    def __init__(self, data_dir: Path, *, base_index: Path | None = None) -> None:
        self._root = data_dir / 'extensions'
        self._data_dir = data_dir
        self._base_index = (base_index or default_service_index()).resolve()
        self._source_root = self._base_index.parent.parent
        self.environments = EnvironmentManager(data_dir, self._source_root)
        self._lock = RLock()
        self._actions = asyncio.Lock()
        self._operation: InstallOperation | None = None
        self._task: asyncio.Task[None] | None = None
        self._failures: dict[str, str] = {}
        self._details: dict[str, str] = {}
        self._state_path = self._root / 'state.json'
        self._sources_path = self._root / 'sources.json'
        self._source_digests: list[str] = []
        self._payloads: dict[str, ExtensionPayload] = {}
        self._records: dict[str, ExtensionRecord] = {}
        self._manifests: dict[str, ExtensionManifest] = {}
        self._services: dict[str, IndexedService] = {}
        self._owners: dict[str, str] = {}
        self._preinstalled: set[str] = set()
        catalog_path = self._base_index.with_name('extensions.json')
        self.has_catalog = catalog_path.is_file()
        if not self.has_catalog:
            return
        self._add_catalog(self._source_root, preinstalled=True)
        if self._sources_path.is_file():
            try:
                self._source_digests = msgspec.json.decode(self._sources_path.read_bytes(), type=list[str])
            except (OSError, msgspec.DecodeError):
                logger.warning('Cannot read extension sources; using the bundled catalog', exc_info=True)
            for digest in self._source_digests:
                if not re.fullmatch(r'[0-9a-f]{64}', digest):
                    logger.warning('Ignoring invalid extension source digest: %r', digest)
                    continue
                try:
                    self._add_catalog(self._root / 'payloads' / digest, preinstalled=False)
                except (OSError, ValueError, msgspec.DecodeError, ConflictError):
                    logger.exception('Cannot load imported extension source %s', digest)
        state_readable = True
        if self._state_path.is_file():
            try:
                self._records = msgspec.json.decode(self._state_path.read_bytes(), type=dict[str, ExtensionRecord])
            except (OSError, msgspec.DecodeError):
                logger.warning('Cannot read extension installation state; using the distribution preset', exc_info=True)
                state_readable = False
            unknown = set(self._records) - self._manifests.keys()
            if unknown:
                logger.warning('Installed extensions absent from this distribution: %s', sorted(unknown))
        for manifest in self._manifests.values():
            record = self._records.get(manifest.extension_id)
            if record is None and manifest.extension_id in self._preinstalled and self._supported(manifest):
                plan = self._payloads[manifest.extension_id].environments.plan(manifest)
                record = ExtensionRecord(version=manifest.version, installed=True, enabled=True,
                                         environment_id=plan.environment_id)
                self._records[manifest.extension_id] = record
            if record is not None and record.installed:
                if record.version != manifest.version:
                    logger.warning('Extension %s changed from %s to %s; explicit reinstall is required',
                                   manifest.extension_id, record.version, manifest.version)
                    self._records[manifest.extension_id] = copy_model(record, update={'installed': False, 'enabled': False})
                    continue
                try:
                    self._copy_model_metadata(manifest)
                    self._write_registration(manifest, record)
                    load_index_into_catalog(path=self._registration(manifest.extension_id), catalog=ServiceCatalog())
                except (OSError, ValueError, msgspec.DecodeError) as exc:
                    logger.exception('Cannot restore extension %s', manifest.extension_id)
                    self._records[manifest.extension_id] = copy_model(record, update={'installed': False, 'enabled': False})
                    self._failures[manifest.extension_id] = f'{type(exc).__name__}: {exc}'
        if state_readable:
            self._save_records()

    def _add_catalog(self, root: Path, *, preinstalled: bool) -> tuple[str, ...]:
        root = root.resolve()
        index_path = root / 'config/service-index.json'
        catalog = msgspec.json.decode((root / 'config/extensions.json').read_bytes(), type=ExtensionCatalog)
        index = read_service_index(index_path)
        services = {item.serviceClass: item for item in index.services}
        owners = self._validate_catalog(catalog, services)
        if not preinstalled and any(manifest.runtime.kind not in {'native', 'pixi', 'shared'} for manifest in catalog.extensions):
            raise ValueError('Published extensions must declare a native, shared, or locked Pixi runtime')
        conflicts = set(services) & self._services.keys()
        ids = {manifest.extension_id for manifest in catalog.extensions}
        if conflicts or ids & self._manifests.keys():
            raise ConflictError('Extension package conflicts with an existing extension ID or service class')
        for item in index.services:
            for relative in (*item.manifests.values(), item.describe):
                path = (index_path.parent / relative).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f'Missing or unsafe payload path for {item.serviceClass}: {relative}')
        environments = self.environments if root == self._source_root else EnvironmentManager(
            self._data_dir, root, official=self.environments.official,
        )
        payload = ExtensionPayload(root=root, index_path=index_path, index=index, environments=environments)
        for manifest in catalog.extensions:
            self._manifests[manifest.extension_id] = manifest
            self._payloads[manifest.extension_id] = payload
        self._services.update(services)
        self._owners.update(owners)
        if preinstalled:
            self._preinstalled.update(catalog.preinstalled)
        return tuple(manifest.extension_id for manifest in catalog.extensions)

    def _validate_catalog(self, catalog: ExtensionCatalog, services: dict[str, IndexedService]) -> dict[str, str]:
        ids: set[str] = set()
        owners: dict[str, str] = {}
        for manifest in catalog.extensions:
            if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}', manifest.extension_id):
                raise ValueError(f'Invalid extension ID: {manifest.extension_id!r}')
            if manifest.extension_id in ids or not manifest.version or not manifest.service_classes:
                raise ValueError(f'Duplicate or invalid extension: {manifest.extension_id}')
            ids.add(manifest.extension_id)
            runtime = manifest.runtime
            if runtime.kind in {'pixi', 'workspace', 'shared'} and not runtime.environment:
                raise ValueError(f'Missing environment for {manifest.extension_id}')
            if runtime.kind == 'shared':
                if runtime.requires_python is not None:
                    SpecifierSet(runtime.requires_python)
                for dependency in runtime.dependencies:
                    requirement = Requirement(dependency)
                    if requirement.url is not None:
                        raise ValueError(f'Shared extensions cannot declare URL dependencies: {dependency}')
            elif runtime.requires_python is not None or runtime.dependencies:
                raise ValueError(f'Only shared extensions can declare official runtime requirements: {manifest.extension_id}')
            if runtime.environment is not None and not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', runtime.environment):
                raise ValueError(f'Invalid environment name for {manifest.extension_id}')
            for directory in manifest.model_directories:
                if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', directory):
                    raise ValueError(f'Invalid model directory for {manifest.extension_id}')
            for service_class in manifest.service_classes:
                if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]*', service_class):
                    raise ValueError(f'Invalid service class for {manifest.extension_id}')
                if service_class not in services:
                    raise ValueError(f'Missing service {service_class} for {manifest.extension_id}')
                if service_class in owners:
                    raise ValueError(f'Service {service_class} has multiple extension owners')
                owners[service_class] = manifest.extension_id
        if set(catalog.preinstalled) - ids:
            raise ValueError('Preinstalled extensions must be declared in the catalog')
        if set(services) - owners.keys():
            raise ValueError(f'Services without an extension owner: {sorted(set(services) - owners.keys())}')
        return owners

    def _manifest(self, extension_id: str) -> ExtensionManifest:
        manifest = self._manifests.get(extension_id)
        if manifest is None:
            raise NotFoundError(f'Unknown extension: {extension_id}')
        return manifest

    def _supported(self, manifest: ExtensionManifest) -> bool:
        return all(sys.platform in self._services[name].manifests or 'any' in self._services[name].manifests
                   for name in manifest.service_classes)

    def _registration(self, extension_id: str) -> Path:
        return self._root / 'registrations' / extension_id / 'service-index.json'

    def _save_records(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        temporary = self._state_path.with_suffix('.tmp')
        temporary.write_bytes(msgspec.json.encode(self._records))
        temporary.replace(self._state_path)

    def _commit_record(self, extension_id: str, record: ExtensionRecord | None) -> None:
        with self._lock:
            previous = self._records.get(extension_id)
            if record is None:
                self._records.pop(extension_id, None)
            else:
                self._records[extension_id] = record
            try:
                self._save_records()
            except OSError:
                if previous is None:
                    self._records.pop(extension_id, None)
                else:
                    self._records[extension_id] = previous
                logger.exception('Cannot persist extension state for %s', extension_id)
                raise

    def active_indexes(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(self._registration(extension_id) for extension_id in self._manifests
                         if (record := self._records.get(extension_id)) is not None
                         and record.installed and record.enabled and extension_id not in self._failures)

    async def _refresh_record(self, extension_id: str, record: ExtensionRecord,
                              refresh: Callable[[], object]) -> None:
        previous = self._records.get(extension_id)
        self._commit_record(extension_id, record)
        try:
            await asyncio.to_thread(refresh)
        except Exception:
            logger.exception('Cannot refresh catalog after updating extension %s', extension_id)
            self._commit_record(extension_id, previous)
            raise

    def statuses(self) -> tuple[ExtensionStatus, ...]:
        return tuple(self.status(extension_id) for extension_id in self._manifests)

    def status(self, extension_id: str) -> ExtensionStatus:
        manifest = self._manifest(extension_id)
        with self._lock:
            record = self._records.get(extension_id)
            detail = self._details.get(extension_id, '')
            state: Literal['unavailable', 'available', 'installing', 'installed', 'disabled', 'failed']
            if self._operation is not None and self._operation.extension_id == extension_id:
                state, detail = 'installing', self._operation.detail
            elif extension_id in self._failures:
                state, detail = 'failed', self._failures[extension_id]
            elif not self._supported(manifest):
                state, detail = 'unavailable', 'This extension does not support this platform.'
            elif record is not None and record.installed:
                state = 'installed' if record.enabled else 'disabled'
            else:
                state = 'available'
            return ExtensionStatus(
                extension_id=extension_id, name=manifest.name, version=manifest.version,
                description=manifest.description, state=state, detail=detail,
                service_classes=manifest.service_classes, runtime_kind=manifest.runtime.kind,
                environment_id=record.environment_id if record is not None and record.installed else None,
                preinstalled=extension_id in self._preinstalled,
            )

    def install_plan(self, extension_id: str) -> ExtensionInstallPlan:
        manifest = self._manifest(extension_id)
        if not self._supported(manifest):
            raise InvalidRequestError('This extension does not support this platform')
        for name in manifest.service_classes:
            self._entry(manifest, name)
        return self._payloads[extension_id].environments.plan(manifest)

    def preset_environments(self) -> tuple[PresetEnvironmentStatus, ...]:
        return self.environments.presets()

    def _entry(self, manifest: ExtensionManifest, service_class: str) -> F8ServiceEntry:
        payload = self._payloads[manifest.extension_id]
        entry = indexed_entry(payload.index_path, payload.index, self._services[service_class])
        if entry is None:
            raise InvalidRequestError(f'Missing launcher for {service_class}')
        args = entry.launch.args or []
        if manifest.runtime.kind in {'pixi', 'workspace'}:
            if (Path(entry.launch.command).name not in {'pixi', 'pixi.exe'} or len(args) != 4
                    or args[:3] != ['run', '-e', manifest.runtime.environment]):
                raise InvalidRequestError(f'Service {service_class} must explicitly use the declared extension environment')
        elif manifest.runtime.kind == 'shared':
            if (entry.launch.command not in {'python', 'python.exe'} or len(args) < 2 or args[0] != '-m'
                    or not re.fullmatch(r'[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*', args[1])):
                raise InvalidRequestError(f'Shared service {service_class} must launch an explicit Python module')
            source = payload.root / 'python'
            module_path = source.joinpath(*args[1].split('.'))
            if not source.resolve().is_relative_to(payload.root):
                raise InvalidRequestError('Shared extension Python directory escapes its payload')
            if not module_path.with_suffix('.py').is_file() and not (module_path / '__main__.py').is_file():
                raise InvalidRequestError(f'Missing Python entrypoint for {service_class}: {args[1]}')
        return entry

    def environment_statuses(self) -> tuple[EnvironmentStatus, ...]:
        with self._lock:
            users: dict[str, list[str]] = {}
            for extension_id, record in self._records.items():
                if record.installed and record.environment_id is not None and extension_id in self._manifests:
                    users.setdefault(record.environment_id, []).append(extension_id)
            return tuple(EnvironmentStatus(
                environment_id=environment_id, extension_ids=tuple(sorted(extension_ids)),
                runtime_kind=self._manifests[extension_ids[0]].runtime.kind,
                ready=self._payloads[extension_ids[0]].environments.ready(environment_id),
            ) for environment_id, extension_ids in sorted(users.items()))

    def _require_idle(self) -> None:
        if self._operation is not None:
            raise ConflictError(f'Extension operation is running: {self._operation.extension_id}')

    def service_enabled(self, service_class: str) -> bool:
        with self._lock:
            owner = self._owners.get(service_class)
            if owner is None:
                return True
            record = self._records.get(owner)
            return record is not None and record.installed and record.enabled and owner not in self._failures

    async def import_package(self, request: ExtensionImportRequest) -> tuple[ExtensionStatus, ...]:
        async with self._actions:
            with self._lock:
                self._require_idle()
                if request.sha256 in self._source_digests:
                    return self.statuses()
            try:
                payload = await asyncio.to_thread(prepare_artifact, request, self._root)
                with self._lock:
                    added = self._add_catalog(payload, preinstalled=False)
                    sources = [*self._source_digests, request.sha256]
                    temporary = self._sources_path.with_suffix('.tmp')
                    try:
                        temporary.write_bytes(msgspec.json.encode(sources))
                        temporary.replace(self._sources_path)
                    except OSError:
                        for extension_id in added:
                            manifest = self._manifests.pop(extension_id)
                            self._payloads.pop(extension_id)
                            for name in manifest.service_classes:
                                self._services.pop(name)
                                self._owners.pop(name)
                        raise
                    self._source_digests = sources
            except (OSError, ValueError, msgspec.DecodeError, zipfile.BadZipFile) as exc:
                logger.exception('Cannot import extension package from %s', request.url)
                raise InvalidRequestError(f'Cannot import extension package: {exc}') from exc
            return self.statuses()

    async def install(self, extension_id: str, refresh: Callable[[], object]) -> ExtensionStatus:
        async with self._actions:
            await asyncio.to_thread(self.install_plan, extension_id)
            with self._lock:
                self._require_idle()
                record = self._records.get(extension_id)
                if record is not None and record.installed and extension_id not in self._failures:
                    return self.status(extension_id)
                operation = InstallOperation(extension_id, self._root / 'logs' / f'{extension_id}.log')
                self._operation = operation
                self._failures.pop(extension_id, None)
                self._details.pop(extension_id, None)
                self._task = asyncio.create_task(self._install_and_refresh(operation, refresh),
                                                 name=f'install-extension-{extension_id}')
                self._task.add_done_callback(self._report_task_failure)
            return self.status(extension_id)

    async def _install_and_refresh(self, operation: InstallOperation, refresh: Callable[[], object]) -> None:
        extension_id = operation.extension_id
        manifest = self._manifest(extension_id)
        previous = self._records.get(extension_id)
        try:
            plan = await asyncio.to_thread(self._payloads[extension_id].environments.ensure, manifest, operation)
            record = ExtensionRecord(version=manifest.version, installed=True, enabled=True,
                                     environment_id=plan.environment_id)
            await asyncio.to_thread(self._prepare_install, manifest, record, operation)
            operation.check_cancelled()
            self._commit_record(extension_id, record)
            await asyncio.to_thread(refresh)
        except ExtensionInstallCancelled:
            self._commit_record(extension_id, previous)
            with self._lock:
                self._details[extension_id] = 'Installation cancelled'
        except Exception as exc:
            logger.exception('Extension installation failed: %s', extension_id)
            with self._lock:
                self._failures[extension_id] = f'{type(exc).__name__}: {exc}'
            self._commit_record(extension_id, previous)
        finally:
            with self._lock:
                self._operation = None

    @staticmethod
    def _report_task_failure(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error('Extension installation task failed unexpectedly',
                         exc_info=(type(error), error, error.__traceback__))

    def _referenced_environments(self) -> set[str]:
        return {record.environment_id for record in self._records.values()
                if record.installed and record.environment_id is not None}

    async def cancel(self, extension_id: str) -> ExtensionStatus:
        self._manifest(extension_id)
        with self._lock:
            operation, task = self._operation, self._task
        if operation is None or operation.extension_id != extension_id or task is None:
            return self.status(extension_id)
        await asyncio.to_thread(operation.cancel)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=20)
        except TimeoutError:
            await asyncio.to_thread(operation.cancel, force=True)
            await asyncio.shield(task)
        return self.status(extension_id)

    async def close(self) -> None:
        with self._lock:
            operation = self._operation
        if operation is not None:
            await self.cancel(operation.extension_id)

    async def set_enabled(self, extension_id: str, enabled: bool, refresh: Callable[[], object],
                          is_running: Callable[[str], bool]) -> ExtensionStatus:
        async with self._actions:
            manifest = self._manifest(extension_id)
            with self._lock:
                self._require_idle()
                previous = self._records.get(extension_id)
                if previous is None or not previous.installed:
                    raise InvalidRequestError('Install the extension before changing its state')
                if not enabled and any(is_running(name) for name in manifest.service_classes):
                    raise ConflictError('Stop running services before disabling the extension')
                if previous.enabled == enabled:
                    return self.status(extension_id)
            await self._refresh_record(extension_id, copy_model(previous, update={'enabled': enabled}), refresh)
            return self.status(extension_id)

    async def uninstall(self, extension_id: str, refresh: Callable[[], object],
                        is_running: Callable[[str], bool]) -> ExtensionStatus:
        async with self._actions:
            manifest = self._manifest(extension_id)
            with self._lock:
                self._require_idle()
                previous = self._records.get(extension_id)
                if previous is None or not previous.installed:
                    return self.status(extension_id)
                if any(is_running(name) for name in manifest.service_classes):
                    raise ConflictError('Stop running services before uninstalling the extension')
            await self._refresh_record(extension_id, ExtensionRecord(version=manifest.version, installed=False, enabled=False),
                                       refresh)
            with self._lock:
                referenced = self._referenced_environments()
            await asyncio.to_thread(self._payloads[extension_id].environments.remove_unused,
                                    previous.environment_id, referenced)
            await asyncio.to_thread(shutil.rmtree, self._registration(extension_id).parent)
            return self.status(extension_id)

    def _write_registration(self, manifest: ExtensionManifest, record: ExtensionRecord) -> None:
        payload = self._payloads[manifest.extension_id]
        destination = self._registration(manifest.extension_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        services: list[IndexedService] = []
        model_root = str((payload.index_path.parent / payload.index.modelRoot).resolve())
        if manifest.model_directories:
            model_root = str((self._data_dir / 'models').resolve())
        plan = payload.environments.plan(manifest)
        if manifest.runtime.kind in {'pixi', 'shared'}:
            if record.environment_id != plan.environment_id or not payload.environments.ready(record.environment_id):
                raise ValueError(f'Environment for {manifest.extension_id} changed or is missing; reinstall the extension')
        for name in manifest.service_classes:
            item = self._services[name]
            entry = self._entry(manifest, name)
            environment = manifest.runtime.environment
            if manifest.runtime.kind == 'pixi':
                assert environment is not None
                command, args = payload.environments.launch(plan, environment)
                entry = copy_model(entry, update={'launch': copy_model(entry.launch, update={
                    'command': command, 'args': [*args, (entry.launch.args or [])[-1]],
                    'workdir': str(payload.environments.workspace(plan)),
                })})
            elif manifest.runtime.kind == 'shared':
                assert environment is not None
                python_root = destination.parent / 'python'
                if not python_root.is_dir():
                    raise ValueError(f'Python package for {manifest.extension_id} is missing; reinstall the extension')
                command, args = payload.environments.python_launch(plan, environment)
                entry = copy_model(entry, update={'launch': copy_model(entry.launch, update={
                    'command': command, 'args': [*args, str(Path(__file__).with_name('_shared_entrypoint.py')),
                                               str(python_root), *(entry.launch.args or [])[1:]],
                    'workdir': str(destination.parent),
                })})
            env = dict(entry.launch.env or {})
            env['F8_MODEL_ROOT'] = model_root
            entry = copy_model(entry, update={'launch': copy_model(entry.launch, update={'env': env})})
            entry_path = destination.parent / f'{name}.yml'
            entry_path.write_text(yaml.safe_dump(msgspec.to_builtins(entry), sort_keys=False), encoding='utf-8')
            describe = (payload.index_path.parent / item.describe).resolve()
            if not describe.is_relative_to(payload.root):
                raise ValueError(f'Description for {name} is outside the extension payload')
            services.append(IndexedService(serviceClass=name, manifests={'any': entry_path.name}, describe=str(describe)))
        index = ServiceIndex(schemaVersion='f8serviceIndex/1', services=tuple(services), modelRoot=model_root)
        temporary = destination.with_suffix('.tmp')
        temporary.write_bytes(msgspec.json.encode(index))
        temporary.replace(destination)

    def _copy_model_metadata(self, manifest: ExtensionManifest) -> None:
        for directory in manifest.model_directories:
            source = self._payloads[manifest.extension_id].root / 'resources' / 'models' / directory
            destination = self._data_dir / 'models' / directory
            destination.mkdir(parents=True, exist_ok=True)
            for metadata in sorted(source.glob('*.yaml')):
                if not (destination / metadata.name).exists():
                    shutil.copy2(metadata, destination / metadata.name)

    def _prepare_install(self, manifest: ExtensionManifest, record: ExtensionRecord,
                         operation: InstallOperation) -> None:
        operation.check_cancelled()
        if manifest.runtime.kind == 'shared':
            payload = self._payloads[manifest.extension_id]
            environment = manifest.runtime.environment
            assert environment is not None
            plan = payload.environments.plan(manifest)
            command, args = payload.environments.python_launch(plan, environment)
            operation.report(f'Checking dependencies against official {environment}')
            output = operation.run([command, *args, str(Path(__file__).with_name('_runtime_probe.py'))],
                                   cwd=payload.environments.workspace(plan), timeout=120)
            validate_shared_package(manifest, payload.root / 'python',
                                    msgspec.json.decode(output.encode(), type=RuntimeProbe))
            operation.check_cancelled()
            destination = self._registration(manifest.extension_id).parent / 'python'
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(payload.root / 'python', destination)
        self._copy_model_metadata(manifest)
        self._write_registration(manifest, record)
        catalog = ServiceCatalog()
        load_index_into_catalog(path=self._registration(manifest.extension_id), catalog=catalog)
        for service_class in manifest.service_classes:
            operation.check_cancelled()
            entry = catalog.service_entry(service_class)
            if entry is None:
                raise ValueError(f'Missing launcher for {service_class}')
            operation.report(f'Checking {service_class}')
            output = operation.run([entry.launch.command, *(entry.launch.args or []),
                                    *(entry.describeArgs or ['--describe'])],
                                   cwd=Path(entry.launch.workdir or self._source_root), timeout=120,
                                   env={**os.environ, **(entry.launch.env or {})})
            payload = msgspec.json.decode(output.encode(), type=dict[str, object])
            validate_describe_monitor_contract(payload)
            describe = validate_as(F8ServiceDescribe, payload)
            if describe.service.serviceClass != service_class:
                raise ValueError(f'Wrong service contract from {service_class}')
