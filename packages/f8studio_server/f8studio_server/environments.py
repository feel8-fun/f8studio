from __future__ import annotations

import hashlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tomllib
import msgspec
from typing import Literal
import urllib.request

from .extension_models import ExtensionInstallPlan, ExtensionManifest, PresetEnvironmentStatus
from .extension_operation import InstallOperation
from .errors import InvalidRequestError
from .environment_definitions import environment_identity, local_wheels, materialize_locked_environment
from .runtime_sources import read_runtime_catalog, equivalent_environment, read_runtime_sources


@dataclass(frozen=True)
class SharedRuntimeTarget:
    manager: EnvironmentManager
    plan: ExtensionInstallPlan
    environment: str
    available: bool = True


class EnvironmentManager:
    def __init__(self, data_dir: Path, source_root: Path, *, official: EnvironmentManager | None = None,
                 dependency_root: Path | None = None, development_root: Path | None = None) -> None:
        storage_file = data_dir / 'runtime-storage.json'
        storage = msgspec.json.decode(storage_file.read_bytes(), type=dict[str, str]) if storage_file.is_file() else {}
        self.root = Path(storage.get('path', str(data_dir))).resolve() / 'runtimes'
        self._source_root = source_root
        self.dependency_root = dependency_root or source_root
        self._development_root = development_root
        self._equivalence_cache: tuple[tuple[tuple[int, int], ...], bool] | None = None
        self._identity_cache: dict[str, tuple[tuple[tuple[int, int], ...], str]] = {}
        self._wheel_cache: dict[str, tuple[tuple[tuple[int, int], ...], tuple[Path, ...]]] = {}
        self._definitions: tuple[int, int, dict[str, object]] | None = None
        self._workspace_environments: dict[str, str] = {}
        self.official = official if official is not None else self
        self.shared_targets: dict[str, SharedRuntimeTarget] = {}
        self.runtime_releases = {item.runtime_id: item for item in read_runtime_catalog(source_root).runtimes}
        self._runtime_managers = {
            name: EnvironmentManager(data_dir, root, official=self, dependency_root=source_root,
                                     development_root=source_root if development == name else None)
            for name, (root, development) in read_runtime_sources(source_root).items()
        }

    @property
    def has_runtime_catalog(self) -> bool:
        return bool(self._runtime_managers)

    def set_runtime_storage(self, path: Path) -> None:
        self.root = path / 'runtimes'
        for manager in self._runtime_managers.values():
            manager.set_runtime_storage(path)

    def for_environment(self, environment: str) -> EnvironmentManager:
        return self._runtime_managers.get(environment, self)

    def legacy_environment_id(self, environment: str) -> str | None:
        bundled = environment == 'studio-runtime' and (self.official._source_root / 'env').is_dir()
        root = self.official._source_root if bundled else self._development_root
        if root is None or not (root / 'pixi.lock').is_file():
            return None
        definitions = tomllib.loads((root / 'pixi.toml').read_text(encoding='utf-8')).get('environments', {})
        if environment not in definitions:
            return None
        identity = environment_identity(root, environment)
        if bundled:
            return f'bundled-base-{identity}'
        location = hashlib.sha256(str(root).encode()).hexdigest()[:16]
        return f'workspace-{location}-{environment}-{identity[:16]}'

    def can_reuse_development_environment(self, environment: str) -> bool:
        if environment == 'studio-runtime' and (self.official._source_root / 'env').is_dir():
            return equivalent_environment(self.official._source_root, self._source_root, environment)
        return self._development_compatible(environment)

    def _development_compatible(self, environment: str) -> bool:
        root = self._development_root
        if root is None:
            return False
        files = [directory / name for directory in (root, self._source_root) for name in ('pixi.toml', 'pixi.lock')]
        if not all(path.is_file() for path in files):
            return False
        signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in files)
        if self._equivalence_cache is None or self._equivalence_cache[0] != signature:
            self._equivalence_cache = signature, equivalent_environment(root, self._source_root, environment)
        return self._equivalence_cache[1]

    def _workspace_root(self, environment: str) -> Path:
        if self._python(self._source_root / '.pixi/envs' / environment).is_file():
            return self._source_root
        if (self._development_root is not None and self._development_compatible(environment)
                and self._python(self._development_root / '.pixi/envs' / environment).is_file()):
            return self._development_root
        return self._source_root

    def _environments(self) -> dict[str, object]:
        if self._runtime_managers:
            return {name: manager._environments()[name] for name, manager in self._runtime_managers.items()}
        path = self._source_root / 'pixi.toml'
        if not path.is_file():
            return {}
        stat = path.stat()
        if self._definitions is None or self._definitions[:2] != (stat.st_mtime_ns, stat.st_size):
            definitions = tomllib.loads(path.read_text(encoding='utf-8')).get('environments', {})
            self._definitions = (stat.st_mtime_ns, stat.st_size, definitions)
        return self._definitions[2]

    @staticmethod
    def _python(prefix: Path) -> Path:
        return prefix / ('python.exe' if os.name == 'nt' else 'bin/python')

    def _identity(self, environment: str) -> str:
        manager = self.for_environment(environment)
        if manager is not self:
            return manager._identity(environment)
        definitions = (self._source_root / 'pixi.toml', self._source_root / 'pixi.lock')
        definition_signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in definitions)
        cached_wheels = self._wheel_cache.get(environment)
        if cached_wheels is None or cached_wheels[0] != definition_signature:
            cached_wheels = definition_signature, local_wheels(self._source_root, environment)
            self._wheel_cache[environment] = cached_wheels
        files = (*definitions, *cached_wheels[1])
        signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in files)
        cached = self._identity_cache.get(environment)
        if cached is not None and cached[0] == signature:
            return cached[1]
        identity = environment_identity(self._source_root, environment)
        self._identity_cache[environment] = (signature, identity)
        return identity

    def preset_names(self) -> tuple[str, ...]:
        return tuple(self._environments())

    def preset_plan(self, environment: str) -> ExtensionInstallPlan:
        return self._preset_plan("runtime-provider", environment)

    def identity(self, environment: str) -> str:
        return self._identity(environment)

    def workspace_python_exists(self, environment: str) -> bool:
        manager = self.for_environment(environment)
        return manager._python(manager._workspace_root(environment) / '.pixi/envs' / environment).is_file()

    def development_python_exists(self, environment: str) -> bool:
        return (self._development_root is not None
                and self._python(self._development_root / '.pixi/envs' / environment).is_file())

    def workspace_plan(self, extension_id: str, environment: str) -> ExtensionInstallPlan:
        return self._pixi_plan(extension_id, 'workspace', environment)

    def pixi_executable(self, operation: InstallOperation) -> Path:
        return self._pixi(operation)

    @property
    def source_root(self) -> Path:
        return self._source_root

    def install_environment(self) -> dict[str, str]:
        cache = self.root.parent / 'package-cache'
        cache.mkdir(parents=True, exist_ok=True)
        return {**os.environ, 'PIXI_CACHE_DIR': str(cache), 'UV_CACHE_DIR': str(cache / 'uv-cache')}

    def _preset_plan(self, extension_id: str, environment: str) -> ExtensionInstallPlan:
        manager = self.for_environment(environment)
        if manager is not self:
            return manager._preset_plan(extension_id, environment)
        if environment not in self._environments():
            raise InvalidRequestError(f'Official environment {environment!r} is unavailable in this distribution')
        if environment == 'studio-runtime' and (self.official._source_root / 'env').is_dir():
            return self._bundled_plan(extension_id)
        kind = 'workspace' if self._development_root is not None or self.workspace_python_exists(environment) else 'pixi'
        return self._pixi_plan(extension_id, kind, environment)

    def _bundled_plan(self, extension_id: str) -> ExtensionInstallPlan:
        manager = self.for_environment('studio-runtime')
        if manager is not self:
            return manager._bundled_plan(extension_id)
        return ExtensionInstallPlan(extension_id=extension_id,
                                    environment_id=f'bundled-base-{self._identity("studio-runtime")}',
                                    runtime_kind='bundled', action='bundled', requires_network=False)

    def presets(self) -> tuple[PresetEnvironmentStatus, ...]:
        return tuple(PresetEnvironmentStatus(environment=name,
                                              ready=self.ready(self._preset_plan('', name).environment_id))
                     for name in self._environments())

    def plan(self, manifest: ExtensionManifest) -> ExtensionInstallPlan:
        runtime = manifest.runtime
        environment = runtime.environment
        if runtime.kind == 'native':
            return ExtensionInstallPlan(extension_id=manifest.extension_id, environment_id=None,
                                        runtime_kind=runtime.kind, action='none', requires_network=False)
        if runtime.kind == 'bundled':
            return self._bundled_plan(manifest.extension_id)
        if runtime.kind == 'shared':
            if environment is None:
                raise ValueError(f'Missing environment for {manifest.extension_id}')
            target = self.official.shared_targets.get(environment)
            plan = target.plan if target is not None else self.official._preset_plan(manifest.extension_id, environment)
            owner = target.manager if target is not None else self.official
            if (target is not None and not target.available) or not owner.ready(plan.environment_id):
                raise InvalidRequestError(f'Environment {environment} is not installed or prepared; prepare it first')
            return ExtensionInstallPlan(extension_id=manifest.extension_id,
                                        environment_id=plan.environment_id,
                                        runtime_kind='shared', action='shared', requires_network=False)
        if environment is None:
            raise ValueError(f'Missing environment for {manifest.extension_id}')
        return self._pixi_plan(manifest.extension_id, runtime.kind, environment)

    def _pixi_plan(self, extension_id: str, kind: Literal['workspace', 'pixi'],
                   environment: str) -> ExtensionInstallPlan:
        manager = self.for_environment(environment)
        if manager is not self:
            return manager._pixi_plan(extension_id, kind, environment)
        if environment not in self._environments():
            raise InvalidRequestError(f'Extension {extension_id} references an undeclared environment: {environment}')
        identity = self._identity(environment)
        location = hashlib.sha256(str(self._source_root).encode()).hexdigest()[:16]
        if kind == 'workspace':
            environment_id = f'workspace-{location}-{environment}-{identity[:16]}'
            self._workspace_environments[environment_id] = environment
            return ExtensionInstallPlan(extension_id=extension_id, environment_id=environment_id,
                                        runtime_kind=kind, action='workspace', requires_network=True)
        environment_id = f'pixi-{location}-{environment}-{identity}'
        ready = (self.root / environment_id / '.ready').is_file()
        return ExtensionInstallPlan(extension_id=extension_id, environment_id=environment_id,
                                    runtime_kind=kind, action='reuse' if ready else 'create',
                                    requires_network=not ready)

    def workspace(self, plan: ExtensionInstallPlan) -> Path:
        target = self.target_for_plan(plan)
        if target is not None:
            return target.manager.workspace(target.plan)
        if plan.runtime_kind == 'shared' and self.official is not self:
            return self.official.workspace(plan)
        if (plan.environment_id or '').startswith('bundled-base-'):
            return self.official._source_root
        if (plan.environment_id or '').startswith('workspace-'):
            environment = self._workspace_environments[plan.environment_id or '']
            return self._workspace_root(environment)
        if plan.environment_id is None:
            raise ValueError('Native extensions have no Python workspace')
        return self._managed_workspace(plan.environment_id)

    def _managed_workspace(self, environment_id: str) -> Path:
        workspace = (self.root / environment_id).resolve()
        if workspace.parent != self.root.resolve():
            raise ValueError('Invalid managed runtime path')
        return workspace

    def ensure(self, manifest: ExtensionManifest, operation: InstallOperation) -> ExtensionInstallPlan:
        manager = self.for_environment(manifest.runtime.environment or 'studio-runtime')
        if manager is not self:
            return manager.ensure(manifest, operation)
        plan = self.plan(manifest)
        if plan.action in {'none', 'reuse', 'bundled', 'shared'}:
            return plan
        operation.check_cancelled()
        pixi = self._pixi(operation)
        workspace = self.workspace(plan)
        if plan.action == 'create':
            workspace.mkdir(parents=True, exist_ok=True)
            if self.dependency_root != self._source_root:
                assert manifest.runtime.environment is not None
                materialize_locked_environment(self._source_root, manifest.runtime.environment, workspace, self.dependency_root)
            else:
                for name in ('pixi.toml', 'pixi.lock'):
                    shutil.copy2(self._source_root / name, workspace / name)
                for source in sorted((self._source_root / 'wheels').glob('*.whl')):
                    operation.check_cancelled()
                    destination = workspace / 'wheels' / source.name
                    destination.parent.mkdir(exist_ok=True)
                    if not destination.exists():
                        shutil.copy2(source, destination)
        environment = manifest.runtime.environment
        assert environment is not None
        operation.report(f'Installing locked {environment} runtime')
        operation.run([str(pixi), 'install', '--locked', '-e', environment,
                       '--manifest-path', str(workspace / 'pixi.toml')], cwd=workspace, env=self.install_environment())
        operation.check_cancelled()
        if plan.action == 'create':
            (workspace / '.ready').write_text(json.dumps({'environment': environment}) + '\n', encoding='utf-8')
        return plan

    def launch(self, plan: ExtensionInstallPlan, environment: str) -> tuple[str, list[str]]:
        target = self.target_for_plan(plan)
        if target is not None:
            return target.manager.launch(target.plan, target.environment)
        if plan.runtime_kind == 'shared' and self.official is not self:
            return self.official.launch(plan, environment)
        workspace = self.workspace(plan)
        executable = self._find_pixi()
        if executable is None:
            raise FileNotFoundError('Pixi is missing for the installed extension runtime')
        return str(executable), ['run', '--frozen', '--no-install', '--manifest-path',
                                 str(workspace / 'pixi.toml'), '-e', environment]

    def python_launch(self, plan: ExtensionInstallPlan, environment: str) -> tuple[str, list[str]]:
        target = self.target_for_plan(plan)
        if target is not None:
            return target.manager.python_launch(target.plan, target.environment)
        if (plan.environment_id or '').startswith('bundled-base-'):
            return str(self.official._python(self.official._source_root / 'env')), ['-I']
        command, args = self.launch(plan, environment)
        return command, [*args, 'python', '-I']

    def target_for_plan(self, plan: ExtensionInstallPlan) -> SharedRuntimeTarget | None:
        if plan.runtime_kind != 'shared':
            owner = self._manager_owning_environment(plan.environment_id)
            if owner is not self:
                name = next(name for name, manager in self._runtime_managers.items() if manager is owner)
                return SharedRuntimeTarget(owner, plan, name)
            return next((target for target in self.official.shared_targets.values()
                         if target.plan.environment_id == plan.environment_id and target.manager is not self), None)
        return next((target for target in self.official.shared_targets.values()
                     if target.plan.environment_id == plan.environment_id), None)

    def _manager_owning_environment(self, environment_id: str | None) -> EnvironmentManager:
        if environment_id is None:
            return self
        for name, manager in self._runtime_managers.items():
            if (environment_id in manager._workspace_environments
                    or (environment_id.startswith(f'{name}-') and environment_id == f'{name}-{manager.identity(name)}')):
                return manager
        return self

    def ready(self, environment_id: str | None) -> bool:
        owner = self._manager_owning_environment(environment_id)
        if owner is not self:
            return owner.ready(environment_id)
        if environment_id is None:
            return False
        target = next((target for target in self.official.shared_targets.values()
                       if target.plan.environment_id == environment_id and target.manager is not self), None)
        if target is not None:
            return target.manager.ready(environment_id)
        if environment_id.startswith('bundled-base-'):
            return (environment_id == self.official._bundled_plan('').environment_id
                    and self.official._python(self.official._source_root / 'env').is_file())
        if environment_id.startswith('workspace-'):
            name = self._workspace_environments.get(environment_id)
            if name is None:
                return False
            return (environment_id == self._pixi_plan('', 'workspace', name).environment_id
                    and self._python(self._workspace_root(name) / '.pixi/envs' / name).is_file())
        return (self._managed_workspace(environment_id) / '.ready').is_file()

    def remove_unused(self, environment_id: str | None, referenced: set[str]) -> None:
        if environment_id is None or environment_id in referenced:
            return
        if environment_id.startswith(('workspace-', 'bundled-base-')):
            return
        target = self._managed_workspace(environment_id)
        if target.is_dir():
            shutil.rmtree(target)

    @staticmethod
    def _find_pixi() -> Path | None:
        found = shutil.which('pixi')
        pixi_home = Path(os.environ.get('PIXI_HOME', str(Path.home() / '.pixi'))).expanduser()
        candidate = Path(found) if found else pixi_home / 'bin' / ('pixi.exe' if os.name == 'nt' else 'pixi')
        return candidate.resolve() if candidate.is_file() else None

    def _pixi(self, operation: InstallOperation) -> Path:
        installed = self._find_pixi()
        if installed is not None:
            return installed
        operation.report('Installing Pixi from the official installer')
        self.root.mkdir(parents=True, exist_ok=True)
        if os.name == 'nt':
            operation.run(['powershell.exe', '-NoProfile', '-Command',
                           '$env:PIXI_NO_PATH_UPDATE="1"; $env:PIXI_VERSION="v0.81.0"; '
                           'iwr -useb https://pixi.sh/install.ps1 | iex'], cwd=self.root, timeout=120)
        else:
            installer = self.root / 'pixi-install.sh'
            with urllib.request.urlopen('https://pixi.sh/install.sh', timeout=30) as response:
                installer.write_bytes(response.read())
            operation.check_cancelled()
            installer_env = dict(os.environ)
            installer_env['PIXI_VERSION'] = 'v0.81.0'
            installer_env['PIXI_NO_PATH_UPDATE'] = '1'
            operation.run(['bash', str(installer)], cwd=self.root, timeout=120, env=installer_env)
            installer.unlink()
        installed = self._find_pixi()
        if installed is None:
            raise FileNotFoundError('Official Pixi installer did not create the Pixi executable')
        return installed
