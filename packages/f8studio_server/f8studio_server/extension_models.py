from __future__ import annotations

from typing import Literal

import msgspec

from f8pysdk.extension_spec import (
    ExtensionCatalog as ExtensionCatalog,
    ExtensionManifest as ExtensionManifest,
    ExtensionRuntime as ExtensionRuntime,
    ExtensionTool,
    RuntimeKind as RuntimeKind,
)

from f8pysdk.specs import F8ServiceDescribe


class ExtensionServiceDetail(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    service_class: str
    describe: F8ServiceDescribe | None


class ExtensionSkillDetail(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    skill_id: str
    content: str


class ExtensionDetail(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    extension_id: str
    services: tuple[ExtensionServiceDetail, ...]
    tools: tuple[ExtensionTool, ...]
    skills: tuple[ExtensionSkillDetail, ...]


class PresetEnvironmentStatus(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    environment: str
    ready: bool


class ExtensionRecord(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    version: str
    installed: bool
    enabled: bool
    environment_id: str | None = None


class ExtensionStatus(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    extension_id: str
    name: str
    version: str
    description: str
    state: Literal['unavailable', 'available', 'installing', 'installed', 'disabled', 'failed']
    detail: str
    service_classes: tuple[str, ...]
    runtime_kind: RuntimeKind
    environment_id: str | None
    preinstalled: bool
    tool_ids: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] = ()
    runtime_environment: str | None = None
    runtime_selectable: bool = False


class ExtensionToggleRequest(msgspec.Struct, frozen=True, kw_only=True):
    enabled: bool


class ExtensionImportRequest(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    url: str
    sha256: str


class ExtensionInstallPlan(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    extension_id: str
    environment_id: str | None
    runtime_kind: RuntimeKind
    action: Literal['none', 'reuse', 'create', 'bundled', 'workspace', 'shared']
    requires_network: bool


class EnvironmentStatus(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    environment_id: str
    runtime_kind: RuntimeKind
    extension_ids: tuple[str, ...]
    ready: bool
    name: str = ''
    source: Literal['official', 'package', 'developer'] = 'official'
    revision: str = ''
    state: Literal['declared', 'preparing', 'ready', 'changed', 'missing', 'failed'] = 'declared'
    detail: str = ''
    service_classes: tuple[str, ...] = ()
    tool_ids: tuple[str, ...] = ()
    base_environment_id: str | None = None
    pinned: bool = False


class EnvironmentCreateRequest(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    name: str
    base_environment_id: str | None = None
    policy: Literal['preserve', 'adjust'] = 'preserve'
    python: str = '3.12.*'
    conda_dependencies: tuple[str, ...] = ()
    pypi_dependencies: tuple[str, ...] = ()


class EnvironmentRevision(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    environment_id: str
    request: EnvironmentCreateRequest
    base_revision: str | None = None
    pinned: bool = False
    resolved_id: str | None = None
    state: Literal['declared', 'preparing', 'ready', 'failed'] = 'declared'
    detail: str = ''


class EnvironmentUsage(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    logical_bytes: int = 0
    unique_file_bytes: int = 0
    shared_link_bytes: int = 0
    exclusive_file_bytes: int = 0


class EnvironmentDetail(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    environment_id: str
    name: str
    revision: str
    manifest: str
    base_environment_id: str | None
    policy: Literal['preserve', 'adjust'] | None
    conda_dependencies: tuple[str, ...]
    pypi_dependencies: tuple[str, ...]
    storage_path: str
    cache_path: str
    usage: EnvironmentUsage
    pinned: bool = False
    changed_packages: tuple[str, ...] = ()
    definition_path: str = ''
    source_environment: str = ''
    provider_id: str | None = None
    provider_version: str | None = None
    abi: str | None = None


class EnvironmentRetentionRequest(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    pinned: bool


class ExtensionRuntimeRequest(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    environment_id: str | None


class RuntimeStorageRequest(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    path: str


class RuntimeStorageStatus(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    path: str
    cache_path: str
    can_change: bool
