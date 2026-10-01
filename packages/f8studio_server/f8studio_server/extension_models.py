from __future__ import annotations

from typing import Literal

import msgspec


RuntimeKind = Literal['native', 'bundled', 'workspace', 'pixi', 'shared']


class ExtensionRuntime(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    kind: RuntimeKind = 'native'
    environment: str | None = None
    requires_python: str | None = None
    dependencies: tuple[str, ...] = ()


class PresetEnvironmentStatus(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    environment: str
    ready: bool


class ExtensionManifest(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    extension_id: str
    name: str
    version: str
    description: str
    service_classes: tuple[str, ...]
    runtime: ExtensionRuntime = ExtensionRuntime()
    model_directories: tuple[str, ...] = ()


class ExtensionCatalog(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    schema_version: Literal['f8extensionCatalog/1']
    extensions: tuple[ExtensionManifest, ...]
    preinstalled: tuple[str, ...] = ()


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
