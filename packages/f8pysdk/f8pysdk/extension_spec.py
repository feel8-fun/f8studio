"""Publisher-facing extension metadata, independent of the Studio server."""
from __future__ import annotations

from typing import Literal

import msgspec

RuntimeKind = Literal['native', 'bundled', 'workspace', 'pixi', 'shared']


class ExtensionRuntime(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    kind: RuntimeKind = 'native'
    environment: str | None = None
    requires_python: str | None = None
    dependencies: tuple[str, ...] = ()


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
