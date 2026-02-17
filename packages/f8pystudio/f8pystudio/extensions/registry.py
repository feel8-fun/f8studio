from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from .plugin_manifest import StudioPluginManifest


def load_manifest_from_module(module_name: str) -> StudioPluginManifest:
    mod = importlib.import_module(str(module_name))
    try:
        manifest = mod.PLUGIN_MANIFEST
    except AttributeError:
        raise ValueError(f"module {module_name!r} does not expose PLUGIN_MANIFEST")
    
    if not isinstance(manifest, StudioPluginManifest):
        raise TypeError(f"module {module_name!r} PLUGIN_MANIFEST must be StudioPluginManifest")
    return manifest


@dataclass
class ExtensionRegistry:
    _manifests: dict[str, StudioPluginManifest] = field(default_factory=dict)

    def register_manifest(self, manifest: StudioPluginManifest, *, overwrite: bool = False) -> None:
        plugin_id = str(manifest.plugin_id).strip()
        if not plugin_id:
            raise ValueError("plugin_id is empty")
        if plugin_id in self._manifests and not overwrite:
            raise ValueError(f"plugin already registered: {plugin_id}")
        self._manifests[plugin_id] = manifest

    def register_module(self, module_name: str, *, overwrite: bool = False) -> StudioPluginManifest:
        manifest = load_manifest_from_module(module_name)
        self.register_manifest(manifest, overwrite=overwrite)
        return manifest

    def manifests(self) -> list[StudioPluginManifest]:
        return list(self._manifests.values())
