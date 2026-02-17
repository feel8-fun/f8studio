from __future__ import annotations

import sys
import types
import unittest

from _bootstrap import ensure_package_importable

ensure_package_importable()

from f8pystudio.extensions import ExtensionRegistry, StudioPluginManifest


class ExtensionRegistryTests(unittest.TestCase):
    def test_register_manifest(self) -> None:
        registry = ExtensionRegistry()
        manifest = StudioPluginManifest(
            plugin_id="demo.plugin",
            plugin_name="Demo",
            plugin_version="1.0.0",
        )
        registry.register_manifest(manifest)
        self.assertEqual(len(registry.manifests()), 1)
        self.assertEqual(registry.manifests()[0].plugin_id, "demo.plugin")

    def test_register_module(self) -> None:
        module_name = "test_plugin_module_manifest"
        fake_module = types.ModuleType(module_name)
        fake_module.PLUGIN_MANIFEST = StudioPluginManifest(
            plugin_id="demo.module",
            plugin_name="Demo Module",
            plugin_version="1.0.0",
        )
        sys.modules[module_name] = fake_module
        try:
            registry = ExtensionRegistry()
            loaded = registry.register_module(module_name)
            self.assertEqual(loaded.plugin_id, "demo.module")
        finally:
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
