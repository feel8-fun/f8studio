# Minimal Plugin Example

This folder shows the minimum structure for a Studio extension module.

## Files
1. `plugin.py`: exposes `PLUGIN_MANIFEST` as `StudioPluginManifest`.

## Usage
Import the module and register its manifest through `ExtensionRegistry`.

`PyStudioProgram` can auto-load extension modules from:

`F8PYSTUDIO_PLUGINS=your_pkg.plugin_module,another_pkg.plugin`
