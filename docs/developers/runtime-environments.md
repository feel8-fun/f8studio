# Runtime environments

Studio manages extension installation separately from Python runtime definitions. Services and tools belonging to a shared Python extension use its selected runtime; skills and resources are capabilities of the extension, not packages installed into that interpreter.

## Workspace ownership

The distribution bootstrap owns only `f8platform`. Its source workspace lives
at `launcher/pixi.toml`; it contains no Studio or Media Gateway modules.
WebStudio owns `extensions/f8webstudio/pixi.toml` and Media Gateway owns
`extensions/f8mediagateway/pixi.toml`. Each carries its own runtime lock and publisher.
WebStudio frontend/backend are always one release unit. The previous
`config/studio-runtime/` aggregate workspace has been removed.

Each Python extension owns its workspace and lock in its own repository.
`config/runtime-environments.json` references these source workspaces for
integrated development. Installed extension artifacts carry their own runtime
inputs. The previous top-level `runtimes/` profiles and feature generator have
been removed. The root `pixi.toml` is a developer/CI workspace.

Environment names and counts are chosen by the extension author. The
extension's `runtime.environment` selects its default launch environment;
other environments can coexist in the same workspace. Identical names in
different extension workspaces are valid. Managed prefix identity includes
workspace ownership and locked inputs; shared caches reuse package files
without merging independent interpreters.

The Python Engine extension contains `f8.pyengine`, `f8.pyexpr`, and
`f8.pyscript`, all defaulting to its `pyengine` environment.

For example, prepare a source extension or Studio itself with:

```sh
pixi install --locked --manifest-path extensions/f8pydl/pixi.toml -e dl
pixi install --locked --manifest-path launcher/pixi.toml -e platform-runtime
pixi install --locked --manifest-path extensions/f8webstudio/pixi.toml -e webstudio
pixi install --locked --manifest-path extensions/f8mediagateway/pixi.toml -e media
```

## Inspect and prepare

The **Runtime Environments** page (`?view=environments`) shows environments with a name, source, revision, status, and the extensions/services/tools referencing it. References include declared consumers, including extensions that have not yet been installed. Extension installation, enabling, and capability details are managed separately on **Extensions** (`?view=extensions`).

- **Declared**: a developer definition has been saved; no dependency resolution or installation has run.
- **Missing**: the expected interpreter/installation is absent.
- **Ready**: the runtime installation exists for its recorded definition. This does not replace dependency checking when a shared extension is installed.
- **Definition changed**: workspace installation records no longer match the current environment definition. Stop its services/tools, then select **Verify and update** to install from the current lock and reconcile extension records.
- **Preparing**: Pixi is resolving/installing. Progress is displayed and preparation can be cancelled.
- **Failed**: preparation failed; the error and installer log location are reported. The base environment is unchanged.

Environment identity uses the selected environment's features, locked package records and referenced wheel artifacts. Unrelated feature or lock changes do not invalidate it. Development workspaces remain explicit workspace environments; they are not copied into Studio's runtime storage when verified.

## Developer revisions

Use **Create developer environment** to create an independent environment or derive a revision from an official/package/prepared developer environment. Enter one Conda or PyPI requirement per line. A definition is saved without downloading dependencies. Select **Prepare** to resolve and install it.

For a derived revision choose:

- **Preserve base package versions**: add base Conda version/build pins and PyPI version pins. Additional incompatible requirements fail resolution. Local Python projects are copied into the snapshot and installed non-editably.
- **Allow changes to base dependencies**: explicit additional requirements replace the corresponding base declarations in the independent snapshot. Other requirements remain inherited; the solver must still satisfy them and transitive constraints.

Revisions do not share an official `solve-group`. Their definitions and base lock metadata are copied when created; editing the original definition does not edit the snapshot. Prepared details show dependency changes against that saved base lock. A bundled interpreter without a reproducible Pixi definition is not offered as a base.

After resolution, identical full lock results and identical copied local source content reuse one managed installation. Different definitions can reference that installation. Different dependency versions remain isolated. Native libraries, GPU compatibility, system requirements, and platform constraints still apply to each runtime.

## Extensions that reuse a runtime

A published extension can continue to provide its own locked Pixi environment. That environment appears as a package runtime and can be explicitly selected by other shared Python extensions.

An extension that is designed to reuse an existing interpreter declares `shared`:

```json
{
  "runtime": {
    "kind": "shared",
    "environment": "studio-runtime",
    "requiresPython": ">=3.12,<3.15",
    "dependencies": ["requests>=2.32,<3"]
  }
}
```

Its Python code/wheel contents live in the extension's `python/` directory. Services launch `python -m module`; tools declare `command: "python"` and `args: ["-m", "module"]`. This lets Studio launch the code using the chosen interpreter without installing the extension over existing runtime packages.

Open the extension detail page and choose a runtime before installing. Uninstall an installed extension before changing its selection. The selection persists across Studio restarts. Installation verifies Python and dependency constraints and prevents extension modules/distributions from shadowing runtime or standard-library modules. Studio does not add missing dependencies to a shared interpreter: create/prepare a compatible revision, or publish an independent Pixi runtime.

Native executables and extensions whose launchers depend on their own Pixi tasks keep their publisher-defined launchers. They do not expose the shared-runtime selector; convert their launch contract to `shared` when they are intended to use an external interpreter.

## Storage and retention

**Runtime storage and shared cache** selects an absolute storage directory before managed installations are prepared. The managed `runtimes/` and `package-cache/` directories are siblings on the same volume. Installer commands use that Pixi/uv cache, allowing supported hardlink/file reuse. Moving the storage setting does not move existing files; prepared managed runtimes must be removed first.

Windows NTFS supports hardlinks on the same volume. Cross-volume caches cannot provide hardlinks. Copy-required package files and different package builds can still consume additional space. Development workspaces remain in their checkout, which may be on a different volume from the managed cache. Models and extension resources remain separately managed.

Environment details report logical file bytes, unique file data bytes deduplicated by file identity, bytes belonging to files with multiple hardlinks, and bytes belonging to files with a single hardlink. These are not physical disk allocation or guaranteed reclaimable space; copy-on-write savings are not measured.

**Keep revision** protects a developer revision from removal. **Remove unused** removes its definition and, when no other revision uses it, its managed installation. Extension runtime selections and child revision base references block removal. Reset a selection or remove dependent revisions first. Removing a revision retains the shared package cache and does not remove models or resources.

## Publisher releases

Runtime providers can publish immutable baselines with `providerId`, `version`, and `abi`. The environment inspector displays these values. Shared extensions can constrain provider versions and ABI, while installation still checks actual interpreter/dependency/wheel compatibility. See [Extension Releases](extension-releases.md) for publisher commands, pinned Studio assembly, update policy, and common feature generation.
