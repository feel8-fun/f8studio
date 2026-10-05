# Runtime environments

The Launcher installs and maintains the Pixi environments declared by extensions.
WebStudio and the standalone Platform portal show the same environment inventory.
Services and tools inside an extension use its declared launch environment. Skills
and resources are extension capabilities, not interpreter packages.

## Workspace ownership

The bootstrap runtime belongs to `platform/pixi.toml`. WebStudio owns
`extensions/f8webstudio/pixi.toml`, and Media Gateway owns
`extensions/f8mediagateway/pixi.toml`. Every extension publishes its own wheels or
binaries, portable manifests and locks. WebStudio frontend and backend are always
one release unit. The root `pixi.toml` is a development/CI workspace.

Environment names and counts are chosen by extension authors. `runtime.environment`
selects the default launch environment; other environments can coexist in the
same package. Identical names in separate extension workspaces are valid. Managed
prefix identities include workspace ownership and locked inputs. Shared caches
reuse package files while the interpreters remain independent.

The Python Engine extension contains `f8.pyengine`, `f8.pyexpr`, and `f8.pyscript`,
all defaulting to its `pyengine` environment. Installing that extension prepares its
runtime automatically. Changing dependencies is a publisher action: update the
extension's manifest and lock and publish a new version. There are no GUI/API
operations for creating or deriving an environment, pinning a developer revision,
or assigning another extension's interpreter.

For source development, use Pixi directly in the extension's own workspace:

```sh
pixi install --locked --manifest-path extensions/f8pydl/pixi.toml -e dl
pixi install --locked --manifest-path platform/pixi.toml -e platform-runtime
pixi install --locked --manifest-path extensions/f8webstudio/pixi.toml -e webstudio
pixi install --locked --manifest-path extensions/f8mediagateway/pixi.toml -e media
```

## Inspect and verify

The **Runtime Environments** page (`?view=environments`) shows names, revisions,
status and the owning extensions/services/tools. References include declared
extensions that are not installed yet. Installation and enablement belong to the
**Extensions** page (`?view=extensions`).

- **Missing**: the expected installation is absent.
- **Ready**: the recorded environment installation exists.
- **Definition changed**: stop the affected processes, then **Verify and update**
  to install from the current lock and reconcile extension records.
- **Preparing**: Pixi installs the locked environment; preparation can be cancelled.
- **Failed**: preparation failed; its error and installer log are available.

Select an environment to see its definition file, selected Pixi environment,
installation path and package versions. Installed prefixes are inspected through
Conda metadata and Python distribution metadata. When the prefix is absent, the
inspector displays packages from the lock, which may cover multiple platforms.

Identity uses selected features, locked packages and wheel artifacts. Unrelated
feature/lock changes do not invalidate that environment. Verifying a source
workspace does not copy its environment into managed runtime storage.

## Storage and cleanup

Environment files and F8's dedicated `package-cache/` are measured separately and
as one combined total. The combined total counts a hardlinked file's data once,
including links shared between an environment and cache. Allocated file blocks
include filesystem block rounding; Unix provides these values. Windows reports
file data sizes and hardlink relationships, with allocated blocks unavailable.
Copy-on-write extents are not measured, and removing a hardlink cannot reclaim data
still referenced elsewhere.

**Release unused environment files** removes recognized managed environment
directories with no installed-extension or retained-application references. It
keeps caches, development checkouts, unknown directories, models and resources.
Individual **Release files** actions also require that no installed extension uses
the environment. Pixi recreates released environments when they are needed.

The shared package cache is retained for Pixi reuse and deduplication. Cache deletion
is not exposed through the GUI, API or CLI.

Choose a storage directory before preparing managed installations. The `runtimes/`
and `package-cache/` directories remain siblings on the same volume. NTFS supports
hardlinks on the same volume; a cache on another volume cannot supply them.
Changing the storage setting does not move existing files, so managed runtime
files must be released first. Source workspaces keep their own environment paths.
