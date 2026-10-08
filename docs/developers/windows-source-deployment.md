# Windows source deployment

Use PowerShell from the repository root. This source workspace contains 17 Git
submodules; their revisions are pinned by the parent repository. Windows native
builds require Visual Studio 2022 with the C++ desktop workload and Windows SDK.
Pixi 0.81.0 is the version used by the workspace and extension CI.

## Update the workspace

Check for local changes in the parent and submodules before updating. Keep local
changes intact; do not reset them to make an update succeed.

```powershell
git status --short --branch
git submodule foreach --recursive git status --short
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
pixi self-update --version 0.81.0 --no-release-note
pixi install --locked -e workspace -e build-check -e cpp
pixi run --locked -e workspace workspace_prepare
```

The workspace tool synchronizes SDK inputs for Platform and all Python extensions,
including optional Diagnostics. Clean nested SDK checkouts advance from the local
root SDK; no GitHub push or manual checkout loop is required. Uncommitted root SDK
edits are mirrored and tracked by the tool. Local extension SDK edits, staged work
and independent commits are preserved and reported before synchronization.

```powershell
pixi run --locked workspace_runtime_prepare
pixi run --locked workspace_runtime_check
```

Preparation installs each declared environment with its own lock, compares the
actually imported SDK with workspace source, rebuilds stale noneditable SDK
installations, and verifies real service entrypoints with `--frozen --no-install`.
For a selected extension:

```powershell
pixi run --locked workspace_runtime_prepare --workspace extensions/f8pyengine
pixi run --locked workspace_runtime_check --workspace extensions/f8pyengine
pixi run --locked workspace_runtime_test
```

The runtime regression uses a disposable real PyEngine environment and covers
source-only SDK edits and deleted modules without changing package versions.
Source-only `workspace_python_describes` checks cannot replace deployment validation.
Platform/Studio startup refreshes already installed environments; full preparation
also installs missing optional environments. Stop and management tasks do not run
runtime preparation.

Keep independent extension CI SDK pins and locks updated together when SDK
package metadata changes. A locked installation failure is actionable; do not
regenerate all dependencies merely to bypass a mismatch. Official extension
artifacts remain independent of the workspace's source mirrors.

The DL workspace supplies CUDA 12 and pins ONNX Runtime GPU to `1.24.x`.
ONNX Runtime `1.30` wheels require CUDA 13 DLLs and cannot use this environment's
CUDA 12 libraries. Verify an actual CUDA model session after changing either
dependency, rather than checking only the compiled provider list.

## Build and register services

Run these commands sequentially. Catalog generation rewrites generated service
registrations; concurrent catalog tasks can contend for files on Windows.

```powershell
pixi run --locked -e build-check studio_web_install
pixi run --locked -e build-check studio_web_build
pixi run --locked -e cpp cpp_bootstrap
pixi run --locked -e cpp cpp_configure_release
pixi run --locked -e cpp cpp_build_release
pixi run --locked -e build-check workspace_python_describes
pixi run --locked -e build-check workspace_native_describes
```

Conan uses `build/cache/conan/<pixi-lock-id>/`. The first build compiles missing
dependencies, including OpenSSL and OpenCV. Subsequent builds reuse that cache.
Native executables and DLLs deploy into `build/workspace/runtime/bundles/`.

To copy retained model files from the old workspace layout, preserving originals:

```powershell
pixi run --locked -e build-check python scripts/install_services.py --index build/workspace/config/service-index.json --python-only --no-install --migrate-resources services
```

The migration verifies SHA-256 and refuses differing destination files. Models
and user configuration remain outside disposable build output.

## Start and verify

```powershell
pixi run --locked platform_ensure
pixi run --locked platform_cli source start webstudio
pixi run --locked platform_cli jobs list
pixi run --locked platform_open
```

Starting WebStudio through Platform also starts Media Gateway. WebStudio is at
`http://127.0.0.1:8210`; its loopback landing page establishes browser access.
Media Gateway listens at `http://127.0.0.1:8211`. Platform's management portal
uses the port recorded in `build/workspace/platform-connection.json`.

If Platform first started before the native build finished, those extensions
remain uninstalled in its saved state. After building and generating descriptions,
register each with `pixi run --locked platform_cli extensions install <extension>`
for `cppengine`, `cvkit`, `implayer`, `screencap`, and `audiocap`. Wait for each
returned job to succeed using `platform_cli jobs show <job-id>`. The Studio catalog
then includes the native services without restarting it.

Verify after the startup job succeeds:

```powershell
pixi run --locked -e build-check pytest -q
pixi run --locked -e build-check studio_web_typecheck
pixi run --locked -e build-check studio_web_test
pixi run --locked -e build-check typecheck
$connection = Get-Content build/workspace/platform-connection.json | ConvertFrom-Json
$env:F8STUDIO_DATA_DIR = Join-Path (Split-Path $connection.token_file) 'studio'
pixi run --locked -e build-check studio_runtime_probe
```

For the desktop tray, first stop the headless platform with `platform_stop`, then
run `pixi run --locked platform_tray`. Stopping Platform also stops its managed
applications. A platform-owned WebStudio can be stopped independently with
`pixi run --locked platform_cli source stop webstudio`.
