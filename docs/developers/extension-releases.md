# Independently published extensions and runtimes

Extension repositories own their build dependencies, tests, native deployment or Python wheels, and extension ZIPs. Each extension publisher owns its workspace and locks; The platform owns the bootstrap runtime; application extensions own independent runtimes. WebStudio publishes its frontend and backend together. The separate f8distribution repository assembles verified releases from published artifacts. Updating an extension does not require compiling other extensions or changing Studio's source.

## Runtime ownership and compatibility

Build dependencies stay in each publisher's build environment. Released runtime
inputs contain locked dependencies and built wheels, with no editable projects or
installation-time builds. The Launcher bootstrap has its own runtime; WebStudio,
Media Gateway and other extensions use their own independent environments.

Environment names are local to a package. Compatible package files can be reused
through F8's Pixi cache without sharing a mutable interpreter. Package updates
therefore change only that extension's environment and do not upgrade another
extension's dependencies. Communication compatibility is expressed through the
application/service protocol contracts and verified in integration tests.

Runtime catalogs identify contained Pixi workspaces and selected environments.
Provider/version/ABI metadata can document a published runtime's provenance; the
management GUI/API does not expose a runtime selector or derived-environment editor.

## Publishing artifacts

Build and test a Python extension's wheel in its own repository. The SDK builder packages that wheel and checks real service descriptions:

```sh
pixi run python -m f8pysdk.extension_packaging --source . --wheel dist/example-1.0-py3-none-any.whl --output dist/example-1.0.zip
```

For Python source extensions declaring `workspace`, the builder converts local inputs to wheels and generates an independent locked runtime at publication time. It carries all declared environments and verifies that the default environment exists. This publisher step may build local dependency wheels and run the solver; Studio installation never does. Publishers can instead supply an already prepared `--runtime-root` to control the released dependencies precisely. Each extension retains its own runtime; users do not choose another extension's interpreter.

For native extensions, supply their independently deployed executable directory with `--runtime-root`. Publisher ZIPs contain `config/artifact.json` with extension ID, version, platform and wheel tags, alongside the capability catalog. Studio validates descriptions again when installing. Extensions with their own interpreter declare `kind: "pixi"` and an environment ID. Pass `--wheel` together with `--runtime-root`, pointing at an unpacked published runtime package. The wheel must appear with identical bytes in that package's lock. The builder embeds its runtime definition and checks the lock; services may declare the same explicit `python -m module` entrypoint through its own interpreter. Tools use their declared module through the private interpreter. Other extensions retain their own locked workspaces and may reuse cached package files.

A runtime publisher first builds or downloads all required wheels and locks a standard Pixi workspace against them. The publishing command snapshots those inputs, checks that they are portable wheels, and verifies the existing lock with `pixi lock --check`. It does not run a dependency solver or compile packages:

```sh
# Run in the independent f8distribution repository
pixi run --locked assemble releases/<release>.json
```

A `studio-runtime` provider must include the Studio server wheel with embedded built Web assets, SDK/core/media wheels and application runtime dependencies. This remains an application release responsibility. Other runtime providers do not need Studio or its source packages.

Every publisher emits a `.zip.sha256` sidecar. Publish ZIP and checksum through the repository's release process. Each new runtime or extension release gets a new version; reusing an existing version for different content is a publisher error. The release lock also pins the checksum so changed bytes are rejected.

## Studio release assembly

The release lock pins artifact IDs, versions, platform and SHA-256. `location` accepts HTTPS or a local path relative to the lock file; local files support offline CI and development. HTTPS redirects must remain HTTPS.

```json
{
  "schemaVersion": "f8platformRelease/1",
  "platform": "linux-x86_64",
  "baseRuntime": "platform-runtime",
  "startup": [],
  "artifacts": [
    {
      "artifactId": "platform-runtime", "version": "1.0.0", "kind": "runtime",
      "location": "artifacts/platform-runtime-1.0.0.zip",
      "sha256": "REPLACE_WITH_THE_PUBLISHED_64_CHARACTER_SHA256"
    },
    {
      "artifactId": "diagnostics", "version": "1.0.0", "kind": "extension",
      "location": "artifacts/diagnostics-1.0.0.zip",
      "sha256": "REPLACE_WITH_THE_PUBLISHED_64_CHARACTER_SHA256"
    }
  ]
}
```

Assemble with the small, separately locked tooling environment. It installs only Python and metadata libraries, and does not install/build extension projects:

```sh
# In the separate f8distribution repository
pixi run --locked assemble releases/<release>.json
```

The assembler verifies digests, safely extracts archives, checks identities/platforms and ownership, loads all extension declarations, and verifies the providers' existing locks. The base launcher references exactly the base provider's locked inputs. `pixi-pack` produces the offline application runtime from that lock. No CMake, extension wheel build, description regeneration, frontend build or dependency solve runs in this path.

Artifacts keep separate content-addressed directories. `config/extension-packages.json` registers extension payloads; they appear in Extensions as available packages. Users install/enable them through the normal manager. `config/runtime-environments.json` registers runtime providers independently of extension installation. `config/release-lock.json` records release provenance. Models, user configuration and managed prefixes remain outside immutable artifacts.

The `assemble-release` GitHub workflow checks out Studio and SDK contracts without extension submodules, assembles the specified lock, and verifies the relocated offline application. It uploads an artifact; it does not publish a remote release. The existing source distribution workflow explicitly uses `--build-workspace` while extension publishers migrate. There is no automatic fallback from failed artifact assembly to source builds.

## Extension environment ownership

Each extension package owns its workspace, dependencies and lock. Environment
names and counts are publisher choices. `runtime.environment` selects the
default launch environment and must exist in the package's runtime catalog.
Additional environments may share a workspace or use separate contained
workspaces. Names are local to an extension package; they need not equal its
extension ID. All referenced release inputs must still be contained, locked
and portable.

Independent prefixes share the F8 package cache, rather than a mutable
interpreter. Extensions carry their own locked runtime inputs; the GUI and API offer no
interpreter sharing or runtime rebinding.
WebStudio has its own application runtime at `extensions/f8webstudio/pixi.toml`; the platform bootstrap lives in `platform/pixi.toml`.

The `pyengine` extension contains `f8.pyengine`, `f8.pyexpr`, and `f8.pyscript`.
Its source workspace and lock are owned by `extensions/f8pyengine`; both Python
namespaces are included in the single `f8pyengine` wheel.

## Updates and rollback

Updating one extension changes its artifact version/checksum. Its locked runtime
is prepared independently; other extension environments remain unchanged. Retain
prior release locks and artifacts for reproducible rollback. Application updates
validate selected protocol consumers and restore the previous version if readiness
fails. Stop services/tools before uninstalling an extension; there is no automatic
replacement of a running extension.

Uninstall/import/install selects a service or tool extension version. Same-version
content changes are rejected. Old payloads and cached package files can support
rollback; unused managed environment files and package caches can be released from
Runtime Environments. No environment selection or rebinding operation is exposed.

## Source development

The root Pixi workspace contains development and CI tools. Studio's application
workspace is `extensions/f8webstudio/pixi.toml`. Python extension workspaces and locks
are owned by their corresponding repositories under `extensions/`.
There is no central feature/profile generator or extension dependency baseline.
Update a package's own manifest and lock when its dependencies change.

The `workspace_snapshot` task is a local integration helper, not an official
publisher. Production assembly is owned by the separate `f8distribution` repository.

## Application extensions and launcher

Applications use the ordinary extension catalog with an `application` capability.
The release lock uses `f8platformRelease/1` with runtime and extension artifacts.
`baseRuntime` selects the bootstrap; `startup` lists application extension IDs.
There is no separate component manifest or artifact kind.

An application declares launch module/distribution/environment, explicit args and
variables, named endpoints, provided/required protocols, and a readiness probe.
`${F8_PACKAGE_ROOT}`, `${F8_DATA_ROOT}`, `${F8_ENDPOINT:extension.endpoint}` and
`${F8_PORT:extension.endpoint}` are expanded without shell interpretation.
Endpoint overrides live in platform user data and change while affected processes
are stopped.

`platform/`, `extensions/f8webstudio/` and `extensions/f8mediagateway/` have independent
publisher workflows. WebStudio frontend/backend versions must match and its archive
must contain both. The SDK owns `f8media_protocol`, shared by Gateway and Studio.

The authenticated loopback platform API and CLI offer `import`, `prepare`, `select`,
`start`, `stop`, `configure`, `update` and `uninstall`. Updates prepare the candidate,
validate selected consumers and restore the old version if readiness fails.
User data and caches live outside immutable payloads.

The offline archive contains the bootstrap interpreter. Application environments
use their published locks and the shared package cache; a fully offline application
installation requires separately prepopulating that cache.
