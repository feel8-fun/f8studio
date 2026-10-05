# Development workspace tasks

The root Pixi workspace owns development integration. Extensions own service
entrypoints, builds, releases, and runtime locks. Select an environment explicitly
with `pixi run --locked -e <environment> <task>`.

## Studio and tray debugging

```bash
pixi run --locked -e web-studio studio_web_install
pixi run --locked -e web-studio studio_web_build
pixi run --locked -e web-studio studio_tray
```

`studio_server` runs in the current terminal; `studio_launch` also opens the
browser; `studio_tray` runs the desktop supervisor with its console/log menu.
Pass server options directly, for example `studio_tray --port 8230 --no-browser`.
Linux tray mode uses the Pixi GTK backend and needs a desktop session. When no
tray backend is available, Studio logs the cause and runs in console mode.

These startup tasks regenerate the development catalog and explicitly select
`build/workspace/config/service-index.json`. They do not compile native services
or install every extension. A missing or invalid extension registration appears
as a failed extension with a reinstall action instead of blocking the server.
Stopped or disabled extensions keep their user settings and models.

`studio_cli` and `studio_mcp` connect to a running server. `studio_web_dev` starts
Vite; run it beside `studio_server` after installing frontend dependencies.

## Prepare and verify the workspace

| Task | Environment | Purpose |
| --- | --- | --- |
| `workspace_prepare` | `build-check` | Prepare library inputs and generated registrations. |
| `workspace_catalog` | `build-check` or `web-studio` | Regenerate registrations without changing built binaries. |
| `workspace_python_describes` | `build-check` | Refresh Python service descriptions using the integration interpreter. |
| `cpp_bootstrap` | `cpp` | Prepare locked Conan dependencies inside the Pixi toolchain. |
| `cpp_configure_release` | `cpp` | Configure the integration CMake build after bootstrap. |
| `cpp_build_release` | `cpp` | Build and deploy native services after configuration. |
| `workspace_native_describes` | `build-check` | Refresh native descriptions after building their executables. |
| `install_services` | `build-check` | Prepare extension-owned environments and missing descriptions. |
| `update_describes` | `build-check` | Refresh descriptions through extension-owned environments. |
| `pytest`, `typecheck`, `lint`, `lint_imports` | `build-check` | Run integration tests and static checks. |
| `extensions_check`, `contracts_check`, `quality_exceptions` | `build-check` | Validate ownership, Studio contracts, and exception policies. |
| `pytest_sdk` | `build-check` | Delegate SDK tests to the SDK workspace. |
| `studio_web_ci` | `build-check` | Install locked frontend dependencies, typecheck, and test. |
| `studio_server_test` | `web-studio-test` | Generate the catalog and run server tests. |
| `studio_web_e2e`, `studio_graph_bench` | `web-studio-test` | Build the frontend and start a test server on a separate port. |
| `workspace_snapshot` | `ci` or `build-check` | Build a local integration snapshot through `build-check`. |
| `workspace_clean` | `build-check` | Remove disposable outputs; use `--dry-run` to preview. |
| `doc_gen`, `doc_check`, `doc_build` | `doc` | Generate service documentation, validate it, and build the site. |

Python description generation is a development check. It does not prepare the
extension's runtime environment. Use Extensions to install an extension before
launching its services. For a full clean rebuild, follow
[Build from Source](build-from-source.md).

Tasks ending in `_probe` that take a server URL operate on an already running
server. `studio_runtime_probe` defaults to port 8210 and requires an installed,
enabled PyEngine; pass `--base-url` for another server. Media and stream probes
also require their selected input sources. Benchmarks require the relevant
browser, media, or hardware prerequisites.

## Independent extension tasks

Run service entrypoints through the owning manifest, for example:

```bash
pixi run --locked --manifest-path extensions/f8pyengine/pixi.toml -e pyengine f8pyengine
pixi run --locked --manifest-path extensions/f8pyengine/pixi.toml -e pyengine f8pyexpr
pixi run --locked --manifest-path extensions/f8pyaudiofeat/pixi.toml -e audiofeat f8pyaudiofeat_core
pixi run --locked --manifest-path extensions/f8pydl/pixi.toml -e dl f8pydl_detector
```

Prepare the extension's SDK checkout as specified in its development guide.
The root integration environment retains source dependencies for tests; it is
not a replacement for each service's runtime environment.

Unity exporter builds and packaging belong to `extensions/f8unitymods` and
require Windows. The root retains `unitymods_validate` and `unitymods_contract`
for integration validation. Official distribution assembly belongs to the
separate `f8distribution` repository.
