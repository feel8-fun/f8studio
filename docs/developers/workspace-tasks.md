# Development workspace tasks

The root Pixi workspace owns development integration. Extensions own service
entrypoints, builds, releases, and runtime locks. Select an environment explicitly
with `pixi run --locked -e <environment> <task>`.

## Studio and tray debugging

```bash
pixi run --locked platform_tray
```

Platform tasks select the small `workspace` orchestration environment automatically
and delegate execution to `platform/pixi.toml`: `platform-runtime` for headless
operations and `platform-desktop` for the tray. WebStudio is not installed into
either Launcher environment.

The tray belongs to Launcher and opens the independent management portal. Select
Start source on WebStudio to run the editor and its declared application dependencies.
Use `platform_dev` for a foreground platform, `platform_open` for its portal, and
`platform_stop` to stop it. These entrypoints share one development platform.

`studio_server` remains a direct source debugging entrypoint. It ensures a separate
platform is available and registers Studio as externally managed. Stop that source
process in its terminal. `studio_launch` also opens the browser. Server options can
be passed directly, for example `studio_server --port 8230 --no-browser`.
`studio_cli` and `studio_mcp` connect to a running Studio server.

See [Platform management](platform-management.md) for ownership, data paths and
headless CLI operations.

## Prepare and verify the workspace

| Task | Environment | Purpose |
| --- | --- | --- |
| `workspace_prepare` | `build-check` | Prepare library inputs and generated registrations. |
| `workspace_catalog` | `workspace`, `build-check` or `web-studio` | Regenerate registrations without changing built binaries. |
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
