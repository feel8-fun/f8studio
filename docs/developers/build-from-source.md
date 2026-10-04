# Build from Source

Clone the repository with its Unity exporter submodule and use Pixi for every managed Python or Node command.

```bash
git clone --recurse-submodules <your-repo-url>
cd f8studio
pixi install -e web-studio-test
pixi run -e web-studio npm --prefix extensions/f8webstudio/f8studio_web ci
```

For an existing checkout:

```bash
git submodule update --init --recursive
pixi lock
```

Python extensions own their source workspaces. Before running an extension from
source, prepare the SDK checkout it declares (usually `.sdk`) as documented in
that repository’s `DEVELOPMENT.md`. Published extension artifacts contain wheels
and locked runtime inputs and do not need SDK source checkouts.

## Start Studio

Build the production Web assets and start the combined local server:

```bash
pixi run -e web-studio studio_web_build
pixi run -e web-studio studio_server
```

For frontend development, keep the server running and start Vite in another terminal:

```bash
pixi run -e web-studio studio_web_dev
```

The production server defaults to `127.0.0.1:8210`. Explicitly configure `--host`, `--allowed-host`, TURN, and authentication controls before exposing it beyond loopback.

## Test and Type Check

```bash
pixi run -e web-studio-test studio_core_test
pixi run -e web-studio-test studio_media_gateway_test
pixi run -e web-studio-test studio_server_test
pixi run -e web-studio-test studio_web_test
pixi run -e web-studio-test studio_web_e2e
pixi run -e web-studio-test studio_python_typecheck
pixi run -e web-studio-test studio_web_typecheck
pixi run -e web-studio-test studio_no_qt_check
pixi run pytest_sdk
```

The no-Qt check scans retained Python source, package manifests, installed distributions, loaded modules, and the active environment's dynamic libraries.

## Code Generation and Documentation

```bash
pixi run -e default protocol_codegen_all
pixi run -e default update_describes
pixi run -e doc doc_gen
pixi run -e doc doc_check
pixi run -e doc doc_build
```

Generated service descriptions and protocol bindings must be committed with their source schema changes.

## Native Services

```bash
pixi run -e cpp cpp_bootstrap
pixi run -e cpp cpp_configure_release
pixi run -e cpp cpp_build_release
pixi run -e cpp cpp_test_release
```

## Media and Graph Benchmarks

```bash
pixi run -e web-studio-test studio_media_bench
pixi run -e web-studio-test studio_p3_combined_bench
pixi run -e web-studio-test studio_graph_bench
```

## Independent publication and distribution

Prepare development library checkouts with `pixi run -e build-check python scripts/workspace_inputs.py prepare`.
Each repository builds and publishes its own artifact:

```bash
pixi run --locked --manifest-path launcher/.ci/pixi.toml publish
pixi run --locked --manifest-path extensions/f8mediagateway/.ci/pixi.toml publish
pixi run --locked --manifest-path extensions/f8webstudio/.ci/pixi.toml publish
```

WebStudio frontend and backend always have one version and one extension archive.
The shared media protocol belongs to the SDK, rather than a gateway source dependency.

Official releases are assembled in the separate `f8distribution` repository from
hash-pinned runtime and extension archives. That repository has no application
source checkout, compiler or frontend build. See [build and release](../development/build-and-release.md).
A workspace snapshot is only a local integration helper:

```bash
pixi run -e build-check workspace_snapshot --output build/workspace-snapshot
```
