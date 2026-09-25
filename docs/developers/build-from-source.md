# Build from Source

Clone the repository with its Unity exporter submodule and use Pixi for every managed Python or Node command.

```bash
git clone --recurse-submodules <your-repo-url>
cd f8studio
pixi install -e web-studio-test
pixi run -e web-studio npm --prefix packages/f8studio_web ci
```

For an existing checkout:

```bash
git submodule update --init --recursive
pixi lock
```

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

## Distribution

Build the launcher, native runtime, non-editable Python wheels, and embedded Web bundle:

```bash
pixi run -e ci dist_ci
pixi run -e ci dist_ci --archive
```

Output is written under `build/dist/f8studio-<platform-tag>`. The generated install script creates only release runtime environments and installs repository wheels with `--no-index --no-deps`. The `f8studio-server` wheel contains the production Web bundle, so it does not depend on a source checkout at runtime.

Build the launcher alone with:

```bash
pixi run -e launcher build_studio_launcher
```

Windows and Linux release verification must run on their respective operating systems. Linux mocks do not satisfy the Windows gate.

## Unity Exporter

```bash
pixi run -e web-studio unitymods_validate
pixi run -e web-studio unitymods_contract
pixi run -e web-studio unitymods_build
pixi run -e web-studio unitymods_test
pixi run -e web-studio unitymods_package
```

Web Studio owns the interactive detect, preview, and confirmed-install flow. The submodule exposes a typed headless setup core and C# exporter artifacts.
