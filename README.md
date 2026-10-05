# f8studio

Development workspace for Feel8 Studio. This repository combines independently
versioned SDK, launcher and extension source checkouts for joint debugging and
integration tests. Official artifact assembly lives in the separate
`f8distribution` repository. The current runtime is Zenoh-first:
- control plane, service discovery, pub/sub, and service-owned state use Zenoh by default
- video/audio data defaults to Zenoh latest-frame/latest-chunk transports
- local large-payload transfers can use Zenoh shared-memory optimization

## Layout
- `launcher/` — independent `f8platform` bootstrap and lifecycle manager, not an extension.
- `schemas` — generated Studio API contracts.
- `sdk` — [independent SDK repository](https://github.com/feel8-fun/f8sdk), including canonical shared protocols and generators.
- `docs` — architecture, service/operator manuals, and development guides.
- `sdk/python` — Python runtime SDK, Zenoh/mem transports, ServiceApp helpers.
- `sdk/cpp` — C++ runtime SDK, Zenoh transport and latest video/audio transports.
- `extensions/f8webstudio/f8studio_core` — typed graph document, patch, catalog, and compiler contracts.
- `extensions/f8webstudio/f8studio_server` — local Web Studio application service, API, CLI, and MCP.
- `extensions/f8webstudio/f8studio_web` — React graph editor and presentation workspaces.
- `extensions/f8mediagateway` — process-isolated Zenoh to WebRTC media gateway.
- `extensions/` — optional service implementations and the Unity game integration submodule; see [extension repositories](docs/development/extension-repositories.md).
- `config/extension-workspace.toml` — source checkout ownership and development integration policy.
- `extensions/*/config` — extension-owned service declarations.
- `extensions/*/resources` — extension-owned model definitions and default assets.
- `build/workspace/config` — generated development catalog, service launch adapters and runtime references.
- `build/workspace/runtime/bundles/<bundle>/<version>` — generated descriptions, executables and bundled libraries.
- `${F8_MODEL_ROOT}` — downloaded weights and user model definitions, outside the source workspace.
- `scripts` — Studio contract generation, describe regeneration, benchmarks, and integration tooling.

To remove old build/test output, use `pixi run -e build-check workspace_clean`.
For a rebuild with fresh local environments, see [clean and rebuild](docs/developers/build-from-source.md#clean-and-rebuild-the-development-workspace).

## Runtime Backend
- Default: `--bus-backend zenoh`
- Local tests may use `--bus-backend mem` where supported.
- Zenoh options are available through `--zenoh-config`, `--zenoh-connect`, `--zenoh-listen`, and `--zenoh-shm-pool-bytes`.

## Web Studio
- Build: `pixi run -e web-studio studio_web_build`
- Start: `pixi run -e web-studio studio_server`
- Open: `http://127.0.0.1:8210`

## Service installation and registration
Workspace tasks first generate `build/workspace/config/service-index.json` from
extension declarations. Studio loads this index deterministically. Startup does
not scan service directories, hash source files or run `--describe`.

- Install missing descriptions: `pixi run install_services`
- Regenerate development registrations: `pixi run workspace_catalog`
- Refresh descriptions after changing service definitions: `pixi run update_describes`
- Refresh one: `pixi run update_describes --service-class f8.pyengine`
- Copy and verify existing models: `pixi run install_services --migrate-resources /path/to/old/services --migrate-layout /path/to/old/services`
- Select another installation: set `F8_SERVICE_INDEX` to its index file.

Registered services receive an absolute `F8_MODEL_ROOT`. Standalone service commands use the platform user data directory unless `F8_MODEL_ROOT` is explicitly set. See [service registration](docs/development/service-registration.md).

## DL services
- Detector: `pixi run --manifest-path extensions/f8pydl/pixi.toml -e dl f8pydl_detector`
- Human detector: `pixi run --manifest-path extensions/f8pydl/pixi.toml -e dl f8pydl_humandetector`
- Classifier: `pixi run --manifest-path extensions/f8pydl/pixi.toml -e dl f8pydl_classifier`
- MediaPipe pose: `pixi run --manifest-path extensions/f8pymppose/pixi.toml -e mediapipe f8pymppose`
- Baseline benchmark (developer tooling): `pixi run -e build-check f8pydl_bench -- --model-yaml <yaml> --video <video>`

## Audio capture
- List recording devices: `build/bin/f8audiocap_service.exe --list-devices`
- Capture system mix (Windows): `build/bin/f8audiocap_service.exe --service-id audiocap --mode capture --backend wasapi`
- Capture microphone (SDL): `build/bin/f8audiocap_service.exe --service-id audiocap --mode capture --backend sdl --device 0`

## Documentation site
- Config: `mkdocs.yml`
- Dependencies: `docs/requirements.txt`
- Generate module pages (offline, requires `describe.json`): `pixi run python scripts/generate_service_docs.py`
- Validate generated content only (offline): `pixi run python scripts/generate_service_docs.py --check`
- Validate nav targets: `pixi run python scripts/check_docs_nav.py`
- Validate markdown links: `pixi run python scripts/check_docs_links.py`
- Build static site: `zensical build`
- Local preview: `zensical serve`

The independent SDK (`sdk/`) uses Apache-2.0. Extension authors may choose
their own licenses, including proprietary licenses, subject to their SDK and
other dependency license obligations. Studio and official extensions retain
their respective licenses.
