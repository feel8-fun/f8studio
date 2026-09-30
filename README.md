# f8studio

Runtime workspace for Feel8 Studio. The current runtime is Zenoh-first:
- control plane, service discovery, pub/sub, and service-owned state use Zenoh by default
- video/audio data defaults to Zenoh latest-frame/latest-chunk transports
- local large-payload transfers can use Zenoh shared-memory optimization

## Layout
- `schemas` — canonical protocol, runtime keys, and generated Studio API contracts.
- `docs` — architecture, service/operator manuals, and development guides.
- `packages/f8pysdk` — Python runtime SDK, Zenoh/mem transports, ServiceApp helpers.
- `packages/f8cppsdk` — C++ runtime SDK, Zenoh transport and latest video/audio transports.
- `packages/f8studio_core` — typed graph document, patch, catalog, and compiler contracts.
- `packages/f8studio_server` — local Web Studio application service, API, CLI, and MCP.
- `packages/f8studio_web` — React graph editor and presentation workspaces.
- `packages/f8media_gateway` — process-isolated Zenoh to WebRTC media gateway.
- `config/service-index.json` — explicit service registrations and model storage location.
- `config/services` — tracked, platform-specific launch declarations.
- `runtime/bundles/<bundle>/<version>` — generated descriptions, executables, and bundled libraries/resources.
- `resources/models` — shared model definitions and installed weights.
- `scripts` — codegen, describe regeneration, benchmarks, and migration tooling.

## Runtime Backend
- Default: `--bus-backend zenoh`
- Local tests may use `--bus-backend mem` where supported.
- Zenoh options are available through `--zenoh-config`, `--zenoh-connect`, `--zenoh-listen`, and `--zenoh-shm-pool-bytes`.

## Web Studio
- Build: `pixi run -e web-studio studio_web_build`
- Start: `pixi run -e web-studio studio_server`
- Open: `http://127.0.0.1:8210`

## Service installation and registration
Studio loads `config/service-index.json` deterministically. Startup does not scan service directories, hash source files, or run `--describe`.

- Install missing descriptions: `pixi run install_services`
- Refresh descriptions after changing service definitions: `pixi run update_describes`
- Refresh one: `pixi run update_describes --service-class f8.pyengine`
- Copy and verify existing models: `pixi run install_services --migrate-resources /path/to/old/services --migrate-layout /path/to/old/services`
- Select another installation: set `F8_SERVICE_INDEX` to its index file.

Registered services receive an absolute `F8_MODEL_ROOT`. Standalone service commands use the platform user data directory unless `F8_MODEL_ROOT` is explicitly set. See [migration notes](docs/development/service-registration-migration.md).

## DL services
- Detector: `pixi run -e onnx f8pydl_detector`
- Human detector: `pixi run -e onnx f8pydl_humandetector`
- Classifier: `pixi run -e onnx f8pydl_classifier`
- MediaPipe pose: `pixi run -e mediapipe f8pymppose`
- Baseline benchmark: `pixi run -e onnx f8pydl_bench -- --model-yaml <yaml> --video <video>`

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
