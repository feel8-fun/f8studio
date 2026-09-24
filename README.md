# f8studio

Runtime workspace for Feel8 Studio. The current runtime is Zenoh-first:
- control plane, service discovery, pub/sub, and service-owned state use Zenoh by default
- video/audio data defaults to Zenoh latest-frame/latest-chunk transports
- legacy SHM remains available only through explicit `legacy_shm` transport fields or old compatibility helpers

## Layout
- `api/specs` — service/operator contracts and generated protocol models.
- `api/bindings` — shared envelope/error models and binding notes.
- `profiles` — platform/feature profile schemas and examples.
- `docs/flows` — sequence/state docs for connection, config, playback, degrade/recover.
- `packages/f8pysdk` — Python runtime SDK, Zenoh/mem transports, ServiceApp helpers.
- `packages/f8cppsdk` — C++ runtime SDK, Zenoh transport and latest video/audio transports.
- `packages/f8studio_core` — typed graph document, patch, catalog, and compiler contracts.
- `packages/f8studio_server` — local Web Studio application service, API, CLI, and MCP.
- `packages/f8studio_web` — React graph editor and presentation workspaces.
- `packages/f8media_gateway` — process-isolated Zenoh to WebRTC media gateway.
- `services` — service manifests, static `describe.json`, and deployed C++ runtime binaries.
- `scripts` — codegen, describe regeneration, benchmarks, and migration tooling.

## Runtime Backend
- Default: `--bus-backend zenoh`
- Local tests may use `--bus-backend mem` where supported.
- Zenoh options are available through `--zenoh-config`, `--zenoh-connect`, `--zenoh-listen`, and `--zenoh-shm-pool-bytes`.

## SHM tools
- Legacy audio waveform viewer: `pixi run -e default python scripts/audioshm_viewer.py --service-id audiocap --use-event`

## Web Studio
- Build: `pixi run -e web-studio studio_web_build`
- Start: `pixi run -e web-studio studio_server`
- Open: `http://127.0.0.1:8210`

## Service discovery (startup speed)
Studio service discovery can avoid spawning `pixi run ... --describe` by using a static `describe.json` in each service directory (e.g. `services/f8/engine/describe.json`).

- Regenerate all: `pixi run -e default update_describes`
- Regenerate one: `pixi run -e default update_describes -- --service-class f8.pyengine`

## DL services
- Detector: `pixi run -e onnx dl_detector`
- Human detector: `pixi run -e onnx dl_humandetector`
- Classifier: `pixi run -e onnx dl_classifier`
- MediaPipe pose: `pixi run -e mediapipe mp_pose`
- Baseline benchmark: `pixi run -e onnx dl_bench -- --model-yaml <yaml> --video <video>`

## Audio capture
- List recording devices: `build/bin/f8audiocap_service.exe --list-devices`
- Capture system mix (Windows): `build/bin/f8audiocap_service.exe --service-id audiocap --mode capture --backend wasapi`
- Capture microphone (SDL): `build/bin/f8audiocap_service.exe --service-id audiocap --mode capture --backend sdl --device 0`

## Documentation site
- Config: `mkdocs.yml`
- Dependencies: `docs/requirements.txt`
- Generate module pages (offline, requires `describe.json`): `python scripts/generate_service_docs.py`
- Validate generated content only (offline): `python scripts/generate_service_docs.py --check`
- Validate nav targets: `python scripts/check_docs_nav.py`
- Validate markdown links: `python scripts/check_docs_links.py`
- Build static site: `zensical build`
- Local preview: `zensical serve`
