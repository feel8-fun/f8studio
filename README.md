# f8studio (greenfield skeleton)

Fresh workspace for the API-first, NATS-only architecture. Current focus is contracts, profiles, flows, and prototype scaffolding; no legacy code carried over.

## Layout
- `api/specs` — OpenAPI contracts (used for codegen) with NATS bindings via `x-nats-*`.
- `api/bindings` — shared envelope/error models and binding notes.
- `profiles` — platform/feature profile schemas and examples.
- `docs/flows` — sequence/state docs for connection, config, playback, degrade/recover.
- `packages/daemon` — C++ daemon (libmpv + state manager + NATS microservices).
- `packages/web` — TS client (web-only + enhanced via daemon), flow editor hooks.
- `packages/shared` — shared models/types; codegen outputs wrappers.
- `tests/contract` — spec-driven contract tests (NATS in-memory).
- `tests/integration` — end-to-end scenarios (web↔daemon).
- `scripts` — codegen, lint, local NATS bootstrap helpers.

## Next steps
- Hook `api/master.yaml` and `schemas/protocol.yml` into codegen/validation flow.
- Add scripts to generate TS/C++ stubs from OpenAPI and wrap NATS req/rep.
- Wire a minimal prototype: Web ping/echo + config apply via NATS in-memory broker.

## SHM tools
- Audio waveform viewer: `pixi run -e default python scripts/audioshm_viewer.py --service-id audiocap --use-event`

## Studio exe (Windows)
- Build: `pixi run -e default studio_exe`
- Requires: `pyinstaller` and `pillow` installed in the active environment.

## Service discovery (startup speed)
Studio service discovery can avoid spawning `pixi run ... --describe` by using a static `describe.json` in each service directory (e.g. `services/f8/engine/describe.json`).

- Regenerate all: `pixi run -e default update_describes`
- Regenerate one: `pixi run -e default update_describes -- --service-class f8.pyengine`
- Force live discovery (ignore `describe.json`): `pixi run -e default studio_live`

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
