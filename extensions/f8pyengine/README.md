## f8pyengine

Python engine-side runtime service (`serviceClass=f8.pyengine`).

Entry wiring lives in `f8pyengine/pyengine_service.py` and is exposed via `f8pyengine/main.py`:
- `python -m f8pyengine.main --describe`
- `python -m f8pyengine.main --service-id engine1`
- Local in-memory tests: `python -m f8pyengine.main --service-id engine1 --bus-backend mem`

### Lovense mock input

`Lovense Mock Server` (`operatorClass=f8.lovense_mock_server`) starts an in-process HTTP server (`POST /command`) compatible with Lovense Local API "Mobile" mode, and publishes each received command to the runtime-owned `event` state field (no exec flow required).

### Lovense local output

`Lovense Out` (`operatorClass=f8.lovense_out`) sends `POST /command` requests to Lovense Local API:

- Two-channel model:
  - `sendPositionCmd` + data port `position(0..1)` -> sends Lovense `Position` command (`apiVer=1`)
  - `sendFunctionCmd` + state fields (`vibrate/rotate/.../stop/timeSec/...`) -> sends Lovense `Function` command (`apiVer=1`)
- Recommended wiring:
  - realtime position: `tick.exec -> lovense_out.sendPositionCmd`
  - transactional function apply: `ui_or_logic.exec -> lovense_out.sendFunctionCmd`
- Toy inventory:
  - node auto-sends one `GetToys` on activation and updates `availableToys`
- `minSendIntervalMs` throttles `Position` sends (default `100ms`, ~10Hz)
- Optional `Function` fields (`loopRunningSec`, `loopPauseSec`, `toy`) are omitted when empty
- `toy` uses option pool from `availableToys`
- Header: `X-platform = platformName`

### Lovense waveforms

Prefer composition over monolithic Lovense adapter nodes:

- `Lovense Mock Server` (`operatorClass=f8.lovense_mock_server`) provides the ingress event state.
- `Python Script` (`operatorClass=f8.python_script`) should parse each Lovense sub-protocol explicitly.
- `Program Wave` (`operatorClass=f8.program_wave`): produces `phase`, `phaseTurns`, `active`, `done` from a program dict.
- `Sequence Player` (`operatorClass=f8.sequence_player`): plays a `sequence` dict over time and outputs the current step `value`.
- `Cosine` (`operatorClass=f8.cosine`): consumes `phase` (0..1) and generates a waveform sample.

Recommended pattern:

- `lovense_mock_server.event` -> parser `python_script.lovenseEvent`
- parser script emits normalized protocol payloads
- downstream user script maps those payloads into `program`, `amplitude`, or `sequence`

Pattern->phase wiring example (reusable):

- `sequence_player.value` (Hz) -> `Phase.hz` (`operatorClass=f8.phase`)
- `Phase.phase` -> `Cosine.phase`

### Mix / Fill

`Switch Mixer` (`operatorClass=f8.switch_mixer`) switches among any number of user-defined input channels under `currentChannel`, with optional smooth crossfade via `fadeMs`. If the selected channel stops receiving valid samples, the node keeps repeating that channel's last valid value.

### Buttplug / Intiface bridge

`Buttplug Out` (`operatorClass=f8.buttplug_out`) provides a single-node integration to Intiface/Buttplug:

- Connects to `wsUrl` (default `ws://127.0.0.1:12345`)
- Publishes discovered devices into:
  - `availableDevices` (`["index|name", ...]`)
  - `deviceInfos` (full per-device capability map with `stepRange`/`durationRange`)
- Uses `selectedDevice` to choose the active target device (falls back to first available device)
- Split command channels:
  - `sendPositionCmd` + data port `position` (0..1)
  - `sendFunctionCmd` + state fields `vibrate` / `rotate` / `oscillate` / `stop`
- `defaultPositionDurationMs` controls position duration for `sendPositionCmd`

Feature-index state fields (`*FeatureIndex`) allow selecting a specific feature per output type; `-1` broadcasts to all matching features.

### The Handy HDSP output

`Handy Out` (`operatorClass=f8.handy_out`) drives The Handy over REST v2 HDSP using normalized input.

- Exec-driven sink: `tick.exec -> handy_out.exec`
- Data input: `signal(0..1) -> handy_out.value`
- Auto mode: `ensureHdspMode=true` sends `PUT /mode {\"mode\":2}` when needed
- Motion command: `PUT /hdsp/xpt` with mapped `position` (0..100), `duration`, `immediateResponse`, `stopOnTarget`
- Mapping: `value(0..1)` -> clamp -> optional `invert` -> `[minPercent, maxPercent]`
- Header auth: `X-Connection-Key = connectionKey`
- Rate-limit aware: consumes `X-RateLimit-*` headers and applies temporary backoff
