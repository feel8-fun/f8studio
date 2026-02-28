## f8pyengine

Python engine-side runtime service (`serviceClass=f8.pyengine`).

Entry wiring lives in `f8pyengine/pyengine_service.py` and is exposed via `f8pyengine/main.py`:
- `python -m f8pyengine.main --describe`
- `python -m f8pyengine.main --service-id engine1 --nats-url nats://127.0.0.1:4222`

### Lovense mock input

`Lovense Mock Server` (`operatorClass=f8.lovense_mock_server`) starts an in-process HTTP server (`POST /command`) compatible with Lovense Local API "Mobile" mode, and publishes each received command to the runtime-owned `event` state field (no exec flow required).

### Lovense waveforms

Prefer composition over monolithic Lovense wave nodes:

- `Lovense Program Adapter` (`operatorClass=f8.lovense_program_adapter`): converts `lovense_mock_server.event` (state) into:
  - `program` (ro state, dict) suitable for `Program Wave`
  - `amplitude` (ro state, 0..1) usable as modulation
  - `sequence` (ro state, dict) for Pattern -> hz step sequence (optional)
- `Program Wave` (`operatorClass=f8.program_wave`): produces `phase`, `phaseTurns`, `active`, `done` from a program dict.
  - Recommended wiring: state edge `lovense_program_adapter.program` -> `program_wave.program`.
- `Sequence Player` (`operatorClass=f8.sequence_player`): plays a `sequence` dict over time and outputs the current step `value`.
  - Wiring for Pattern: state edge `lovense_program_adapter.sequence` -> `sequence_player.sequence`.
- `Cosine` (`operatorClass=f8.cosine`): consumes `phase` (0..1) and generates a waveform sample.
  - Typical mapping for 0..1 output range: `dc=0.5`, `amp=0.5 * amplitude` (use `f8.expr` or `f8.range_map`).

Pattern->phase wiring example (reusable):
- `sequence_player.value` (Hz) -> `Phase.hz` (`operatorClass=f8.phase`)
- `Phase.phase` -> `Cosine.phase`

### Mix / Fill

`Mix (Silence Fill)` (`operatorClass=f8.mix_silence_fill`) outputs input `A` normally, but when `A` stays nearly-constant for `silenceMs`, it crossfades to `B` as a filler signal.

### Buttplug / Intiface bridge

`Buttplug Bridge` (`operatorClass=f8.buttplug_bridge`) provides a single-node integration to Intiface/Buttplug:

- Connects to `wsUrl` (default `ws://127.0.0.1:12345`)
- Publishes discovered devices into:
  - `availableDevices` (`["index|name", ...]`)
  - `deviceInfos` (full per-device capability map with `stepRange`/`durationRange`)
- Uses `selectedDevice` to choose the active target device (falls back to first available device)
- Accepts parallel control inputs:
  - `vibrate` (0..1)
  - `rotate` (-1..1)
  - `oscillate` (0..1)
  - `position` (0..1) with optional `positionDurationMs`
  - `stop` (boolean)

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
