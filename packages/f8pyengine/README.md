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
