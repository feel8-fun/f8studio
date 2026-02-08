## f8pyengine

Python engine-side runtime service (`serviceClass=f8.pyengine`).

Entry wiring lives in `f8pyengine/pyengine_service.py` and is exposed via `f8pyengine/main.py`:
- `python -m f8pyengine.main --describe`
- `python -m f8pyengine.main --service-id engine1 --nats-url nats://127.0.0.1:4222`

### Lovense mock input

`Lovense Mock Server` (`operatorClass=f8.lovense_mock_server`) starts an in-process HTTP server (`POST /command`) compatible with Lovense Local API "Mobile" mode, and publishes each received command to the runtime-owned `event` state field (no exec flow required).

### Lovense waveforms

- `Lovense Thrusting Wave` (`operatorClass=f8.lovense_thrusting_wave`): subscribe the mock server's `event` via a state edge into `lovenseEvent`, then tick it via exec to generate a continuous 0..1 thrusting waveform.
- `Lovense Vibration Wave` (`operatorClass=f8.lovense_vibration_wave`): basic step-sequencer interpretation of Pattern events (`S:<ms>#` + `strength` list), also driven by tick exec.

### Mix / Fill

`Mix (Silence Fill)` (`operatorClass=f8.mix_silence_fill`) outputs input `A` normally, but when `A` stays nearly-constant for `silenceMs`, it crossfades to `B` as a filler signal.
