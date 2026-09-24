## When to Use

- Use `Wave Funscript` when the motion in your graph should be driven by an existing authored `.funscript` file (commonly used in haptic media) rather than a synthetic oscillator or a hand-drawn pattern.
- It is the primary tool for achieving repeatable, frame-accurate playback of synchronized haptic scripts within the Feel8 system.
- Best for scenarios where you want to "remix" or "re-map" an existing script to different hardware or use it as a part of a larger hybrid automation graph.

## Common Wiring Patterns

- **Scripted Playback**: Point the node to a local `.funscript` file. Select the desired axis (usually "L0"), drive the `t` input from the current playback time (e.g., from `Playback Sync`), and route the resulting `value` to your hardware out nodes.
- **Dynamic Rescaling**: Pass the `value` output through a `Range Map` to adjust script intensity in real-time while it is playing.
- **Interpolation Sweep**: Start with the default `linear` interpolation for a faithful reproduction of the original author's intent. Only use smoother modes like `spline` if the script points are too sparse and cause mechanical jitter on your specific device.

## Pitfalls / Gotchas

- **Duration Mismatches**: If your timing source (`t`) exceeds the duration of the funscript, the wave will loop or stop depending on the `wrap` settings. ensure your master clock matches the script's expectations.
- **Axis Confusion**: Many funscripts contain multiple axes of data. Verify you have selected the correct axis (e.g., `L0` for stroke, `V0` for vibration) in the node properties.
- **Interpolation Overshoot**: Smoother interpolation modes can sometimes introduce "overshoot" or "wobble" between far-apart points that was not in the original script. Always monitor the wave shape in `f8-viz-wave` before final use.
