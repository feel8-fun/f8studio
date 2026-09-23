## When to Use

- Use `Tempest` when you want a phase-driven, procedural waveform that has more "personality" and controllable asymmetry than a standard sine or cosine wave.
- It is ideal for motion patterns that require controllable curvature, "breathing" rhythms, or eccentric motion shapes (e.g., a fast stroke with a slow return).
- Best for creating organic, non-mechanical rhythmic oscillations in haptic or visual graphs.

## Common Wiring Patterns

- **Organic Oscillator**: Feed the `phase` input from `f8-phase`. Play with the `eccentric` and `curve` properties while watching the result on `f8-viz-wave` to find an interesting pulsing pattern.
- **Dynamic Shaping**: Map external signals (like audio energy) to the `amp` or `speed` parameters to make the waveform grow and shrink in intensity based on the environment.
- **Actuator Driver**: Use the output value after a `Range Map` to drive physical hardware with a more complex motion profile than a simple sine wave.

## Pitfalls / Gotchas

- **Complexity Overhead**: If the `eccentric` setting is too extreme, the resulting waveform can appear "broken" or glitchy if the upstream phase source is not perfectly stable.
- **Normalization Requirements**: Unlike `Cosine`, `Tempest` can produce a wider variety of shape ranges. Always re-normalize the result with a `Range Map` before sending it to hardware.
- **Design Intent**: Use it as a specialized "shaping" tool. If you only need a simple, predictable rhythmic wave, `f8-cosine` is often easier to tune and reason about.
