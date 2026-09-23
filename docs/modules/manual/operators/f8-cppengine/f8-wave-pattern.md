## When to Use

- Use `Wave Pattern` when you want to design a looping or one-shot waveform by manually placing control points on a timeline rather than writing code or using simple oscillators.
- It is the best choice for hand-tuned motion shapes where a designer needs precise control over every peak, valley, and transition.
- Ideal for complex, non-periodic gestures or repeating patterns that cannot be easily described by a sine wave.

## Common Wiring Patterns

- **Designer LFO**: Feed the `t` input from a `f8-phase` or timeline source. Edit the `points` and `interp` (interpolation) properties to shape the motion, then send the result into a `Range Map` or output operator.
- **Gesture Library**: Keep a collection of `Wave Pattern` nodes as a preset library and switch between them with C++ engine control logic.
- **Interpolation Tuning**: Experiment with different interpolation modes (`pchip`, `akima`, `linear`) while watching the output on `f8-viz-wave` to find the most natural feel.

## Pitfalls / Gotchas

- **Interpolation Overshoot**: Non-linear interpolation modes like `spline` or `pchip` can introduce extra "hills" or "valleys" between your points if the points are spaced too closely or unevenly. Monitor the result visually to avoid unwanted movement.
- **Cycle Wraparound**: The `maxT` property defines the loop boundary. If your last point doesn't align with your first point at `maxT`, you will see a sharp "jump" when the wave repeats.
- **Point Density**: Keep the point list as sparse as possible. Adding too many unnecessary points makes the pattern harder to tune and can lead to jittery motion.
