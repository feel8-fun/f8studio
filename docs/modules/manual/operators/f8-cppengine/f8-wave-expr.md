## When to Use

- Use `Wave Expr` to generate a procedural, looping waveform defined by a mathematical expression (e.g., `sin(t * 2 * pi)`) rather than by manually drawing points or keyframes.
- It is ideal for creating reusable modulation shapes (LFOs) that depend on time `t` and other parameters exposed as node properties.
- Use it when you need perfectly consistent, algorithmic motion that remains stable over long periods.

## Common Wiring Patterns

- **Clock-Driven LFO**: Drive the `t` input from a `f8-phase` or `f8-tick` clock source. Feed the resulting `value` output into an actuator-facing `Range Map` or `Envelope` operator.
- **Parametric Modulation**: Expose a small set of properties (e.g., `freq`, `amp`) as variables. Use them in your expression (e.g., `amp * sin(t * freq)`) to allow interactive tuning of the wave shape without editing the code.
- **Wave Combining**: Use the output of one `Wave Expr` to modulate the frequency or amplitude of another to create complex, evolving patterns.

## Pitfalls / Gotchas

- **Reserved Symbols**: The expression language has reserved names for mathematical functions. Avoid using variable names that collide with these functions (e.g., don't name a property `sin`).
- **Loop Period Alignment**: The `maxT` property defines the cycle period. If your upstream time source and the `maxT` setting disagree, your waveform may appear to "jump" or stutter at the end of each cycle.
- **Time Continuity**: Sudden jumps in the `t` input will cause immediate jumps in the output. If you are syncing to an external timeline, consider using a `Smooth Filter` downstream to handle discontinuities.
