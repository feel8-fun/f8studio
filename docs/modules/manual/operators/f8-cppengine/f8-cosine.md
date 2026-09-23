## When to Use

- Use `Cosine` to transform a normalized phase signal (usually 0 to 1) into a smooth, periodic waveform based on the cosine function.
- it is a fundamental building block for rhythmic motion, breathing patterns, or periodic modulation in your graph.
- Use it when you need a smooth, continuous oscillation that can be easily tuned for amplitude and DC offset.

## Common Wiring Patterns

- **Standard LFO**: Feed the `phase` input from an `f8-phase` or `f8-tick` operator. Send the mapping `value` into a `Range Map` or `f8-viz-wave`.
- **Rhythmic Modulation**: Use multiple `Cosine` nodes with different frequencies to create complex, multi-layered "interference" patterns for more organic motion.
- **Dynamic Tuning**: Bind the `amp` (amplitude) or `dc` (offset) properties to other control nodes to dynamically change the intensity or center point of the oscillation.

## Pitfalls / Gotchas

- **Phase Dependency**: If the upstream phase source is jittery or incorrect, no amount of amplitude tuning will fix the resulting waveform. Verify your phase source first.
- **Output Range**: By default, a cosine wave moves between `-amp+dc` and `+amp+dc`. Ensure your downstream nodes (like `f8-tcode`) are prepared for these values, or use a `Range Map` to normalize them.
- **Phase Wraparound**: If your phase source doesn't wrap cleanly at 1.0, the cosine wave will have a visible "jump" or discontinuity.
