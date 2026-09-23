## When to Use

- Use `Envelope` when you want to convert a noisy, high-frequency, or fast-changing scalar signal into a smoother "magnitude trace" or follower.
- It is most commonly used in audio-reactive graphs (following loudness) and gesture-reactive graphs (following motion intensity).
- Use it to extract the general "energy" of a signal while ignoring its individual peaks and valleys.

## Common Wiring Patterns

- **Energy Follower**: Feed it from a feature output (like audio loudness). Send the resulting envelope into a `Smooth Filter` or `Range Map` to drive motion.
- **Visual Saliency**: Use the envelope of a motion signal to drive the transition of a classifier's weight, making the system more sensitive when more motion is detected.
- **Sequence Tuning**: Place it before downstream scaling so that later nodes in the chain receive a stable, predictable signal range.

## Pitfalls / Gotchas

- **Latency vs. Smoothness**: Over-smoothing the envelope (long attack/decay) can make a graph feel "heavy" or laggy, even if the source data is accurate. Tune the attack and release times carefully.
- **Input Quality**: If the input source is fundamentally wrong, empty, or entirely noise, no amount of envelope tuning will produce a meaningful control signal. Validate your source with `f8-viz-wave` first.
- **Range Assumptions**: Envelopes often produce values in a different range than the input. Use a `Range Map` immediately after the envelope to bring it into a standard 0-1 control space.
