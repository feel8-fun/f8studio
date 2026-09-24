## When to Use

- Use `Highpass Filter` to remove slow movement and emphasize quicker changes in a signal.
- It is useful when downstream logic should react to impacts, pulses, or short-term motion rather than steady offsets.
- This operator is often a good companion to periodicity or onset-style analysis.

## Common Wiring Patterns

- **Pulse Extraction**: Put it before an envelope or threshold detector to highlight transient motion.
- **Drift Rejection**: Use it after a slowly moving control source when only rapid changes should pass through.
- **Motion Feature Branch**: Split a signal into lowpass and highpass branches for separate visual or control purposes.

## Pitfalls / Gotchas

- **Cutoff Too High**: Aggressive cutoffs can make the output feel thin or unstable.
- **Sample Interval Match**: `sampleIntervalMs` must reflect the real update rate for the filter to behave as expected.
- **Noise Boost**: High-pass filtering can make small upstream jitter more visible, not less.
