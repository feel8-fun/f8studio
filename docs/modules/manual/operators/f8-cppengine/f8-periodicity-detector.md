## When to Use

- Use `Periodicity Detector` to estimate whether a scalar signal contains a stable repeating pattern.
- It is useful for motion quality checks, rhythm detection, or gating logic based on periodic confidence.
- This operator works best when the upstream signal already represents a meaningful one-dimensional feature.

## Common Wiring Patterns

- **Periodic Motion Gate**: Feed `is_periodic` or `confidence` into downstream state or exec logic to enable outputs only during stable repetition.
- **Tempo/Period Probe**: Use `periodMs` or `period_hz` to inspect the dominant rhythm of an input stream.
- **Feature Stack**: Place it after detrending and filtering so the detector sees a cleaner signal.

## Pitfalls / Gotchas

- **Garbage In**: Raw noisy inputs usually need filtering first, or confidence will be erratic.
- **Window Tuning**: `window`, `min_lag`, and `max_lag` strongly affect what periods can be detected.
- **Confidence Semantics**: High confidence means repeatability, not necessarily high amplitude or good control quality.
