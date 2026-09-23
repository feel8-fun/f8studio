## When to Use

- Use `Bandpass Filter` when only a middle frequency range is interesting and both slow drift and fast noise should be suppressed.
- It is helpful for isolating rhythmic motion or a known movement band.
- This is the right choice when lowpass alone is too broad and highpass alone is too noisy.

## Common Wiring Patterns

- **Rhythm Isolation**: Feed pose- or audio-derived control signals through `Bandpass Filter` before periodicity or feature extraction.
- **Signal Cleanup**: Use it before `Envelope` when the target motion lives in a narrower band than the raw stream.
- **Parallel Analysis**: Compare unfiltered, lowpass, and bandpass branches side by side in wave visualization while tuning.

## Pitfalls / Gotchas

- **Bad Cutoff Ordering**: `low_cutoff` must stay below `high_cutoff`, and both should make sense for the source cadence.
- **Over-Narrow Window**: A very tight band can ring or remove the motion you actually care about.
- **Cadence Sensitivity**: Incorrect `sampleIntervalMs` settings distort the effective pass band.
