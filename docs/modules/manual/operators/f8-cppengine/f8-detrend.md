## When to Use

- Use `Detrend` when a signal has slow drift or baseline movement that should be removed before downstream analysis.
- It is useful ahead of envelope, periodicity, or threshold-based logic that should react to motion rather than offset.
- This operator works well for both scalar streams and small vectors.

## Common Wiring Patterns

- **Motion Isolation**: Place `Detrend` before `Envelope` or `Range Map` when a source slowly wanders over time.
- **Pre-Filter Stage**: Remove baseline drift before applying `Lowpass Filter`, `Highpass Filter`, or periodicity analysis.
- **State Reset Path**: Use `reset_on_state_change` when upstream mode changes would otherwise carry old baseline history forward.

## Pitfalls / Gotchas

- **Too Aggressive Alpha**: Over-tuning the detrend state can erase intentional slow motion along with unwanted drift.
- **History Reset**: Resetting too often can produce abrupt jumps in the output.
- **Expectation Gap**: `Detrend` removes baseline; it is not a substitute for smoothing or band-limiting.
