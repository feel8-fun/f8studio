## When to Use

- Use `Smooth Filter` when a numeric signal already exists but contains jitter, noise, or sudden spikes that need temporal stabilization.
- It is a essential second-stage "cleanup" node after feature extraction (audio) or coarse coordinate mapping (vision/pose).
- Best for creating smooth, organic motion from noisy real-world data sources.

## Common Wiring Patterns

- **Cleanup Pipeline**: Place it after an `Envelope` or `Range Map` operator. Compare the raw vs. filtered signal in parallel on `f8-viz-wave` to find the ideal smoothing coefficient.
- **Actuator Guard**: Place a filter immediately before high-frequency device outputs to prevent "chatter" and reduce mechanical wear on actuators.
- **Signal Separation**: Use different filter settings for different semantic signals (e.g., a "slow" filter for average loudness and a "fast" filter for onset tracking).

## Pitfalls / Gotchas

- **Responsiveness Tradeoff**: Higher smoothing values increase stability but introduce noticeable latency (lag). Always tune the filter while interacting with the system to find the "sweet spot" for your use case.
- **Input Error Hiding**: A heavy filter can hide systemic instability or logic errors in the upstream path. Always validate your raw signal before applying aggressive smoothing.
- **Initialization Jumps**: The filter may produce a large "jump" output when the first data point arrives if the internal state isn't reset properly.
