## When to Use

- Use `Lowpass Filter` to smooth noisy scalar or vector signals while preserving slower movement.
- It is a good fit when downstream control logic should ignore jitter or small high-frequency fluctuations.
- Reach for it when you want a more frequency-aware smoother than a simple EMA.

## Common Wiring Patterns

- **Noise Cleanup**: Place it after pose-derived values, sensor streams, or expression outputs before mapping to hardware.
- **Control Stabilization**: Pair it with `Range Map` and `Rate Limiter` to build a calmer control chain.
- **Pre-Visualization**: Filter a signal before plotting it in a wave view to see long-term movement more clearly.

## Pitfalls / Gotchas

- **Latency Tradeoff**: Lower cutoffs reduce noise but also delay the response.
- **Sample Interval Match**: Keep `sampleIntervalMs` aligned with the actual update cadence, or the filter shape will be misleading.
- **Wrong Tool**: Use `Highpass Filter` or `Bandpass Filter` instead when you need to isolate faster motion components.
