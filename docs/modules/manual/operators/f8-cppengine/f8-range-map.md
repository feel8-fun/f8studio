## When to Use

- Use `Range Map` when one numeric range (e.g., 0-100) must be clipped and remapped into another (e.g., 0-1).
- It is the default scaling node for preparing any signal before it reaches actuator-facing outputs or visualizations.
- Use it to convert raw sensor data, detection scores, or feature magnitudes into normalized control values.

## Common Wiring Patterns

- **Signal Normalization**: Feed it cleaned scalar values from a detector or audio feature node, then send the mapped output to `f8-tcode`, `f8-serial-out`, or `f8-viz-wave`.
- **Calibration Loop**: Tune `inMin` and `inMax` against real-time observed source values (using `f8-viz-wave` for reference) before finalizing the `outMin` and `outMax` for your hardware.
- **Curve Shapping**: Use the `exponent` property to apply non-linear curves (e.g., exponential growth for more sensitivity at lower values).

## Pitfalls / Gotchas

- **Input Calibration**: A bad or uncalibrated input range (`inMin/inMax`) is the most frequent cause of "dead" actuators or clipping. Verify your source signal range first.
- **Output Clipping**: Out-of-bounds input values are clipped to `outMin/outMax` by default. ensure your mapping account for the full expected range of the signal.
- **Semantic Clarity**: Map your values into a 0-1 range early in the graph to maintain a consistent "normalized" signal flow, moving the device-specific scaling to the very end of the chain.
