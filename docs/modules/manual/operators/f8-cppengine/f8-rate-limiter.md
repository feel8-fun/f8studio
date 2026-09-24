## When to Use

- Use `Rate Limiter` when you need to ensure that an output value changes no faster than a specific "safe" rate (slope limiting) or doesn't exceed a maximum step size.
- It is a critical "safety node" for protecting physical actuators and hardware from sudden, violent movements caused by noise or upstream logic jumps.
- Best for smoothing out "teleporting" values from detectors or trackers that occasionally lose their lock.

## Common Wiring Patterns

- **Safety Guard**: Place it late in the control chain, after all mapping, smoothing, and logic but immediately *before* the hardware output node (e.g., `Lovense Out`, `Serial Out`).
- **Visual Validation**: Use `f8-viz-wave` to compare the "Unlimited" vs. "Limited" signals side-by-side to ensure the limits are safe for your specific hardware while remaining responsive enough for the user.
- **Actuator Lifetime**: Use conservative limits during general development to reduce mechanical wear on your devices, then widen them for "high-dynamic" scenarios when necessary.

## Pitfalls / Gotchas

- **Placement Order**: If placed too early in the graph, the rate limiter can distort the logic of subsequent operators (like Envelopes or Smooth Filters) that expect to see raw signal dynamics.
- **Responsiveness Lag**: Aggressive rate limiting will make your graph feel sluggish or "unresponsive," even if the upstream vision/audio detection is frame-perfect. Always tune this while physically testing the system.
- **Step vs Slope**: Understand the difference between limiting the *total change per second* (slope) and the *maximum allowed change in a single frame* (step). Misconfiguring these can lead to unexpected "damping" effect.
