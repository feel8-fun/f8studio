## When to Use

- Use `f8.cppengine` for native operators that need predictable high-frequency execution and lower Python overhead.
- Choose it when an operator is implemented and registered in the C++ runtime rather than `f8.pyengine`.
- Keep graph ownership explicit: operators hosted by this service must declare `f8.cppengine` as their service class.

## Common Wiring Patterns

- Feed data from capture or playback services into C++ operators, then publish results to downstream analysis, control, or presentation nodes.
- Use `buffered` delivery for normal graph scheduling and `callback` only for operators designed for callback execution.
- Split independent high-rate pipelines across service instances when one execution queue becomes a latency bottleneck.

## Pitfalls / Gotchas

- A C++ operator must be present in the deployed native build and static describe metadata; adding it only to a graph does not load code dynamically.
- Keep per-frame timing and counters on monitor/data telemetry rather than service state fields.
- Verify the native runtime bundle on each target platform because Linux results do not prove Windows binary loading or device behavior.
