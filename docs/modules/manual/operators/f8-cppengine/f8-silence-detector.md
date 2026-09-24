## When to Use

- Use `Silence Detector` when you want to detect that a signal has effectively stopped changing and expose that result as sparse graph state.
- It is a good fit for fallback routing, watchdog-style graph logic, and any situation where "no meaningful movement" should trigger another behavior.
- Best when the downstream node should react to a stable state change rather than reading a per-sample analysis signal.

## Common Wiring Patterns

- **Fallback Switching**: Feed the primary signal into `Silence Detector.value`, then use graph logic to switch `Switch Mixer.currentChannel` to a fallback port when `isSilent` becomes true.
- **State-Driven Logic**: Pair it with `State Expr`, `State Trigger`, or UI bindings when other graph nodes should respond to silence as a boolean condition.
- **Sparse Monitoring**: `isSilent` is intentionally the only runtime state output; activity timestamps are not published as state because active signals can update at tick rate.

## Pitfalls / Gotchas

- **Threshold Tuning**: If `deltaThreshold` is too small, normal noise will keep the node from ever becoming silent; if too large, subtle real motion may be ignored.
- **Exec-Driven Sampling**: Detection updates on exec, so the effective responsiveness depends on how often the graph drives the node.
- **Signal Semantics**: It detects lack of change, not low absolute amplitude. A constant non-zero signal still becomes "silent" under this definition.
