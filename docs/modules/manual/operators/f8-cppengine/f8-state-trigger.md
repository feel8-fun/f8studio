## When to Use

- Use `State Trigger` to fire an execution (exec) signal only when a specifically watched state value or data input changes.
- It is the most efficient way to handle "event-like" reactions (e.g., a button press, a mode change) without polling or calculating every single tick.
- Use it to trigger initialization sequences or to update "static" UI elements only when new data is actually available.

## Common Wiring Patterns

- **Event Gating**: Feed a stateful value from a `Control Panel` or a service state edge into it. Use the `changed` exec output to trigger side effects like re-configuring a service or starting a sequence.
- **Data Cleanup**: Use it to clear a diagnostic visualizer or log a message only when a "Done" signal or an error status actually occurs.
- **Property Binding**: Link the output to a node property that should only be updated when its input source changes, reducing redundant processing in downstream nodes.

## Pitfalls / Gotchas

- **High-Frequency Traps**: If the watched value changes every frame (e.g., raw bone coordinates or a noise signal), this node becomes effectively another high-frequency `Tick` source, negating its efficiency benefits.
- **Missed Changes**: Ensure that your graph logic relies on *changes* rather than *states*. If you need a continuous check, use a `Tick` node instead.
- **Hysteresis Requirements**: For noisy signals that flip-flop between values, consider adding a `Smooth Filter` or a small amount of dead-zone logic upstream to prevent the trigger from firing too often.
