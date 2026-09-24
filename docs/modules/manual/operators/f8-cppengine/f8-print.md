## When to Use

- Use `Print` when you need a quick, execution-driven (exec) diagnostic sink to output values or messages to the Studio console or engine logs.
- It is a development-time tool for verifying that specific logic branches are actually firing and for inspecting variable values in real-time.
- Not intended for production user interfaces; use `f8-viz-text` for persistent on-canvas monitoring.

## Common Wiring Patterns

- **Branch Verification**: Trigger it from a `Sequence` or `State Trigger` only on the specific branch you are diagnosing. This ensures the log message reflects the correct execution context.
- **Value Inspection**: Connect the input to any data port (scalar, string, or complex JSON) to see its contents at the exact moment of execution.
- **Trigger Logging**: Use it to confirm that "one-shot" events (e.g., successful calibration, sequence done) have occurred without needing to watch a visualizer constantly.

## Pitfalls / Gotchas

- **Log Spam**: Leaving too many `Print` nodes active in a high-frequency loop (like a `Tick` node) will flood the console, potentially hiding more important system logs and slightly impacting performance.
- **Sink Behavior**: `Print` is a terminal "sink." If you need to pass the data further down the graph for other operators to use, ensure you branch the signal *before* it reaches the print node.
- **Ordering**: Remember that messages will appear in the order they are executed in the engine thread, which may differ from their spatial arrangement on the Studio canvas.
