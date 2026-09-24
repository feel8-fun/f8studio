## When to Use

- Use `Program Wave` when your graph needs to emit or shape a structured "wave/program" payload (a collection of points or parameters) rather than just a single scalar value.
- It is essential for authoring complex motion sequences or "scripts" upstream of playback nodes or device-specific formatters.
- Use it when you want to group multiple motion parameters into a single reusable "program" that can be shared across multiple output channels.

## Common Wiring Patterns

- **Motion Sequencer**: Pair it with a `Tick` or `Phase` source. Use the output to drive a `Sequence Player` or a protocol-specific node like `f8-tcode`.
- **Global Modulation**: Keep the `Program Wave` node upstream from individual device nodes so that one complex motion pattern can drive multiple hardware targets in sync.
- **Payload Inspection**: Always route a branch through `f8-viz-text` to verify that the generated point data matches the expected schema for your target protocol.

## Pitfalls / Gotchas

- **Schema Strictness**: Program-oriented nodes depend on a very specific internal payload structure. Any mismatch in port names or data types will cause downstream nodes to ignore the data silently.
- **Abstraction Overload**: If your graph only needs to move a single actuator between two points, a `Program Wave` might be more complex than necessary. For simple cases, sticks to `f8-cosine` or `f8-range-map`.
- **Frequency Matching**: Ensure the update rate of your program is high enough to avoid stuttering on the physical device, but low enough common bandwidth limits of serial or network protocols.
