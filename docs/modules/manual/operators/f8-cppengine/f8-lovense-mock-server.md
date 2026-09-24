## When to Use

- Use `Lovense Mock Server` when you need to test your Lovense local API integrations without having high-end hardware physically connected or powered on.
- It is invaluable for release rehearsals, development of complex haptic adapters, and protocol debugging during rapid iteration.
- Use it to simulate different device types (Lush, Nora, etc.) and verify that your commands are correctly formatted.

## Common Wiring Patterns

- **Validation Branch**: Keep the mock server active on a side branch. Use it to confirm that your `Lovense Out` or downstream parser/post-process nodes are receiving the expected Lovense command payloads.
- **Protocol Sniffing**: Inspect the emitted `event` data output and execution triggers using `Print` or `Text Viz` nodes to see exactly how the "virtual device" is responding to your graph.
- **Automated Testing**: Use it in automated scenario tests to verify that a logic chain produces the correct device commands without needing human interaction.

## Pitfalls / Gotchas

- **Virtual vs Physical**: The mock server validates protocol flow and message timing, but it cannot simulate the physical feel, mechanical latency, or battery/Bluetooth nuances of the actual hardware.
- **Events Are Data, Not State**: Incoming commands can arrive at device-command rate, so the latest command is exposed through the `event` data port instead of a high-frequency state field.
- **Bind Conflicts**: If the mock server fails to start or appears "silent," check if another service (or even the actual Lovense Connect app) is already using the local API ports on your machine.
- **Initialization order**: The mock server should ideally be started before the nodes that attempt to connect to it.
