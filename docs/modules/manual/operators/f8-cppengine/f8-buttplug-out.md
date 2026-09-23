## When to Use

- Use `Buttplug Out` when you want to target any hardware supported by the Buttplug.io (Intiface Desktop) ecosystem.
- It acts as the universal bridge for the Feel8 graph, allowing one logic chain to control hundreds of different haptic devices via the Buttplug protocol.
- Ideal for scenarios where the specific hardware is unknown or might be swapped by the end user.

## Common Wiring Patterns

- **Generic Haptic Out**: Feed it cleaned, bounded control values (0.0 to 1.0) from a `Range Map`. Avoid sending raw detector or feature outputs directly to the hardware.
- **Protocol Separation**: Keep your Buttplug-specific configuration separate from your core motion logic to ensure the graph remains portable to other protocols like TCode or LOVENSER.
- **Service Monitoring**: Use `f8-viz-text` to monitor the data being sent to the Buttplug server if a device isn't reacting as expected.

## Pitfalls / Gotchas

- **Capability Mismatches**: Not all devices support the same commands (e.g., some have only "Vibrate," while others have "Linear" or "Rotate"). Verify that your target device supports the command style you are sending from the graph.
- **Range Mapping Errors**: A bad upstream range (e.g., sending values > 1.0 or < 0.0) will often cause the Buttplug client or server to throw errors or ignore the commands entirely. Always use a `Range Map` immediately before this node.
- **Server Dependency**: This node requires Intiface Desktop or a compatible Buttplug server to be running on the host or network. If the server is not reachable, the node will appear idle or log connection errors.
