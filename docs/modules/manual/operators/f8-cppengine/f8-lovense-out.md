## When to Use

- Use `Lovense Out` when your C++ engine graph needs to control Lovense devices such as Lush, Nora, or Max in real time.
- It acts as the final device-facing sink, converting your graph signals into commands that are sent to the Lovense Connect app or a dedicated dongle.
- Choose this for high-precision control of vibration, rotation, or contraction intensity.

## Common Wiring Patterns

- **Direct Intensity Control**: Feed it already-mapped control values (usually 0 to 20 for intensity) from an `f8-range-map`.
- **Logic Debugging**: Keep the `Lovense Mock Server` active during development to verify your graph logic without needing to wear or run the physical device.
- **Multi-Device Support**: Use multiple `Lovense Out` nodes to target different devices independently within the same scenario.

## Pitfalls / Gotchas

- **Intensity Resolution**: Lovense devices often have a limited number of "steps" (e.g., 0-20 or 0-100). Sending high-resolution floats (like 0.12345) will be rounded by the device transport layer, which can lead to a "steppy" feel if not handled carefully.
- **Transport Latency**: Communication via the Lovense Connect local API can introduce small delays. If the reaction feels laggy, reduce the command frequency or check your local network congestion.
- **Connection persistence**: Ensure the Lovense Connect app is running and your device is discovered before starting the Feel8 scenario, as the node will not automatically search for new devices once the graph is active.
