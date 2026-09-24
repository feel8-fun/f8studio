## When to Use

- Use `Handy Out` when your graph needs to send normalized position commands directly to "The Handy" device over a network connection.
- It is a specialized device sink that maps `0..1` position values to the Handy HDSP API while maintaining the Feel8 graph's temporal consistency.
- Use it to synchronize your computer vision or audio-driven logic with a physical Handy device in real-time.

## Common Wiring Patterns

- **Standard Handy Setup**: Feed its `value` port a normalized `0..1` position signal. Apply signal shaping and safety limits upstream.
- **Monitoring Bridge**: Keep an `f8-viz-text` or `TCodeViz` attached to the input port while validating your connection to the device's API.
- **Connection Management**: Use the node properties to manage the Handy's connection key and transport mode (e.g., WebSocket vs REST).

## Pitfalls / Gotchas

- **Transport Role**: Treat this node strictly as a communication layer. Do not attempt to fix signal timing or motion logic here; those issues should be addressed in `Range Map`, filtering, or rate-control nodes.
- **Latency Over network**: Commands sent to The Handy are subject to network jitter. If the motion feels "stuck" or delayed, check your local Wi-Fi stability and the `intervalMs` setting to ensure you aren't overwhelming the device's buffer.
- **Command Telemetry Is Data, Not State**: HTTP status/result and sent-position diagnostics are emitted on data ports. They are not stored as state because successful commands can occur at motion tick rate.
- **Key Sensitivity**: The connection key is sensitive information. Avoid sharing session files that contain your unique device key if you are collaborating with others.
