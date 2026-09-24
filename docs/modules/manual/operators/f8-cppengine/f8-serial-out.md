## When to Use

- Use `Serial Out` as the final hardware sink for sending TCode or other text-based command streams to a device connected via a COM port (USB-Serial, Arduino, ESP32).
- It is the most common path for controlling DIY machines (like the OSR2 or SR6) and custom embedded hardware.
- Use it when you need low-latency, direct-to-metal communication without an intermediate network or Bluetooth layer.

## Common Wiring Patterns

- **Hardware Command Loop**: Feed it from an `f8-tcode` operator or any other finalized string-producing node. Always keep a `TCodeViz` node in parallel during initial hardware bring-up to see exactly what is being sent.
- **Safety Chains**: Ensure all safety limits, smoothing, and range mapping are handled *upstream* in the graph. The serial node should strictly be a "dumb" transport layer.
- **Port Setup**: Select the correct COM port and baud rate in the node properties. It is recommended to use 115200 or higher for smooth haptic feedback.

## Pitfalls / Gotchas

- **Port Access Conflicts**: The most common error is trying to open a port that is already in use by another application (e.g., Arduino IDE, another Studio instance). Verify the port is free before starting the scenario.
- **Baud Rate Mismatch**: If the baud rate on the node doesn't match the firmware on your device, you will see garbage characters or the device will not react at all.
- **Format Errors**: Validate the outgoing string format in the editor before blaming the hardware. Missing a newline character or a semicolon at the end of a command is a frequent cause of "unresponsive" hardware.
