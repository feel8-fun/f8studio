## When to Use

- Use `TCode` to convert one or more normalized numeric control channels (0-1) into a standard TCode string stream (e.g., `L0500`, `V0123`).
- It is the essential formatting bridge between your abstract motion logic and the physical commands understood by most haptic devices.
- Use it to bundle multiple independent signals (Stroke, Vibrate, Roll, etc.) into a single, synchronized command packet.

## Common Wiring Patterns

- **Device Command Chain**: Feed it mapped numeric signals from `Range Map`. Branch the resulting `tcode` string output to `f8-serial-out` or a `TCodeViz` for inspection.
- **Synchronized Updates**: Align the `intervalMs` property with the tick rate of your graph (e.g., 20ms for 50Hz) to ensure smooth, jitter-free device movement.
- **Multi-Channel Authoring**: Use a single `TCode` operator to manage all axes of a complex device (like an OSR2) by wiring each axis to a dedicated input port.

## Pitfalls / Gotchas

- **Channel Semantic Mismatch**: If you wire a "Vibration" signal into a "Stroke" axis in the TCode node, the physical device will move incorrectly. Double-check your axis mappings (`L0`, `V0`, `R1`, etc.).
- **Visual Verification First**: Always view the emitted TCode string in a `TCodeViz` node *before* debugging serial ports or network connections. If the string looks wrong, the problem is upstream.
- **Resolution Limits**: TCode typically expects 4-digit precision (0-9999). Ensure your upstream values are not being clipped or rounded in a way that creates a "steppy" feel on the device.
