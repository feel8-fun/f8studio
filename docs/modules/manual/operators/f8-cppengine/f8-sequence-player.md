## When to Use

- Use `Sequence Player` to play back pre-recorded or hand-authored motion sequences over time, exposing current signal values, frame indices, and play state.
- It is the primary tool for creating reproducible demos, scripted motion passages, or "preset" movements that can be triggered by scenario logic.
- Use it when you need a high-precision, frame-perfect reproduction of a specific motion pattern.

## Common Wiring Patterns

- **Preset Triggering**: Feed it a sequence payload (e.g., from a JSON file or an upstream generator). Branch the `value` output into `Range Map` or device nodes, and monitor the `done` port to trigger subsequent state changes.
- **Master Progress Sync**: Pair it with `Playback Sync` if other parts of the graph need to coordinate their behavior based on the sequence's current progress.
- **Interactive Playback**: Use the `play`, `pause`, and `stop` commands to control playback dynamically based on user input or detection events.

## Pitfalls / Gotchas

- **Payload Contract**: If the sequence data schema is incorrect (e.g., missing timestamps or wrong data types), the player may fail silently or produce stuttering motion. Validate your data with `f8-viz-text` first.
- **Ownership Confusion**: Avoid having multiple operators fighting for control of the same timeline. Decide whether the `Sequence Player` or an external clock (like `f8-phase`) is the authoritative timebase for a given branch.
- **Timing Jitter**: Unlike procedural oscillators, sequence playback resolution depends on the engine's tick rate. Ensure your `f8-tick` rate is high enough to capture the detail in your recorded sequence.
