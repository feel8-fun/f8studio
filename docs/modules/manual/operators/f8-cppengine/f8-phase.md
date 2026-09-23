## When to Use

- Use `Phase` when your graph needs a normalized oscillator (0 to 1) or a continuous cycle counter to drive periodic events.
- It is the standard "clock" for the Feel8 graph, serving as the authoritative driver for `Cosine`, `Wave Pattern`, and many rhythmic modulation chains.
- Use it to synchronize multiple independent operators to the same temporal heartbeat.

## Common Wiring Patterns

- **Master Clock**: Feed the `phase` output into a variety of waveform generators (`Cosine`, `Tempest`). Use `phaseTurns` for logging or sync branches that need to know how many full cycles have elapsed.
- **Interactive Tempo**: Connect the `hz` (frequency) input to an external control (like a slider or an audio BPM analyzer) to make your motion react to the environment in real-time.
- **Triggered Reset**: Use the `reset` input or command to restart the oscillator from 0 when a specific event occurs (e.g., a new track starts).

## Pitfalls / Gotchas

- **Redundant Clocks**: If your graph already has an authoritative timebase (e.g., from a video player or an external MIDI clock), adding a secondary `Phase` node can make the system behavior difficult to synchronize and reason about.
- **Visual Validation**: Always verify the phase reset and wrap behavior with an `f8-viz-wave` node before connecting it to physical hardware to avoid sudden mechanical jerks.
- **Hz Resolution**: Very high frequencies might produce aliasing-like effects if the graph's overall update rate is too low.
