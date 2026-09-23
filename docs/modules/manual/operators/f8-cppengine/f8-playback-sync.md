## When to Use

- Use `Playback Sync` when your graph needs to be aware of the timeline, progress, or state of a media stream or sequence playback (active, paused, seeking).
- It is the primary tool for tying complex motion logic to a shared external playback clock (e.g., from `f8-implayer` or `f8-sequence-player`).
- Use it to synchronize state machine transitions with specific timestamps in a media file.

## Common Wiring Patterns

- **Master Clock Lock**: Pair it with an `IM Player` or `Sequence Player` source. Feed the resulting progress into downstream operators to ensure they stay perfectly in sync with the media content.
- **UI Progress Monitoring**: Route the progress and duration outputs to `Text Viz` or a custom dashboard to give the user a visual indication of the current scenario timeline.
- **Conditional Scripting**: Use the `active` or `looping` status to enable/disable specific parts of your graph based on whether media is currently playing.

## Pitfalls / Gotchas

- **Timebase Competition**: Drift and unexpected "jump" resets are common when multiple nodes in a graph try to be the authoritative timebase. Choose ONE source as the master clock and use `Playback Sync` to distribute it.
- **Granularity Differences**: If your sync source has low time resolution, downstream motion may appear "steppy." Consider using a `Smooth Filter` to interpolate between progress updates if perfectly smooth motion is required.
- **Disconnection Handling**: Always consider what should happen if the playback source is stopped or disconnected. Use default values or fallback logic to prevent actuators from getting "stuck" at a specific position.
