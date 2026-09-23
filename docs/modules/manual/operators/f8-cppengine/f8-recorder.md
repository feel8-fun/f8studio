## When to Use

- Use `Recorder` to capture debug samples and sparse state changes from a running graph.
- It is intended for inspection, reproducibility, and offline analysis rather than production data logging at arbitrary scale.
- This node is especially helpful when you need to compare live behavior against later replays.

## Common Wiring Patterns

- **Debug Capture**: Trigger recording during a tuning session, then inspect the saved data after reproducing a problem.
- **Regression Fixture Builder**: Capture representative sessions that can later be replayed through `Replayer`.
- **Selective Session Logging**: Toggle `enabled` or `recording` around the specific time window you care about.

## Pitfalls / Gotchas

- **Path Management**: Make sure `path` points somewhere writable and predictable for your environment.
- **Not a Telemetry Stack**: This node is for targeted debug capture, not long-running archival logging.
- **No Live Counters In State**: Recorded sample/event counters are kept internally and in the recording file, not published as state, so a high-frequency recording session does not flood Studio state sync.
- **Scope Awareness**: Record only the samples you need, or the captured session becomes harder to reason about.
