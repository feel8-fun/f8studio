## When to Use

- Use `Replayer` to play back data captured by `Recorder` for debugging and repeatable iteration.
- It is useful when you want to validate graph behavior against a known session without needing the live upstream source.
- This node helps turn intermittent runtime issues into deterministic repro cases.

## Common Wiring Patterns

- **Offline Repro**: Load a capture file and drive downstream logic from `sample` events while live inputs stay disconnected.
- **A/B Graph Tuning**: Replay the same session repeatedly while adjusting filters, mappings, or thresholds.
- **Looped Demo Source**: Enable `loop` for repeated playback during UI or behavior tuning.

## Pitfalls / Gotchas

- **File Compatibility**: `Replayer` expects the recorder output format; arbitrary files will not work.
- **Time Mode Choice**: Check `timeMode` before judging behavior, since playback pacing affects the graph feel.
- **False Confidence**: Replayed sessions are great for regression checks, but they do not replace live end-to-end validation.
