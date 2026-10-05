## When to Use

- Use `Decision` to evaluate typed Choice, Score, or Noul questions through a configured System-One provider.
- Provide structured `state` and optionally a `video` stream; the latest image is sampled only on an exec trigger.
- Set `providerId` to a connection configured in Studio and `routeQuestion` to a question in `questions`.

## Common Wiring Patterns

- Trigger `exec` at a controlled interval and route `decided` into logic that reads the selected `value`.
- Route `uncertain` to a review or idle path, and `error` to logging or an explicit fallback.
- Inspect `answers`, `probabilities`, and `confidence`; tune `minConfidence` and `minProbability` for the question type.
- Leave `studioUrl` empty when Studio launches the engine; set it explicitly for a separate server.

## Pitfalls / Gotchas

- Decisions are asynchronous. A newer pending trigger can supersede an older one, and expired results are discarded using `maxAgeMs`.
- `minIntervalMs` limits request cadence; sending more triggers does not create a request queue of every frame.
- Choice acceptance checks both confidence and selected probability; Score uses confidence, and Noul uses the stronger boolean probability.
- Changing configuration or lifecycle invalidates pending results. Keep per-request timing and counters on the `metrics` data output.
