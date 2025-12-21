# Integration Tests

- Boot Web (headless) + daemon + NATS in-memory; run end-to-end flows.
- Scenarios: connect/degrade/recover, config apply -> state delta, playback control happy-path and timeout/error.
- Assert single-client enforcement and state sync on reconnect (snapshot then deltas).
