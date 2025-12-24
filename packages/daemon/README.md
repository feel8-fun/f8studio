# Daemon (C++)

- Headless local service; bridges Web commands to libmpv and other components.
- Exposes NATS microservice endpoints per `api/specs/master.yaml` and future domain specs.
- Uses State Manager pattern (in-memory KV, delta pub/sub). No disk persistence in v1.
- Build: cmake + conan2.0; wrap generated stubs from OpenAPI (NATS binding) for handlers.
- Engine runtime design (single-graph, hot updates, cross-instance wiring) is documented in `docs/services/engine.md`.
