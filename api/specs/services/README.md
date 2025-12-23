# Service Specs

Place domain-specific OpenAPI specs here (still NATS-bound). Examples:
- `playback.yaml` for libmpv control surface.
- `state.yaml` for snapshot/delta commands.
- `graph.yaml` for graph CRUD/validation endpoints.
- `engine.yaml` for engine runtime controls (apply graph, snapshot).

Keep shared envelope/errors in `../master.yaml` or `../bindings`.
