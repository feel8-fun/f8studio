# Engine Service (runtime graph host)

- Runs a single active graph per instance; manages state, executes nodes/operators, and handles cross-instance wiring (state/data buses).
- Applies to both native app engines and web-worker engines; same runtime model, different host/runtime env.
- Built on the minimal ServiceHostBase (state/KV/heartbeat/register) plus engine-specific data bus pub/sub and graph runtime.

## Responsibilities
- Load graph snapshot from KV (`graph`) at startup; initialize state from seeded keys.
- Apply graph diffs/commands at safe points (end of tick), validate, update in-memory graph, bump version (KV revision), and write the new snapshot back to KV. On validation failure, keep the old graph running.
- Maintain per-instance state manager; publish `f8.state.<instanceId>.delta` locally; fan out cross-instance state via `state.<instanceId>.set` for subscribed remote nodes.
- Data bus wiring (engine-only): subscribe/publish to compiled cross-edge subjects (`f8.bus.<edgeId>`), with backpressure/queueing policies as needed.
- Lifecycle with master: register → ensure per-instance KV bucket exists → run → unregister → clean up bucket (if owned by master).

## Tick-based scheduling
- Engine runs on discrete ticks to align processing of mixed-rate realtime data; all node executions within a tick see a consistent snapshot of inputs/state.
- End-of-tick is the safe point: apply queued graph diffs, commit state transitions, emit deltas/fanout, and advance clocks. This avoids mid-tick structural changes and keeps multi-rate streams coherent.
- Ticks can be fixed-step or adaptive; external triggers (e.g., high-rate streams) enqueue work but effects become visible on the next tick boundary.

## Web-engine UI hooks (visualization/control)
- Web engine shares the same runtime model, plus UI renderers per node.
- Data printer node: subscribes to its data bus input; on tick, renders the latest payload into the node UI (React component), no extra side effects.
- Input control node: renders a UI control (e.g., slider); user interactions update the node state; state changes fan out via the normal state bus (`state.<instanceId>.set`) to downstream nodes (web or native services).
- UI is purely view/control; execution graph semantics stay the same. Data/State flow uses the existing buses; DOM work lives inside the renderer.

## Inputs/outputs (NATS-facing)
- Command subjects follow `f8.<serviceSlug>.<instanceId>.<verb>` (e.g., apply graph diff, pause/resume, state snapshots). Cross-instance data uses `f8.bus.<edgeId>`, cross-instance state fanout uses `state.<instanceId>.set`.
- Envelope remains the same as master spec; dev mode currently does not enforce bearer tokens.

## KV layout (per instance)
- `graph`: active graph snapshot (use KV revision as the etag/version).
- Optional history/patches if needed (`graph/patches/<rev>`), but runtime operates on the snapshot.
- State and operator-specific keys remain under the same bucket (e.g., `state/...`, `operators/...`).

## Hot update rules
- Only one graph active; apply diffs atomically with dependent changes (e.g., removing a state also removes dependent fanIn/fanOut/edges).
- Versioned updates: caller includes expected version; engine rejects mismatched writes to prevent stale overwrites.
- Safe point application: queue changes and apply between ticks; no full restart unless a fatal validation error forces rebuild (rare fallback).

## Cross-instance wiring (half-edges)
- Compiler/splitter generates half-edge configs per subgraph: state fanout (source knows subscribers) and data bus subjects per edge.
- Engine reads delivered config, sets up pub/sub accordingly; does not need full global topology knowledge.
- Operator-to-external links: when an operator links to an external node (even in another engine), the compiler promotes the link through container boundaries by adding synthetic container ports and mapping to the correct half-edges/subjects. Runtime still scopes operators to their engine; promotion is a compile-time convenience.
