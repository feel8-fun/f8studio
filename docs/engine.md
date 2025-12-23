# Engine Service (runtime graph host)

- Runs a single active graph per instance; manages state, executes nodes/operators, and handles cross-instance wiring (state/data buses).
- Applies to both native app engines and web-worker engines; same runtime model, different host/runtime env.
- Built on the minimal ServiceHostBase (state/KV/heartbeat/register) plus engine-specific data bus pub/sub and graph runtime.

## Responsibilities
- Load graph snapshot from KV (`graph`) at startup; initialize state from seeded keys.
- Apply graph diffs/commands at safe points (end of tick), validate, update in-memory graph, bump version (KV revision), and write the new snapshot back to KV. On validation failure, keep the old graph running. If master/control plane is unavailable, continue running the last good graph and reject new apply requests with a clear read-only error.
- Maintain per-instance state manager; publish `f8.state.<instanceId>.delta` locally; fan out cross-instance state via `f8.state.<instanceId>.set` for subscribed remote nodes.
- Data bus wiring (engine-only): subscribe/publish to compiled cross-edge subjects (`f8.bus.<edgeId>`), with backpressure/queueing policies as needed.
- Lifecycle with master: register -> ensure per-instance KV bucket exists -> run -> unregister -> clean up bucket (if owned by master).

## Tick-based scheduling
- Engine runs on discrete ticks to align processing of mixed-rate realtime data; all node executions within a tick see a consistent snapshot of inputs/state.
- End-of-tick is the safe point: apply queued graph diffs, commit state transitions, emit deltas/fanout, and advance clocks. This avoids mid-tick structural changes and keeps multi-rate streams coherent.
- Ticks can be fixed-step or adaptive; external triggers (e.g., high-rate streams) enqueue work but effects become visible on the next tick boundary.

## Web-engine UI hooks (visualization/control)
- Web engine shares the same runtime model, plus UI renderers per node.
- Data printer node: subscribes to its data bus input; on tick, renders the latest payload into the node UI (React component), no extra side effects.
- Input control node: renders a UI control (e.g., slider); user interactions update the node state; state changes fan out via the normal state bus (`f8.state.<instanceId>.set`) to downstream nodes (web or native services).
- UI is purely view/control; execution graph semantics stay the same. Data/State flow uses the existing buses; DOM work lives inside the renderer.

## Inputs/outputs (NATS-facing)
- Command subjects follow `f8.<serviceSlug>.<instanceId>.<verb>` (e.g., apply graph diff, pause/resume, state snapshots). Cross-instance data uses `f8.bus.<edgeId>`, cross-instance state fanout uses `f8.state.<instanceId>.set`.
- Envelope remains the same as master spec; dev mode currently does not enforce bearer tokens.
- Engine is primarily a NATS microservice; if HTTP is needed, use a single gateway that proxies HTTP->NATS (no per-engine HTTP port).

## Service manifest
- Provide an engine `service.json` (schemaVersion `f8service/1`) describing commands (pause/resume/stop/applyGraph, etc.), status, and configuration fields (e.g., tick rate). Avoid putting high-churn metrics like uptime in persisted state; publish them via status instead. Set `allowAdd*` as appropriate.

## Rate-mismatch handling (cross-instance data edges)
- Each cross-instance data edge carries a consume strategy to handle producer/consumer rate differences (configured at link compile time and exposed in the editor):
  - High->low fps (producer faster): `latest` (default; take newest, drop older), `average/window` for numeric types.
  - Low->high fps (producer slower): `hold`/`repeat` last, or `interpolate` for numeric types.
  - Safety defaults: bounded queues per edge (default queueSize = 64), `drop-old` on for cross-instance edges; optional timeout to evict stale payloads. Enable JetStream/bridge + DLQ only when cross-instance reliability is required.
- Engines apply the strategy at tick boundaries: consume the edge's queue per tick according to the configured policy, so topology order stays intact and multi-rate streams stay coherent.

## Editor edge compatibility rules (data/state)
- Enforced at edit time in the editor (and can be rechecked at runtime for safety):
  - `any` type on either end = compatible.
  - State bus: types must match exactly (aside from `any`), ranges are not validated on connect.
  - Data bus: consumer schema must be satisfied by producer schema (producer supplies all required consumer fields with compatible types). If producer lacks required consumer fields, block the connection.

## Control plane (master <-> engine) - to define and implement
- APIs (extend `api/specs/services/engine.yaml`): graph apply/snapshot (existing), plus start/stop/pause/resume, status/health, operator catalog fetch, and error reporting.
- Handshake/rollout: master sends subgraph + half-edge config, engine applies at safe point and replies with revision; failures report errors without disrupting current graph.
- Health: write status into the instance KV bucket (e.g., `status/summary`) and rely on KV watches; master can probe if needed to mark node status for the editor. Suggested fields: `status` (`starting|running|paused|stopped|error`), `lastError`, `graphEtag` (KV revision/etag), `updatedAt` (timestamp).
- Consistency: use KV revision as etag; engines reject stale updates. Default to single-key graph blob + etag; if multiple keys are ever used, group them behind a commit record. When master is unreachable, engines stay on the last applied graph and mark apply endpoints read-only.
- Operator catalog: engine should publish its supported operators into its KV bucket under a known key (e.g., `catalog/operators`) for editor/master discovery, in addition to the NATS catalog endpoint.
- Health feed subjects: optional; KV watch is primary. If used, keep the payload small (status/lastError) and omit high-churn metrics.

## KV layout (per instance)
- `graph`: active graph snapshot (use KV revision as the etag/version).
- Optional history/patches if needed (`graph/patches/<rev>`); runtime operates on the snapshot. Default is a single KV key with etag-based writes; add commit records later only if audit/rollback is required.
- State and operator-specific keys remain under the same bucket (e.g., `state/...`, `operators/...`).

## Hot update rules
- Only one graph active; apply diffs atomically with dependent changes (e.g., removing a state also removes dependent fanIn/fanOut/edges).
- Versioned updates: caller includes expected version; engine rejects mismatched writes to prevent stale overwrites.
- Safe point application: queue changes and apply between ticks; no full restart unless a fatal validation error forces rebuild (rare fallback).

## Native plugin governance (engine-side)
- Form factor: native DLLs managed by the engine plugin manager; JS plugins deferred.
- Entry point: export `extern "C"` registration (e.g., `bool f8_plugin_register(f8_registry* reg)`); no exceptions across DLL boundaries.
- Registry contents: plugin meta (name, version, description, compatible engine ABI/version), entry points for create/destroy, and declared capabilities (operators/node types provided).
- ABI safety: use C function pointers; allocations and frees must be on the same side. Reject load if ABI version mismatches or the plugin is not on the allowlist.
- Isolation: no sandbox for now; optional isolation via separate thread/process when stability is a concern. Logging/errors return status codes, not exceptions.

## Cross-instance wiring (half-edges)
- Compiler/splitter generates half-edge configs per subgraph: state fanout (source knows subscribers) and data bus subjects per edge.
- Engine reads delivered config, sets up pub/sub accordingly; does not need full global topology knowledge.
- Operator-to-external links: when an operator links to an external node (even in another engine), the compiler promotes the link through container boundaries by adding synthetic container ports and mapping to the correct half-edges/subjects. Runtime still scopes operators to their engine; promotion is a compile-time convenience.
