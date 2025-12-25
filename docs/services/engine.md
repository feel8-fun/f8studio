# Engine Service (runtime graph host)

- Runs a single active graph per instance; manages state, executes nodes/operators. Cross-instance wiring (state/data buses) comes from the shared ServiceHostBase used by all services when ports are declared.
- Applies to both native app engines and web-worker engines; same runtime model, different host/runtime env.
- Built on ServiceHostBase (state/KV/heartbeat/register + cross-instance wiring) plus the engine-specific operator graph runtime.

## Responsibilities
- Load graph snapshot from KV (`graph`) at startup; initialize state from seeded keys.
- Apply graph diffs/commands (delivered via `deployOpGraph`) at safe points (end of tick), validate, update in-memory graph, bump version (KV revision), and write the new snapshot back to KV. On validation failure, keep the old graph running. If master/control plane is unavailable, continue running the last good graph and reject new apply requests with a clear read-only error.
- Maintain per-instance state manager; write state into the instance KV; cross-instance readers consume the latest value via KV watch (bucket + key).
- Data bus wiring (when ports are declared): subscribe/publish to compiled cross-edge subjects (`f8.bus.<edgeId>`). Engine uses this for operator data ports (`dataInPorts`/`dataOutPorts`); other services use the same pattern if they expose data ports.
- Lifecycle with master: register -> ensure per-instance KV bucket exists -> run -> unregister -> clean up bucket (if owned by master).

## Tick-based scheduling
- Engine runs on discrete ticks to align processing of mixed-rate realtime data; all node executions within a tick see a consistent snapshot of inputs/state.
- End-of-tick is the safe point: apply queued graph diffs, commit state transitions, emit deltas/fanout, and advance clocks. This avoids mid-tick structural changes and keeps multi-rate streams coherent.
- Ticks can be fixed-step or adaptive; external triggers (e.g., high-rate streams) enqueue work but effects become visible on the next tick boundary.

## Execution flow (operators)
- Exec pins: operators may declare `execInPorts` and `execOutPorts` (they can be empty for entry/terminal/pull-only nodes). The compiler introduces `kind=exec` edges between exec pins; each execIn pin accepts only one incoming exec link (expose multiple execIn pins for multiple triggers).
- Runtime: an operator runs in a tick only if it receives an exec token and its data deps are satisfied (pull-based evaluation fetches upstream data and caches outputs within the tick). Data-only nodes can still be pulled by downstream operators even if they are not on the exec chain.
- Branching: operator logic decides which execOut(s) to fire. For a bool branch, the operator reads state/input and triggers the true or false execOut accordingly. Switch-like behavior is encoded by multiple execOutPorts and operator logic selecting one (or default).
- Fan-in: operators with multiple execInPorts can be triggered by any incoming exec edge (aggregate).

## Web-engine UI hooks (visualization/control)
- Web engine shares the same runtime model, plus UI renderers per node.
- Data printer node: subscribes to its data bus input; on tick, renders the latest payload into the node UI (React component), no extra side effects.
- Input control node: renders a UI control (e.g., slider); user interactions update the node state; state changes write through the state manager/KV so downstream nodes watching the bucket/key observe updates.
- UI is purely view/control; execution graph semantics stay the same. Data/State flow uses the existing buses; DOM work lives inside the renderer.

## Inputs/outputs (NATS-facing)
- Control plane (predefined verbs): `f8.<instanceId>.<verb>` for `deployOpGraph`, `activate`, `deactivate`, `terminate`, `status`, and optional `stateSnapshot`/`setState`. Service-defined commands use `f8.<instanceId>.cmd` with `{command, params, traceId?}`.
- Data/State bus: cross-instance data uses `f8.bus.<edgeId>` (service ports to operator ports, operator ports to service ports, or service-to-service). Control/state reads/writes still go through the control verbs above, and state propagation across instances is via KV watch on the configured bucket/key.
- Envelope remains the same as master spec; dev mode currently does not enforce bearer tokens.
- Engine is primarily a NATS microservice; if HTTP is needed, use a single gateway that proxies HTTP->NATS (no per-engine HTTP port).

## Service manifest
- Provide an engine `service.json` (schemaVersion `f8service/1`) describing commands (pause/resume/stop/deployOpGraph, etc.), status, and configuration fields (e.g., tick rate). Avoid putting high-churn metrics like uptime in persisted state; publish them via status instead. Set `allowAdd*` as appropriate.

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

## Control plane (master <-> engine)
- Graph apply/snapshot: master calls `f8.<instanceId>.deployOpGraph` with the compiled subgraph/half-edges; engine applies at a safe point and replies with revision/etag. Snapshots can be exposed via `f8.<instanceId>.status` or a dedicated `stateSnapshot`.
- Lifecycle: `activate`/`deactivate`/`terminate` verbs on `f8.<instanceId>.<verb>`.
- Health: write status into the instance KV bucket at `status/summary` and rely on KV watches; master can also probe with `$SRV.PING|INFO|STATS.<instanceId>` to mark node status for the editor. Suggested field: `status` (common lifecycle: `starting|deactivated|activated|terminating|error`). Keep any other summary fields consistent with the shared service status conventions.
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
