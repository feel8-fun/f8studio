# NATS Binding Conventions

- Transport: NATS request/reply is canonical. An optional HTTP gateway can expose the same OpenAPI contract for local testing/debugging (Swagger) by proxying requests into NATS.
- Each operation includes `x-nats-subject`, optional `x-nats-queue`, `x-nats-timeout-ms` (no fixed reply subject—clients use per-request inboxes).
- Envelope (all requests/replies share this wrapper):
  ```json
  {
    "msgId": "uuid",             // per logical request; reused on retries for idempotency/dedup
    "traceId": "uuid",           // end-to-end correlation across calls
    "clientId": "string",        // Web client instance/session; daemon enforces single client
    "hop": 0,
    "ts": "RFC3339",
    "apiVersion": "v1",
    "payload": {},
    "headers": { "k": "v" }      // optional
  }
  ```
- Reply adds:
  ```json
  { "status": "ok|error", "errorCode": "string", "errorMessage": "string" }
  ```
- Delivery: NATS core is at-most-once. Clients set timeouts and retry with backoff on `timeout|unavailable`, reusing `msgId` for safe dedup. Treat other errors as terminal.
- Error model: keep a small fixed set (`invalid`, `conflict`, `unauthorized`, `not-found`, `unavailable`, `timeout`, `internal`); include `errorCode` and human-readable `errorMessage`.
- Subjects: prefer `f8.<serviceSlug>.<instanceId>.<verb>` for instance-directed commands (no load-balancing); `$SRV.*` for service discovery/health; `f8.state.<instanceId>.delta` for per-instance state broadcasts. `serviceSlug` must match `[a-z0-9_-]+`.
- Roles: Web UI is the client (singleton per daemon, enforced at daemon level—not per service profile). A web-worker or micro engine can publish/subscribe under the same `clientId` for lightweight tasks (databus, fanout to visualization), but is not treated as a durable service. Operator catalogs are published as state by the engine at runtime, not part of static manifests.
- Versioning: semantic (`v1`); do not break existing fields in-place—add new fields or new subjects.

## Authn/Authz
- Long-term plan: registration issues a service-scoped bearer token and `instanceId`; instance-scoped calls (applyConfig, unregister, state updates) present the token in `Envelope.headers.authorization` (`Bearer <token>`) or HTTP `Authorization`. Unregister must use the same token; KV buckets per-instance; tokens short-lived/rotatable; gateway forwards `Authorization`.
- Current dev mode: no bearer enforcement; all callers are allowed. Token fields are reserved for later hardening to reduce upfront friction. When enabling auth, wire the validation in master and adapters and tighten KV ACLs.

## Handler layering (NATS + HTTP)
- Keep business logic transport-agnostic: handlers accept `(context, envelope)` and return envelope replies.
- Adapters:
  - NATS adapter: decode envelope from NATS request, call handler, encode reply to NATS.
  - HTTP adapter/gateway: map HTTP headers/body to envelope (including `authorization`, `trace-id`, `client-id`), call the same handler, return JSON.
- OpenAPI remains single-source; include `x-nats-*` hints plus an HTTP server entry for the gateway.

## Graph distribution and cross-instance wiring
- Template vs instance: operator templates live in repo files (e.g., `services/<service>/operators.json`); running instances keep their own dynamic state/ports/scripts in per-instance KV snapshots. Frontend publishes graph diffs via commands; services apply at safe points and write back the new graph/version.
- Cross-instance edges are compiled into half-edges: each end gets its own config; no instance needs full global topology.
  - State bus: source instance knows its stateOut keys and which remote nodes subscribe; on state change it emits `state.<instanceId>.set` (k,v,rev) to its fanout subject. Targets subscribe and apply to their stateIn.
  - Data bus: compiler allocates a unique subject per cross-edge (e.g., `f8.bus.<edgeId>`); source publishes to it, targets subscribe.
- ServiceHostBase provides state fanout and data bus plumbing based on declared ports. Services that expose `dataInPorts`/`dataOutPorts` (or operator-level ports) get the wiring; services that do not declare ports incur no extra overhead. Subjects and permissions come from the compiled half-edge config delivered with each subgraph.
- Operator-to-external links: editor can draw links directly from operator ports; the compiler auto-promotes these across container boundaries by synthesizing container ports and half-edges, so runtime still deploys per-instance wiring without exposing internal nodes globally.

## Runtime graph model (hot update)
- Single graph per engine: runtime only holds one active graph per instance; stored as a KV snapshot (e.g., `graph`, using the KV revision as the version/etag) in the instance bucket.
- Update flow: frontend submits a graph diff/command; engine validates and queues it; applies at tick/safe-point boundaries; then writes the new snapshot back to KV and (optionally) broadcasts a version/state delta.
- Concurrency: include `version`/`etag` in updates to avoid stale overwrites; reject on mismatch and require the caller to refresh and reapply.
- Consistency: structural changes (add/remove nodes/ports/edges) are bundled with dependent changes (e.g., remove state also deletes dependent fanIn/fanOut) before commit, so applied atomically.
- Failure: if a patch fails validation, keep the old graph running and return an error; only fall back to rebuild/restart as a last resort.
