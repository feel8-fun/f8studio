# Service Role

Contract for any runtime service process; engine is a specialized service that hosts an OperationGraph.

## Base contract (ServiceHostBase)
- Provide registration with master, heartbeat/ping handling, per-instance state manager (KV), and command dispatch using the shared envelope.
- Own `instanceId` (confirmed by master) and identify its service type via the manifest (not in subjects).

## Startup and registration
- Load `service.json` (schema `f8service/1`) from the service package.
- Connect to NATS and call `f8.master.registry.register` with the manifest and optional hash/bootstrap info.
- On `ok`, receive/confirm `instanceId`; master ensures KV bucket `kv_f8_<instanceId>` exists.
- Initialize state from KV, publish an initial `status/summary`, and enter `deactivated` until activation.

## Command surface
- Listen on predefined verbs `f8.<instanceId>.<verb>`:
  - Lifecycle: `activate`, `deactivate`, `terminate` (clean unregister), `status`.
  - Graph apply: `deployOpGraph` for the compiled payload from master.
- Listen on the command channel `f8.<instanceId>.cmd` for service-specific commands declared in the manifest (`{command, params}`).
- Replies include `status=ok|error`, `errorCode`, `errorMessage` per the binding conventions.

## State, KV, and telemetry
- Per-instance bucket holds `graph`, `state/*`, `status/summary`, optional history, and service-specific keys.
- Cross-instance consumers read the latest state via KV (bucket + key) instead of a dedicated fanout subject.
- Keep high-churn metrics out of KV; short status payloads are acceptable.

## Graph and wiring
- Apply subgraph payloads from master: local nodes, intra edges, and half-edge config for cross links.
- For each outbound half-edge, publish to `f8.bus.<edgeId>`; for each inbound half-edge, subscribe and consume the latest payload (no per-link queue/backpressure config).
- Respect type/port compatibility and rate-mismatch strategies supplied by the compiler.
- Engines additionally host an OperationGraph, apply diffs at tick boundaries, maintain version/etag from KV, and publish operator catalogs to `catalog/operators`.

## Lifecycle and shutdown
- State progression: `starting` -> `deactivated` -> `activated` <-> `deactivated` -> `terminating` | `error`.
- On shutdown or fatal error, stop the state manager, send `f8.master.registry.unregister`, and allow master to remove the KV bucket if master created it.
