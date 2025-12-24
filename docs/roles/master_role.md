# Master Role

Singleton orchestrator on the host; owns the service catalog, instance lifecycle, and graph deployment for the local daemon.

## Shared terms
- `instanceId`: per-process id (short base32/58) assigned or confirmed by master; used for subjects and KV buckets.
- `ServiceGraph`: user-authored topology of service nodes and edges delivered by the web client.
- `OperationGraph`: engine-only operator graph that runs inside an engine service instance.
- `half-edge`: compiled cross-instance link; carries subject/strategy for one side of a cross data edge (`f8.bus.<edgeId>`) or bucket/key mapping for cross-instance state via KV watch.

## Responsibilities
- Scan configured “available services” folders for `service.json` (schema `f8service/1`), validate, and publish the catalog of service types (KV + registry) for the web client.
- Keep the registry of running instances: service class (from manifest), `instanceId`, pid/process info, KV bucket, status, manifest hash. Enforce uniqueness and clean up on unregister.
- Provide the control plane to the single web client session (`clientId`), including ping/capabilities, catalog fetch, deploy, and status queries.
- Manage process lifecycle for instances requested by the ServiceGraph: start missing processes from their manifest `launch` command, monitor health, and stop instances that are removed.
- Create/seed per-instance KV buckets (`kv_f8_<instanceId>`) before launch when master owns the process.
- Enforce registry admission: incoming registers must include both `serviceClass` (service type) and `instanceId`, otherwise reject. Accepted instances are recorded in the instance registry KV bucket managed by master.
- Watch the `services` folder for manifests, validate them, and publish the service catalog into the service registry KV bucket managed by master.

## Graph deployment (ServiceGraph -> subgraphs)
1) Receive ServiceGraph from the web client after a topology change (via `deployGraph`).
2) Validate nodes/ports against the available-service catalog; resolve placements/instanceIds.
3) Ensure required instances exist: create KV bucket, seed config/state, start the process if not running, and wait for register.
4) Compile the graph: split into per-instance subgraphs; engines get an OperationGraph; other services get the subset relevant to them.
5) Assign edgeIds for every cross-instance data edge: emit half-edge configs for `f8.bus.<edgeId>` subjects (pub/sub). State sharing relies on KV buckets (bucket + key watch) rather than state fanout subjects.
6) Deliver subgraphs and half-edges to each instance via their command surface (e.g., `cmd.deployOpGraph`), track revisions/etags, and persist the compiled graph snapshot in KV.
7) On failure, keep prior graphs running; retry or roll back without interrupting healthy instances.

## Control plane and subjects
- Web handshake: `f8.master.ping {clientId, requestedProfile}`; master holds a single-client lease with TTL refreshed by pings; reject or evict a second client unless the lease has expired.
- Registry RPCs: `f8.master.registry.register|unregister` for services; master assigns `instanceId` when omitted (registry also stores the service type).
- Predefined instance commands (activate, deactivate, deployOpGraph, etc.): `f8.<instanceId>.<verb>` (no load balancing).
- Service-defined commands: `f8.<instanceId>.cmd` where the payload carries `{command, params}`; reply mirrors the request envelope with status/result.
- Health: periodically probe `$SRV.PING|INFO|STATS.<instanceId>` and reconcile responders against the instance registry KV; mark missing responders as degraded/unresponsive. Instances also write concise status into their KV bucket at the key `status/summary` for the web client to watch.

## Failure/recovery expectations
- If master is down, running services keep their last applied subgraph; deploy is disabled until master returns.
- On restart, master rebuilds the registry (catalog scan + service re-register), revalidates KV buckets, and can redeploy the last ServiceGraph when the web client reconnects.

## Master-managed KV layout (examples)
- `kv_f8_master_instances`: per-instance entries keyed by `<instanceId>` (no prefix) containing `{ serviceClass, pid?, status, kvBucket, manifestHash, updatedAt }`; optional health fields like last ping/fail count can live alongside.
- `kv_f8_master_services`: service catalog entries keyed by `<serviceClass>` (no extra id), value is the validated manifest or a summary `{ serviceClass, version, label, tags, launch, manifestHash }`.
- Per-instance buckets `kv_f8_<instanceId>` (created/owned by master when it launches the process): `status/summary`, `graph`, `state/*`, and engine-specific keys like `catalog/operators`.
