# Web Client Role

Singleton browser UI that edits graphs and drives the local daemon through master.

## Session and transport
- Establish a single session with master using `f8.master.ping` and a `clientId`; master holds a lease with TTL, refreshed by pings, and rejects a second client until the lease expires or is evicted.
- Transport is NATS request/reply with the shared envelope (`msgId`, `traceId`, `clientId`, `payload`, `headers`, etc.). An HTTP gateway is only a debug proxy into the same NATS subjects.

## Responsibilities
- Discover catalog and instances from master: fetch available service types (from the catalog KV/registry) and the running-instance registry, including status and KV bucket ids.
- Author and version ServiceGraphs: edit locally, diff, and submit deploys (`deployGraph`) to master whenever topology changes.
- Deployment call (`deployGraph`) sends the full ServiceGraph (and any per-node config) to master; master handles compile/start/apply and returns the revision/etag.
- After deploy, drive instances by sending predefined commands to `f8.<instanceId>.<verb>` (e.g., `activate`, `deactivate`) and service-defined commands via `f8.<instanceId>.cmd` with `{command, params}`. Graph deployment is always through master (`deployGraph`), not direct to services.
- State control rules:
  - Before an instance is running, its state keys are unavailable.
  - Once the instance is up but not activated, web client can set writable keys (`init`, `rw`, `wo`).
  - After activation, `init` keys are locked; only `rw`/`wo` keys can be changed.
- Observe state/status via KV watches (`status/summary`, `graph`, `state/*`), using bucket+key from compiled wiring when state must be read across instances. Optional subject: `f8.bus.<edgeId>` when the UI needs to visualize data on a compiled cross edge.

## Offline/degraded flow
- When master times out, switch to degraded mode: keep local editing, disable deploy/daemon-backed actions, surface a warning.
- On reconnection: redo `f8.master.ping`, fetch current snapshots/etags from KV, reconcile drafts, then resume deploy and control.
- Use timeouts/backoff and reuse `msgId` on retries per the envelope rules.

## Engine-specific notes
- Engine nodes in the ServiceGraph produce an OperationGraph; deploy still flows through master.
- Operator catalogs are discovered from the engine's KV (`catalog/operators`) after the engine registers; the UI updates palettes from that data rather than static manifests.
