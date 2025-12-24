# Graph Compiler Outline (master side)

Goal: deterministically turn the editor graph into deployable per-instance subgraphs and half-edges.

Inputs:
- Full graph (nodes, edges), with nodes referencing services/operators (inPorts/outPorts, commands, state).
- Edge metadata per `schemas/edge.schema.json` (kind, scope, strategy, queue, timeout).
- Instance placement (which service/engine hosts which nodes); auto-promotion rules for operator->external links.

Steps:
1) Validate graph:
   - Port compatibility (data/state rules), required fields, allow* flags.
   - Edge kind/scope resolution (intra vs cross).
2) Assign edgeIds/subjects:
   - Data/State (cross): `f8.bus.<edgeId>` (edgeId is unique; can represent data or state fanout).
   - Intra edges can stay in-memory identifiers.
   - Defaults: cross-instance data edges default to `latest` strategy, queueSize=64, `drop-old` on; timeout optional. JetStream/bridge + DLQ only when cross-instance reliability is required.
   - Data edges can connect service ports to operator ports (and vice versa) across instances; all map to `f8.bus.<edgeId>`.
3) Auto-promotion:
   - When an operator links to an external node, synthesize container ports across boundaries (nested supported) and map the link to half-edges.
4) Split into subgraphs:
   - Per instance: nodes it hosts, intra edges, inbound/outbound half-edges with subjects, strategies, queue/timeout.
   - Include the operator catalog reference for engines.
5) Emit artifacts:
   - Master stores the full compiled graph as a single blob (default) with a new revision/etag to avoid partial updates; if multiple keys are used, include a commit record.
   - Per-instance payload sent to engines/services via control plane (NATS) and persisted to their KV buckets.
   - Operator catalog location for engines: write to the instance bucket at `catalog/operators` (JSON array) for editor discovery.
- Status path convention: engines/services should expose status in `status/summary` in their bucket to align with editor consumption.
6) Rollout:
   - Master sends apply to each target via the shared service control surface (`deployOpGraph`); engines apply at tick boundary and confirm with new revision.
   - On failure or master unreachability, engines keep the old graph running and mark apply endpoints read-only; master can retry/rollback when available.

Open items:
- Exact subject naming for state snapshots/health/status.
- How to batch multi-key updates (commit record vs single blob).
- Operator catalog fetch: engine writes supported operators to KV under a known key for the editor/master to consume.
