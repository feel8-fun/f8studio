NATS KV Schema (v1)

This is the first-version cross-language KV layout for f8studio.

## Configuration

- `F8_NATS_URL`: NATS server URL (enables NATS mode in editor).
- `F8_NATS_BUCKET`: JetStream KV bucket name (default: `svc_<serviceId>`).
- `F8_SERVICE_ID`: service instance id (eg. an engine block node id). If omitted, a random id is used.
- `F8_ACTOR_ID`: unique actor id for a client (default: random per launch).
- `F8_OPERATOR_RUNTIME_MODULES`: comma-separated python modules imported by engine services to register operator implementations (optional).

## Keys

All keys are scoped by service id:

- Topology snapshot (editor-owned, engine reads):
  - `svc.<serviceId>.topology` -> JSON bytes (OperatorGraph.to_dict payload)

- State KV entries (shared; engines and editor can write depending on access):
  - `svc.<serviceId>.nodes.<nodeId>.state.<field>` -> JSON bytes:
    - `value`: any JSON-serializable
    - `source`: `"editor" | "engine" | "topology"`
    - `actor`: client id string (used to ignore self-echo)
    - `ts`: ms timestamp (int)

## Cross-Instance Edges (Half-Edges)

Cross-instance edges are stored as two "half-edges", one in the source service topology
and one in the target service topology. This avoids the need for a separate relay process.

- Stored inside `svc.<serviceId>.topology` (per-service OperatorGraph snapshot).
- Each half-edge is a normal `F8EdgeSpec` with extra fields:
  - `scope="cross"`
  - `edgeId`: stable id shared by both halves
  - `direction`: `"out"` (local is source/from) or `"in"` (local is target/to)
  - `peerServiceId`: remote service id (remote node/port can be derived from `from/to` + `direction`)
  - `strategy/queueSize/timeoutMs`: used by the receiver for rate mismatch handling (data)

## Subjects (Core NATS Pub/Sub)

Service-to-service data ports use core NATS pub/sub (no JetStream).

Cross-instance data is routed by **producer output port**, so fan-out publishes once
per port and multiple receivers can subscribe independently:

- Producer publishes:
  - `svc.<fromServiceId>.nodes.<fromNodeId>.data.<portId>` -> JSON bytes `{ "value": ..., "ts": 0 }`

- Cross-instance half-edges include `subject` with the same value so receivers know what to subscribe to.

- Request last value (optional):
  - Request: `svc.<serviceId>.data.<port>.get`
  - Reply: same JSON payload as publish, or empty if no value yet.

## Notes

- `serviceId` and `nodeId` are used as single NATS key tokens and must not contain `.`.
- In this repo, service/operator nodes override NodeGraphQt's default `0x...` ids and use `uuid4().hex` so ids are stable and cross-process friendly.
- NATS KV is last-write-wins by revision. For future collaborative state editing, this can evolve into:
  - CRDT per key, or
  - server-side merge rules with per-field clocks.
