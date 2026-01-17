NATS KV Schema (v1)

This is the first-version cross-language KV layout for f8studio.

## Configuration

- `F8_NATS_URL`: NATS server URL (enables NATS mode in editor).
- `F8_NATS_BUCKET`: JetStream KV bucket name (default: `svc_<serviceId>`).
- `F8_SERVICE_ID`: service instance id (eg. an engine block node id). If omitted, a random id is used.
- `F8_ACTOR_ID`: unique actor id for a client (default: random per launch).
- `F8_OPERATOR_RUNTIME_MODULES`: comma-separated python modules imported by engine services to register operator implementations (optional).

## Keys

All keys are scoped by bucket (one bucket per service id):

- Rungraph snapshot (editor-owned, engine reads):
  - `rungraph` -> JSON bytes (OperatorGraph.to_dict payload)

- State KV entries (shared; engines and editor can write depending on access):
  - `nodes.<nodeId>.state.<field>` -> JSON bytes:
    - `value`: any JSON-serializable
    - `source`: `"editor" | "engine" | "rungraph"`
    - `actor`: client id string (used to ignore self-echo)
    - `ts`: ms timestamp (int)

## Cross-Service Edges

Cross-service edges are stored as normal edges with explicit endpoints:
- `fromServiceId/fromOperatorId/fromPort`
- `toServiceId/toOperatorId/toPort`

Cross-service is determined by `fromServiceId != toServiceId`.

## Subjects (Core NATS Pub/Sub)

Service-to-service data ports use core NATS pub/sub (no JetStream).

Cross-instance data is routed by **producer output port**, so fan-out publishes once
per port and multiple receivers can subscribe independently:

- Producer publishes:
  - `svc.<fromServiceId>.nodes.<fromNodeId>.data.<portId>` -> JSON bytes `{ "value": ..., "ts": 0 }`

- Receivers derive the subject from `fromServiceId/fromOperatorId/fromPort`.

- Request last value (optional):
  - Request: `svc.<serviceId>.data.<port>.get`
  - Reply: same JSON payload as publish, or empty if no value yet.

## Notes

- `serviceId` and `nodeId` are used as single NATS key tokens and must not contain `.`.
- In this repo, service/operator nodes override NodeGraphQt's default `0x...` ids and use `uuid4().hex` so ids are stable and cross-process friendly.
- NATS KV is last-write-wins by revision. For future collaborative state editing, this can evolve into:
  - CRDT per key, or
  - server-side merge rules with per-field clocks.
