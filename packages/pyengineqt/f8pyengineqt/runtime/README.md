ServiceRuntime SDK (v1)

This module is the first version of the runtime-side ServiceSDK:

- one process -> one shared NATS connection (core pub/sub + JetStream KV)
- watch per-service rungraph from KV: `rungraph` inside bucket `svc_<serviceId>`
- route intra data edges in-process
- route cross data edges by subscribing to producer subjects:
  - `svc.<fromServiceId>.nodes.<fromNodeId>.data.<portId>`
- watch local state KV and deliver updates to nodes (`on_state`)
- bind cross-state edges by watching remote KV using `peerServiceId` and mirroring into local KV
- cross-data strategies (v1):
  - all strategies are pull-based (SDK buffers on receive; consumer pulls when needed)
  - `latest`: newest sample since last pull
  - `hold/repeat`: newest if available else last pulled value
  - `average`: average buffered numeric samples since last pull
  - `interpolate`: interpolate between prev/newest at pull time
  - `timeoutMs`: staleness check at pull time

Entry points:

- `f8pyengineqt.runtime.service_runtime.ServiceRuntime`
- `f8pyengineqt.runtime.service_runtime_node.ServiceRuntimeNode`
- `f8pyengineqt.runtime.nats_transport.NatsTransport`

Demo (requires running NATS server):

- `.\.venv\Scripts\python.exe scripts\service_runtime_demo.py`
