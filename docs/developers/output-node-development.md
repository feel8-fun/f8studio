# Output Node Development

Output nodes are the device boundary for F8Studio. They translate graph values
into one concrete protocol and own the resulting connection lifecycle.

## Ownership Boundary

An output node owns:

- device discovery required by its protocol
- connection, reconnect, and disconnect behavior
- protocol encoding and transport calls
- final send throttling and protocol-specific value quantization
- low-frequency connection status
- actionable, deduplicated error reporting

Web Studio renders the node's declared specification. It must not import device
SDKs or maintain a global device registry. Engine services host and schedule the
node, but they do not own its protocol connection.

If the node is not present in the deployed graph, its protocol must not scan,
connect, or consume device resources.

## Signal Inputs

Prefer explicit numeric ports for continuous controls. New output nodes should
use normalized values where the protocol permits it:

| Port | Meaning | Preferred range |
| --- | --- | --- |
| `position` | Linear or positional target | `0.0..1.0` |
| `vibrate` | Vibration intensity | `0.0..1.0` |
| `rotate` | Rotation intensity | `0.0..1.0` |
| `pump` | Pump or contraction intensity | `0.0..1.0` |
| `durationMs` | Optional command duration | `>= 0` |

Keep protocol-specific features explicit. Do not hide unrelated channels in an
untyped command dictionary. Existing port names remain supported; alignment of
an established port requires an explicit versioned migration.

An encoder and transport should be separate nodes only when the encoded value
is useful for visualization, storage, or another transport. `TCode -> Serial
Out` is the canonical example. SDK messages that only exist for one connected
device should remain internal to that output node.

## State And Telemetry

Use state for configuration and low-frequency semantic status:

- `enabled`
- connection URL, port, device selection, and protocol options
- `connected` when the protocol has a persistent connection
- `availableDevices` when the protocol supports discovery

Secrets and local connection identifiers must use `redactOnPublish`.

Per-command results, sent positions, latency, request counts, dropped updates,
and repeated failures belong on data or monitor channels. They must not be
published as high-frequency state.

## Lifecycle And Safety

An output node must:

1. validate and clamp protocol-bound values
2. enforce its final send interval when required
3. stop sending while disabled or inactive
4. release connections and background tasks when closed
5. report failures with context and traceback at the node boundary
6. suppress repeated identical high-frequency errors

Components that contain a physical output node must save it disabled. A user
must configure and explicitly enable physical output after inspecting the
signal path.

## Extension Checklist

A new protocol integration should require only:

1. an operator specification and runtime node
2. focused runtime tests with a fake or mock transport
3. generated describe metadata
4. optional tagged output components
5. operator documentation

Adding a protocol must not require changes to Web Studio, ServiceBus, Zenoh, or a
global engine device manager.
