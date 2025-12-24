# Naming and subjects

- **instanceId**: runtime-generated short id (base32/58) per daemon instance; used for all instance-scoped subjects and KV buckets.

## Subjects
- Instance-directed commands: `f8.<instanceId>.<verb>`
  - Example: `f8.ab12cd.play`
  - No load-balancing; each instance gets its own subject.
- Discovery/health: `$SRV.PING|INFO|STATS.<instanceId?>`
- Cross-instance data bus: `f8.bus.<edgeId>` (unique per compiled cross-edge; pub/sub)

Rules: tokens cannot contain `. * >` or whitespace; keep ASCII and short.

## KV buckets
- Per-instance bucket: `kv_f8_<instanceId>` (no dots; matches `[A-Za-z0-9_-]+`).
- Keys may contain dots for hierarchy: `state/system.power`, `operators/<operatorType>`, `graph/<graphId>`.

### KV lifecycle (master-owned)
- Master-started service: before launching the process, master creates the per-instance KV bucket and seeds any known config/state with `rw` and `init` attrs from the provided config.
- Service-initiated register: on successful register, master ensures the bucket exists (idempotent) but does not seed values (since it may already exist).
- Unregister: after the service stops its state manager, master deletes the per-instance bucket if present.

## Examples
- `instanceId`: `ab12cd`
- Command subject: `f8.ab12cd.play`
- KV bucket: `kv_f8_ab12cd`
