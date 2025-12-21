# Naming and subjects

- **serviceSlug**: subject-safe token `[a-z0-9_-]+`. Use for NATS subjects and KV bucket names.
- **instanceId**: runtime-generated short id (base32/58) per daemon instance.

## Subjects
- Instance-directed commands: `f8.<serviceSlug>.<instanceId>.<verb>`
  - Example: `f8.f8_player.ab12cd.cmd.play`
  - No load-balancing; each instance gets its own subject.
- Discovery/health: `$SRV.PING|INFO|STATS.<serviceSlug>.<instanceId?>`
- State deltas: `f8.state.<instanceId>.delta` (per service instance; same scope as the KV bucket)
- Cross-instance state fanout: `state.<instanceId>.set` (source instance emits k/v/rev to subscribers)
- Cross-instance data bus: `f8.bus.<edgeId>` (unique per compiled cross-edge; pub/sub)

Rules: tokens cannot contain `. * >` or whitespace; keep ASCII and short.

## KV buckets
- Per-instance bucket: `kv_f8_<serviceSlug>_<instanceId>` (no dots; matches `[A-Za-z0-9_-]+`).
- Keys may contain dots for hierarchy: `state/system.power`, `operators/<operatorType>`, `graph/<graphId>`.

### KV lifecycle (master-owned)
- Master-started service: before launching the process, master creates the per-instance KV bucket and seeds any known config/state with `rw` and `init` attrs from the provided config.
- Service-initiated register: on successful register, master ensures the bucket exists (idempotent) but does not seed values (since it may already exist).
- Unregister: after the service stops its state manager, master deletes the per-instance bucket if present.

## Examples
- `serviceSlug`: `f8_player`
- `instanceId`: `ab12cd`
- Command subject: `f8.f8_player.ab12cd.cmd.play`
- KV bucket: `kv_f8_f8_player_ab12cd`
