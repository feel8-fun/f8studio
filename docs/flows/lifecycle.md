# Service Lifecycle (with f8master registry)

States: `starting` -> `registered/deactivated` -> `activated` <-> `deactivated` -> `terminating` | `error`.

## Registration (f8master is authority)
- Service loads `service.json` locally, validates.
- Calls `f8.master.registry.register` (req/rep) with:
  - `manifest`: full service profile (inline JSON).
  - `manifestHash` (sha256 of manifest, optional but recommended).
  - `instanceId`: proposed or empty (master can assign).
  - `bootstrapRev` / `stateHash` (optional) if KV snapshot was preseeded.
- Master validates schema, slug/ID uniqueness, manifest size, hash. On `ok`, returns confirmed `instanceId` and optional `kvBucket`.
- If rejected, service exits (`error` state).

## After registration
- Master (or service) ensures per-instance KV bucket exists: `kv_f8_<serviceSlug>_<instanceId>`.
- Service loads initial snapshot from KV (`state/snapshot`), applies defaults, enters `registered/deactivated`.
- NATS handlers are live; processing is gated.

## Activation / Deactivation / Termination
- `activate`: `f8.<serviceSlug>.<instanceId>.cmd.activate` -> service enters `activated`, starts processing.
- `deactivate`: `f8.<serviceSlug>.<instanceId>.cmd.deactivate` -> pause processing, keep state/NATS alive.
- `terminate`: `f8.<serviceSlug>.<instanceId>.cmd.terminate` -> service unregisters via `f8.master.registry.unregister`, cleans up, exits.

## Discovery / Health
- Optional passive info via `$SRV.INFO.<serviceSlug>.<instanceId>`; primary truth is f8master’s registry.

## Subject & KV naming (recap)
- Commands: `f8.<serviceSlug>.<instanceId>.cmd.<verb>`
- Registry RPCs: `f8.master.registry.register`, `f8.master.registry.unregister`
- KV bucket: `kv_f8_<serviceSlug>_<instanceId>`

## Notes
- Keep manifest message under NATS size limits; include `manifestHash` to detect corruption.
- Reject activation if snapshot is missing/mismatched (hash/rev).
- Only f8master admits registrations; uniqueness enforced there.***
