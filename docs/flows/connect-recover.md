# Connect, Degrade, Recover (Web <-> f8master over NATS)

Sequence (single user, single connection):

```mermaid
sequenceDiagram
  participant Web
  participant Master

  Web->>Master: f8.master.ping {clientId, requestedProfile}
  Master-->>Web: {status:ok, capabilities, profile}
  Web->>Web: switch to enhanced mode

  Note over Web,Master: Normal ops (config apply, playback control, deltas)

  Master--xMaster: disconnect / crash
  Web->>Web: detect timeout, mark degraded (web-only), disable enhanced actions

  Master->>Master: restart + resubscribe
  Web->>Master: f8.master.ping
  Master-->>Web: {status:ok}
  Web->>Master: f8.state.<instanceId>.snapshot
  Master-->>Web: snapshot
  Web->>Web: re-enable enhanced mode
```

Notes:
- Transport is NATS request/reply (not expanded in the diagram).
- Enforce single client: master rejects a second `clientId` (or kicks old session).
- When master drops, engines keep running the last applied graph but mark apply endpoints read-only; web enters degraded mode (no deploy).
- Web-only mode keeps graph edit/save; playback/daemon-backed actions disabled.
- After reconnection, fetch snapshot/etag before sending new edits to avoid stale apply; then re-enable enhanced mode.
