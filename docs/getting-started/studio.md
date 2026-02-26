# Studio (GUI)

`f8pystudio` is the interactive node-graph environment for assembling and operating service graphs.

## Launch

```bash
pixi run -e default studio
```

Force live describe discovery (ignore static `describe.json`):

```bash
python -m f8pystudio.main --discovery-live
```

## Typical Workflow

1. Start Studio.
2. Add service nodes from the service library.
3. Configure service state fields in the property panel.
4. Add operator nodes where needed (primarily in `f8.pyengine`).
5. Wire state/data/exec edges.
6. Save session JSON for reuse in headless mode.

## Edge Rules

Studio enforces 3 independent edge kinds:

- `exec` (white, thick line)
- `data` (gray line)
- `state` (yellow line)

Rules:

- `exec` can only connect `exec -> exec`.
- `exec` endpoints must both be operator nodes in the same `svcId` (same engine instance).
- `exec` is single-in and single-out per port (reconnect replaces the old edge).
- `data` can only connect `data -> data`.
- `state` can only connect `state -> state`.
- `data`/`state` are `multiple-out, single-in`, and allow cross-service links.

Legacy sessions:

- Invalid connections are stripped automatically on load, with warning logs.

Toolbar visibility:

- Use `Exec Lines`, `Data Lines`, `State Lines` toggle actions to show/hide each edge kind independently.

## When to Use Studio

Use Studio when you need:

- visual graph editing,
- interactive debugging,
- quick field tuning,
- operator composition experiments.

Use headless Runner when you need unattended execution without GUI.
