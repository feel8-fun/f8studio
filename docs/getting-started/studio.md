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

## When to Use Studio

Use Studio when you need:

- visual graph editing,
- interactive debugging,
- quick field tuning,
- operator composition experiments.

Use headless Runner when you need unattended execution without GUI.
