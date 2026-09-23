# Web Studio Quickstart

Feel8 Web Studio is the graph editor for creating, configuring, deploying, and observing Feel8 services. It runs in Chromium-based browsers while the local Studio server keeps graph state, credentials, devices, runtime processes, and media signaling on the host.

## Launch

From a packaged release, run `f8studio.exe` on Windows or `./f8studio` on Linux. The launcher installs the pinned runtime when needed, waits for the server health endpoint, and opens `http://127.0.0.1:8210`.

From a source checkout:

```bash
pixi run -e web-studio studio_web_build
pixi run -e web-studio studio_server
```

Open `http://127.0.0.1:8210`. For frontend development, run the API and Vite server in separate terminals:

```bash
pixi run -e web-studio studio_server
pixi run -e web-studio studio_web_dev
```

## Build a Graph

1. Create or select a project from the project selector.
2. Drag a service from Node Library to the canvas.
3. Drag operators into their owning service container.
4. Connect compatible `exec`, `data`, `state`, or `command` terminals.
5. Select a node and edit schema-driven fields in Inspector.
6. Deploy and inspect service status, monitor events, logs, and presentation output.

Service nodes represent deployable runtime processes. Operator nodes execute inside their parent service. A state input becomes read-only while an incoming state edge owns its value. Invalid ownership, port kind, schema, and multiplicity combinations are rejected before commit.

## Editor Workspaces

| Workspace | Purpose |
| --- | --- |
| Graph | Canvas, Node Library, Inspector, deploy jobs and inline previews |
| Presentation | Larger video, audio, curve, track, TCode and 3D views |
| Assets | Components, variants, immutable versions and project snapshots |
| Code | Local Monaco editor with completion, hover and diagnostics |
| Local | Unity setup, serial devices, skeleton verification and hotkeys |
| Agents | Deterministic or model-backed graph construction and diagnosis |
| Logs | Recent service output, deployment results, runtime errors, and media signaling errors |

Graph changes are saved to the local SQLite store. Layout and graph revisions are tracked separately, and other open tabs receive committed changes over the event stream without rebuilding the whole page.

## Components and Variants

Capture a selected subgraph as a component when you want to reuse structure. Capture a variant when you want reusable state values for one service/operator type. Assets have immutable versions and can be exported or imported as typed JSON.

## Runtime Media

Video Viz previews play directly inside graph nodes. The dedicated media gateway subscribes to Zenoh and serves video/audio through WebRTC. In strict VPN or SSH environments, configure TURN and forward both the Studio HTTP port and TURN TCP port as described in [the migration status](../plans/web-studio-status.md#严格-vpn--ssh-模式).

## AI, CLI and MCP

The Agent workspace, CLI, and MCP server all modify the same authoritative project document. Start the sidecars with:

```bash
pixi run -e web-studio studio_cli --help
pixi run -e web-studio studio_mcp --help
```

Provider credentials remain server-side. Deterministic graph tools work without a model credential.

## Troubleshooting

- A blank root page usually means the production Web bundle was not built; run `studio_web_build` in a source checkout.
- A service timeout means its Zenoh command endpoint did not become ready. Inspect the deploy job and service process logs.
- Video with no frames usually means WebRTC ICE connectivity failed. Verify the media gateway and TURN configuration separately from the Studio HTTP port.
- A rejected edge includes the ownership or type rule that failed. Fix the terminal types instead of editing persisted JSON manually.

See [Web Studio Architecture](../developers/web-studio-architecture.md) and [Node Atlas](../node-atlas/index.md) for implementation and node details.
