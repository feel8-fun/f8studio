# Editor Design (graph, wiring, validation)

- Node palettes show declared ports/commands; allow adding ports/commands only when the corresponding flags are true (`editableDataInPorts`/`editableDataOutPorts`/`editableExecInPorts`/`editableExecOutPorts` for operators, service editable* flags, `editableCommands`).
- Ports: operators use `dataInPorts`/`dataOutPorts` plus `execInPorts`/`execOutPorts`; services use their data port naming. Data vs state vs exec ports are rendered distinctly and cannot be cross-connected.
- Edge creation:
  - Auto-promote operator->external links across container boundaries by synthesizing container ports and half-edges; nested containers follow the same rule.
  - Edge metadata follows `schemas/edge.schema.json`: `kind` (`data|state|exec`), `scope` (`intra|cross`), optional rate strategy/queue/timeout (only for cross data edges).
  - On connect, enforce compatibility:
    - `any` on either end => compatible.
    - State bus: types must match exactly (aside from `any`).
    - Data bus: consumer schema must be satisfied by producer schema (producer provides all required fields with compatible types); otherwise block the link.
    - Exec edges: only between exec pins; single incoming link per `execInPort`.
- Rate mismatch UI for cross-instance data edges:
  - Strategy select (`latest`, `average`, `hold`, `repeat`, `interpolate`), optional queue size, timeout.
  - Only shown when `kind=data` and `scope=cross`; hide for state/intra edges; warn if average/interpolate on non-numeric types.
  - Defaults: strategy=`latest`, queueSize=64, `drop-old` on for cross-instance edges; timeout optional.
- Persistence/export:
  - Save edges with kind/scope/strategy/queue/timeout per `edge.schema.json` (exec edges use `kind=exec` and `scope=intra|cross`).
  - Persist added ports/commands per node; operator manifests use `dataInPorts`/`dataOutPorts`/`execInPorts`/`execOutPorts`, service manifests use their port naming.
- At export/split time, generate half-edges and subjects (`f8.bus.<edgeId>` for data) plus bucket/key mappings for cross-instance state (KV watch), and carry strategy metadata for cross data edges.
- Validation UX:
  - Inline errors for disallowed additions, type/schema incompatibility, or missing required fields.
  - Mark cross edges and strategy badges on edges/ports for visibility.

## Branch/merge nodes
- Model branch (switch/gate) and merge (select/priority) explicitly. Each data path uses its own port; merges have multiple input ports (no multi-link into a single port). Topology stays static; branch nodes decide per tick which output carries data; merge nodes define a fixed selection policy.

## Offline/limited mode
- If master is unavailable, engines keep running the last applied graph, but new apply requests are rejected (read-only notice to the user). Editor enters degraded mode: no deploy, show warning.
- Editing is still allowed as draft; deployment requires master to compile/deploy the full graph. Unknown/missing services block deployment.
- You can still draw cross-instance links while offline; they are marked pending and will not execute until master returns and compiles the graph.

## Renderer extensibility (hooks pre-baked)
- Keep `rendererClass` as the lookup key. Resolution order: built-in registry -> dynamic registry -> fallback generic renderer.
- Provide a registration hook: `registerRenderer(rendererClass, componentOrFactory)`; future user plugins call this when loaded.
- Plugin loading plan: load JS bundles from a trusted local plugins folder (e.g., `~/.feel8/plugins/`, configurable). Require a manifest (name, version, rendererClasss, entry path); allow a reload-plugins action for dev. Loader can be disabled by default until an allowlist is configured.
- Safety: only load allowlisted/signed bundles; on load failure, fall back to the generic renderer so nodes still render.
- Contract to document later: props include node state, ports, commands, status, and callbacks to send commands/update state; host provides theme/styling context.
- Delivery path: browser fetches plugins via local HTTP served by master/gateway (manifest endpoint + static JS under `/plugins/...`); no direct disk access from the page.

## Status/health
- Primary: status in each instance's KV bucket; editor watches KV for updates. Optional status pub subject can exist, but keep payload small (status/lastError); avoid high-churn metrics.
## UI layout
- Left panel:
  - Top: Operator Library.
  - Bottom: Service Library.
  - Bottom-left status pane shows connection to master (connected/disconnected).
- Center canvas: graph editor. Top-left shows current graph name (`untitled` until saved; `*` when dirty). Top-right has tool buttons, e.g., "auto-activate new nodes" toggle (if on, new nodes start activated; off = start disabled).
- Right panel: Properties Inspector for selected item.

## Properties Inspector behavior
- Supports ServiceNode, OperatorNode, Edge, and canvas background (Web Engine).
- Editable fields (access=rw) commit on Enter/blur, not per keystroke; commits send updates to master.
- Edge inspector shows kind/scope/strategy, queue, timeout (per edge schema).
- Background selected -> show Web Engine properties.

## Containment rules
- Operator nodes must be placed inside a Service node (they run in that service process).
- Service nodes can stand alone. Web Engine is the default canvas host: you can place Web Engine-supported operators directly on empty canvas; when nothing is selected, the inspector shows Web Engine props.
- Dragging an engine service from library triggers master to launch an engine process; engine writes its supported operators to its KV bucket; after that, operators can be dragged into that engine's service node as children.

## Auto-generated node elements
- Ports: generate data ports on node sides, top-to-bottom (operator: `dataInPorts`/`dataOutPorts`; service: its declared ports). Render exec ports distinctly (e.g., different color/shape); enforce single incoming link per execIn port.
- showOnNode state fields render inline rows with control in the middle, input handle on the left, output handle on the right. Handles follow access: rw/init/wo get inputs; rw/ro get outputs (wo = write-only, no output handle).
- Commands: simple invoker UI per node-combobox of commands, auto-rendered fields for params, send button, and a reply display area.
- Service node header: show label (or name) and a status color (green=healthy, gray=stopped, red=error).

## Notifications
- Errors/tips surface as toasts.
