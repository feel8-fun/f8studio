# Component Authoring

Components are reusable authoring templates. Inserting one creates ordinary
nodes and connections; updating the source asset leaves inserted nodes alone.
A component can be one customized PyScript node or a selection of nodes.

## Save a selection

1. Configure the nodes, script code, schemas and ports in Graph.
2. Select one or more nodes and choose **Save selection as component** in the
   graph toolbar.
3. Name the template and choose which eligible saved state values to include.
4. Open **Assets** to inspect, preview or export the saved component.

**Capture graph** in Assets captures the complete project instead. Neither
capture changes the project or collects live runtime state.

The template retains full definition snapshots, script code/configuration,
port IDs, layout and internal connections. Private/transient/read-only instance
values are removed. Definition defaults/examples stay intact. Author exclusions
remove values, never fields, ports or definitions.

When an operator's service is outside the selection, the component records a
required host binding. The host's local configuration is not captured. A cut
external connection becomes an exposed endpoint rather than a dangling edge.
Retained internal state connections omit the target's initial value. A cut state
connection keeps an eligible authored fallback; if a required state has no saved
fallback or schema default, capture reports the exact unresolved field.

## Preview and insert

Select a component in Assets to see its read-only graph preview. Embedded
snapshots permit preview even when its extension is absent. The preview has
static state/port visuals and media placeholders; it never executes script code,
starts services, installs extensions or connects device/media streams.

Choose a target project, a fixed component version, and an existing matching
service for every external host binding, then choose **Apply**. A single PyScript
template can reuse the project's current PyEngine. Selecting a service as part
of the original component intentionally includes that service in the template.

Server remaps every inserted node and edge ID, preserves node-scoped port IDs,
translates layout by the requested x/y offsets and validates the complete graph.
Insertion uses one graph patch, so undo removes the whole insertion. Source asset
ID, fixed version, host bindings, and node/edge/endpoint mappings are recorded in
the same database transaction. A database failure changes neither the project
nor its insertion source record. A retry with the same request ID returns its
saved result; use a new request ID to insert another copy. Request replay follows
the project's existing bounded receipt-history window.

Connect the exposed ports in Graph after insertion. Missing implementations,
incorrect hosts, schema conflicts and stale revisions produce errors before
commit. A newer asset version does not replace nodes already inserted.

## Public and local contracts

New captures use portable `f8component/1`. It reuses graph definition references
and stable port identity, adds explicit external `hostBindings` and `endpoints`,
and avoids internal derived GraphNode/ports as its saved format. Readers retain
explicit conversion of older local `f8studio-component/1` and `/2` assets.

Cloud publication is a separate `f8publication/1` envelope containing a manifest
and content hash. License, provenance and extension dependencies belong in that
manifest; local editing revisions never become publication versions. Contract
specification, generated JSON Schema and Python/Web fixtures live in
`extensions/f8webstudio/contracts`.

A variant remains a lightweight state preset, not a complete script template.
Use a component when custom ports, state schemas or code must be retained.

## Shared automation paths

HTTP, Agent, CLI and MCP all call the same application-service operations:

| Operation | HTTP |
| --- | --- |
| Capture selection | `POST /api/projects/{id}/components` |
| Preview fixed version | `GET /api/assets/{assetId}/versions/{version}/preview` |
| Preview insertion | `POST /api/projects/{id}/components:preview` |
| Insert atomically | `POST /api/projects/{id}/components:insert` |

Capture requests include `expectedGraphRevision`, `expectedLayoutRevision`,
`name`, optional `nodeIds` and `excludedStates`. Omit nodeIds to capture all;
an explicitly empty selection is rejected.

Insertion requests include `requestId`, both expected revisions, `assetId`,
`version`, `hostBindings` (binding ID to target service ID), and optional x/y.
The response contains the patch result and source mappings. Preview uses the
exact same request and IDs while leaving the project unchanged. Agent mutations
retain the existing preview/approval workflow.

CLI commands take a typed JSON request file:

```text
pixi run studio_cli capture-component PROJECT request.json
pixi run studio_cli component-preview ASSET VERSION
pixi run studio_cli preview-component-insertion PROJECT request.json
pixi run studio_cli insert-component PROJECT request.json
```

The standalone `GraphView` entry is `f8studio_web/src/graph-view.ts`; it shares
node surfaces, styles and projection with the editor and accepts a validated
StudioDocument snapshot. Build it with `pixi run studio_graph_view_build`.
The JS/CSS library outputs are in `extensions/f8webstudio/build/graph-view`,
with React and React Flow as peer dependencies. It needs no Studio server,
installed catalog or live store once supplied with a snapshot.

## Later work

Cloud publishing/Library synchronization, extension registry, reusable publication
profiles, component parameters and nested runtime subgraphs remain separate
work. Official bundled templates, linked cloud drafts, role/category forms and
`graph_match_library` are not implemented by the current WebStudio.
