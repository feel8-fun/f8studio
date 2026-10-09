# Component Authoring

Components are reusable authoring templates. Inserting one creates ordinary
nodes and connections; updating the source asset leaves inserted nodes alone.
A component can be one customized PyScript node or a selection of nodes.

## Save a selection

1. Configure the nodes, script code, schemas and ports in Graph.
2. Select one or more nodes, right-click a selected node or the canvas, and choose
   **Save selection as Component…**. Right-clicking an unselected node captures
   that node directly, without changing the current selection.
3. Name the template, add a Markdown introduction and search tags, and choose
   which eligible saved state values to include.
4. Find it in the node catalog's **Components** section or **Add from Library**
   (the canvas plus button, Tab, or right-click **Add node…**). Assets remains
   available for metadata/content editing and export.

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

Choose **Details / Versions…** in the catalog, or **Details** in Add from Library,
to see the introduction, tags, fixed version, read-only graph preview, required
hosts and exposed endpoints in the same search window. Search matches names,
descriptions and tags; All / Nodes / Variants / Components filters share one
result list. Select a component in Assets for the same graph preview. Embedded
snapshots permit preview even when its extension is absent. The preview has
static state/port visuals and media placeholders; it never executes script code,
starts services, installs extensions or connects device/media streams.

Click a component or press Enter in the search to add the displayed fixed version.
The selected/right-clicked matching service takes priority for external bindings;
a unique matching host is automatic. When choices remain, choose an existing
matching service for every external host binding in the same window, then choose
**Add node**. Details always waits for confirmation. In Assets, choose a target
project and version, bind hosts and choose **Apply**. A single PyScript
template can reuse the project's current PyEngine. Selecting a service as part
of the original component intentionally includes that service in the template.

Web Studio Runtime is a builtin singleton. Even when captured inside a template,
insertion reuses the target project's `studio` host, preserving its settings and
layout, or creates that host if absent. Ordinary included services still receive
new IDs. External Studio host bindings may have arbitrary template aliases, but
must bind to the target's `studio` host. Preview hosts are placeholders and cannot
be deployed as executable graphs.

Server remaps inserted node and edge IDs, preserves node-scoped port IDs,
translates layout by the requested x/y offsets and validates the complete graph.
Library additions place each externally hosted group inside its chosen container,
preserving relative positions within the group and reserving space for other
groups bound to the same host. Optional `hostOffsets` provide translations per
external binding; callers omitting them retain the original global translation.
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

A Variant is a complete single-node template; it uses `f8component/1` content
with exactly one template node. Presets contain only parameter values. Existing
`f8studio-variant/1` parameter assets are migrated to Presets without changing
their asset IDs or stored versions; older exports remain importable.

## Node Variants

Right-click the customized operator or service and choose **Save as Variant…**.
Enter a name, description and tags, and choose which eligible saved values to
include. Full definition snapshots, custom schemas/ports, script code and saved
node dimensions are retained. Service Variants capture only the service; use a
Component to include its operators. Runtime/private/read-only values are cleaned
using the same publication contract as Components. Saving does not deploy or
change the source graph.

Ordinary authored configuration (intervals, frequencies, thresholds and display
settings) is persistent and publishable. Machine-specific file/executable paths
and device selections stay persistent locally but are not published. Read-only
outputs and transient controls (for example, rescan or clear triggers) are not
persistent. Unchanged configuration uses the embedded definition's default;
the capture dialog marks it as included through the definition. An absent
`stateValues` entry does not mean the configuration is lost. Saved overrides
are selectable separately; excluding an override restores the definition default.

Inline node controls and the Inspector share the saved project document.
Persistent, writable configuration displays that document's value or definition
default, even when the service is stopped or a retained runtime sample differs.
Runtime samples drive read-only outputs, connected state inputs and transient
controls. Editing saved configuration does not require a runtime broadcast.

Find Variants underneath their service/operator type by expanding **Variants**
in the node catalog. **Add node** (Tab or the right-click menu) searches node
types and Variants together. Clicking a Variant or pressing Enter adds its
current version directly. **Versions…** opens compact version/host options in
the same search popup. A selected matching host takes priority; a unique matching
host is selected automatically. Multiple hosts otherwise require a choice. Builtin
Studio operators reuse `studio`, creating that host if absent. New instances
receive new IDs and record their source asset and version in the same transaction
as graph insertion. Missing/incompatible extensions remain previewable but block
insertion. Script operators added through Graph are placed inside their host.

Edit the instance using the normal Inspector, schema or code editors, then
right-click **Update Variant…** to explicitly save a template update. The current
node's source is preselected; another compatible Variant may be chosen. **Save as
new Variant…** creates an independent asset. Updating the asset leaves other
instances untouched. Source records survive server restarts and graph undo/redo;
they are local authoring provenance and are not part of shared graph exports.

Only changed normalized template content increments its version. Moving the
source node, changing asset name/description/tags, or saving unchanged content
does not create a content version. Variant update requests require
`expectedVersion`; stale writers receive a revision conflict. Identity changes
to service/operator classes require a new Variant.

| Operation | HTTP |
| --- | --- |
| Node Variant catalog | `GET /api/variants` |
| Node source versions | `GET /api/projects/{id}/variants` |
| Capture/update node Variant | `POST /api/projects/{id}/variants` |
| Preview/insert fixed Variant version | Same Component preview/insertion routes |

Capture accepts `nodeId`, both expected graph/layout revisions, `name`,
`description`, `tags`, `excludedStates`, and optional `assetId` + `expectedVersion`
for updates. Content and source association commit atomically. Assets provides
metadata editing, fixed version preview/insertion, export and deletion for both
Components and Variants; Presets retain their separate apply-to-existing-node
behavior with class compatibility checks.

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
`version`, `hostBindings` (binding ID to target service ID), optional x/y and
`hostOffsets` (external binding ID to finite x/y translation).
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

Cloud publishing/direct Library queries, extension registry, reusable publication
profiles, component parameters and nested runtime subgraphs remain separate
work. Official bundled templates, linked cloud drafts, role/category forms and
`graph_match_library` are not implemented by the current WebStudio.

The [Unified Library plan](../development/unified-library-plan.md) records the
implemented local search/details workflow and the remaining Cloud work. The Web
Library provider boundary uses explicit local and Cloud references; Cloud
references retain registry ID, fixed version and content hash. Local routes
reject Cloud references. A future Cloud adapter calls Studio Server rather than
importing online entries into the local Assets database. Online search has
separate debounced, cancellable, paginated request state; provider contract tests
cover stale responses, errors and fixed-version insertion. No Cloud provider is
configured in the product yet, so online source filters and social controls are
not displayed. Cloud login/publication, linked drafts and social actions remain
unimplemented.
