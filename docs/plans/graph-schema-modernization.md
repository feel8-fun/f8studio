# Graph schema modernization

This document records the implemented authoring boundary and the remaining protocol migration work. The internal `StudioDocument` is still the revisioned editing model; `f8graph/2` is the portable project graph format.

## Portable graph

`GET /api/projects/{project_id}/graph/export` returns `f8graph/2`. `POST /api/projects/{project_id}/graph/import` accepts it and restores it as a new graph and layout revision. The graph toolbar exposes both operations. Import replaces the current project's graph after confirmation.

```json
{
  "format": "f8graph",
  "formatVersion": 2,
  "metadata": { "graphId": "graph1", "projectId": "project1" },
  "definitions": { "services": {}, "operators": {} },
  "services": {},
  "operators": {},
  "connections": [],
  "resources": {},
  "presentation": { "layout": [], "nodeOrder": [] }
}
```

Definitions are keyed by SHA-256 of their normalized spec JSON and shared by instances. Service and operator instances carry identity, definition reference, enabled flag, state values, and optional `portIds`. `portIds` maps a semantic port key such as `data:output:video` to a stable endpoint ID when a port was renamed. Connections refer to endpoint IDs. Ports, live values, runtime sessions, and revision counters are absent from the portable document. Import verifies the version, definition hashes, references, node order, port map, and graph constraints before it changes a project. `resources` is reserved and must be empty until it has a defined contract.

The definition snapshot contains protocol metadata only; it does not embed Python, JS, or native implementation code. The installed catalog remains the authority for edit permissions. The server compares created, pasted, imported, and edited specs with the installed definition, then applies the definition's collection and field policies. The GUI submits `setServiceSpec` or `setOperatorSpec` with a spec and explicit port renames; the server derives ports and rejects edits that would leave an invalid connection.

Runtime graph revision and compiled deployment input exclude node names, layout, UI controls, display labels, description fields, and other presentation metadata. Runtime interfaces, values, edges, enabled nodes, and service launch configuration remain semantic inputs.

## Current boundaries

- `schemas/protocol.yml` now defines `F8UiControlSpec` (`kind`, `optionsFromState`, `language`, `rendererKey`) for states and command parameters. Python source descriptors use it; legacy `uiControl` remains readable in saved definitions and UI overrides. Studio's discovered `describe.json` files are ignored local generated artifacts; regenerate them with `pixi run update_describes` after changing descriptor sources.
- State `required` still means deploy value validation. Data port and command `required` still mean protected definition. The GUI does not expose unlocking protected entries; the server rejects attempts to change their protection flag before deletion.
- Operator exec ports in the SDK still use string lists. The authoring editor treats them as named interfaces, while `portIds` makes connected endpoint identity independent of the name after a rename.
- Service `launch` still resides on `F8ServiceSpec`; separating an installed service definition, machine launch policy, and instance configuration needs a protocol update.
- SQLite project documents and saved asset versions still use the internal `f8studio-document/1` model. Their database schema and revisions are independent of portable `f8graph/2`; no automatic rewrite of existing archives is performed.
- The Inspector form covers common state, data, exec, command, and command-parameter edits. Advanced JSON remains for nested value schemas and less common metadata. Server-side validation is authoritative.

## Next protocol step

Complete the protocol migration by replacing legacy `uiControl` in saved definitions and UI overrides, defining structured exec port descriptors and distinct value-required/definition-protected fields, and separating service launch policy from the portable definition. Those changes affect Python and C++ generated contracts and require a coordinated codegen and runtime update. Once the descriptors are migrated, remove the legacy fields and advance the internal document format independently of the portable exchange format.
