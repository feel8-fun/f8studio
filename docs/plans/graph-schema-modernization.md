# Graph schema modernization

This document records the implemented authoring boundary. The internal `StudioDocument` is the revisioned editing model (`f8studio-document/2`); `f8graph/3` is the portable project graph format.

## Portable graph

`GET /api/projects/{project_id}/graph/export` returns `f8graph/3`. `POST /api/projects/{project_id}/graph/import` accepts it and restores it as a new graph and layout revision. The graph toolbar exposes both operations. Import replaces the current project's graph after confirmation.

```json
{
  "format": "f8graph",
  "formatVersion": 3,
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

Runtime graph revision and compiled deployment input exclude node names, layout, UI controls, display labels, description fields, and other presentation metadata. Runtime interfaces, values, edges, and enabled nodes remain semantic inputs. Launch configuration is resolved from the installed service entry when deployment starts.

## Current boundaries

- `schemas/protocol.yml` defines structured `F8UiControlSpec` (`kind`, `optionsFromState`, `language`, `rendererKey`) for states and command parameters. Local generated `describe.json` snapshots with `uiControl` are normalized during discovery; new definitions use `control`.
- State fields and command parameters use `valueRequired` for value validation. Data ports, commands, and exec ports use `definitionProtected` for authoring permissions. The GUI does not expose unlocking protected entries; the server rejects attempts to change their protection flag before deletion.
- Operator exec ports use `F8ExecPortSpec` (`name`, optional label/description/protection). Compilers reduce them to the runtime's string port names. `portIds` keeps connected endpoint identity stable across a rename.
- `F8ServiceSpec` contains no launch policy. Machine-specific launch configuration stays in `F8ServiceEntry.launch` (`service.yml`), while graph instance state values remain in the project document.
- SQLite project documents and saved asset versions use `f8studio-document/2`. Existing version 1 documents and `f8graph/2` exports require a manual conversion; there is no automatic archive rewrite.
- The Inspector form covers common state, data, exec, command, and command-parameter edits. Advanced JSON remains for nested value schemas and less common metadata. Server-side validation is authoritative.
