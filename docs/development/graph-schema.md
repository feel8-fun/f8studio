# Graph schema and portable format

This document defines the current authoring boundary. The internal `StudioDocument` is the revisioned editing model (`f8studio-document/3`); `f8graph/4` is the portable project graph format.

## Portable graph

`GET /api/projects/{project_id}/graph/export` returns `f8graph/4`. `POST /api/projects/{project_id}/graph/import` accepts versions 3 and 4 (verifying the original definition hashes before migration) and restores it as a new graph and layout revision. The graph toolbar exposes both operations. Import replaces the current project's graph after confirmation.

```json
{
  "format": "f8graph",
  "formatVersion": 4,
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

- `sdk/schemas/protocol.yml` defines structured `F8UiControlSpec` (`kind`, `optionsFromState`, `language`, `rendererKey`) for states and command parameters. Local generated `describe.json` snapshots with `uiControl` are normalized during discovery; new definitions use `control`.
- State fields and command parameters use `valueRequired` for value validation. Data ports, commands, and exec ports use `definitionProtected` for authoring permissions. The GUI does not expose unlocking protected entries; the server rejects attempts to change their protection flag before deletion.
- Operator exec ports use `F8ExecPortSpec` (`name`, optional label/description/protection). Compilers reduce them to the runtime's string port names. `portIds` keeps connected endpoint identity stable across a rename.
- `F8ServiceSpec` contains no launch policy. Machine-specific launch configuration stays in `F8ServiceEntry.launch` (`service.yml`), while graph instance state values remain in the project document.
- SQLite projects and project snapshots use `f8studio-document/3`; version 2 is migrated on read without changing edit revisions or rewriting historical archives. Components use `f8studio-component/2` and accept version 1 on read/import. Version 1 project documents and `f8graph/2` still require manual conversion.
- The Inspector form covers common state, data, exec, command, and command-parameter edits. Advanced JSON remains for nested value schemas and less common metadata. Server-side validation is authoritative.

## Saved and shared state values

`F8StateSpec` adds two independent booleans in `f8service/2` and `f8operator/2`:

| persistent | publishable | Instance value behavior |
| --- | --- | --- |
| true | true | Save locally; include in shared graphs/components unless excluded for this export. |
| true | false | Save locally; omit from shared graphs/components. |
| false | false | Send updates to the running service only; do not save or increment graph revision. |

Read-only states are not persistent or publishable. `persistent=false, publishable=true` is invalid. `redactOnPublish=true` remains a hard publication restriction, and `publishable` does not affect runtime state broadcasts. For legacy descriptors, writable values default to persistent/publishable, with read-only and redacted fields excluded as appropriate. Loading historical local projects also applies newly installed restrictive field policies and removes obsolete saved transient values.

`GET .../graph/export` is a full local backup and includes private persistent values. `POST .../graph/share` exports a sanitized copy. Its typed request includes `expectedGraphRevision`, `expectedLayoutRevision`, and optional `excludedStates: [{nodeId, field}]`. The server removes private values and initial values with effective enabled upstream state bindings. Field definitions, defaults, examples, ports, and the original local project remain intact. Graph authors can uncheck additional saved values in the shared export dialog.

`POST /api/projects/{project_id}/components` uses the same request plus a component `name` to capture the graph through the server. Component creation, update, import, reads, and historical export all enforce value publication policy. State-only variants use the installed definition to check policy and validate values; nonempty variants without an installed definition fail with an actionable error. Project snapshots keep local persistent configuration.

Sharing does not collect live state, require upstream input, or create component parameters. Parameterized components, publish profiles saved for reuse, Cloud manifests and content versions remain separate future work.
