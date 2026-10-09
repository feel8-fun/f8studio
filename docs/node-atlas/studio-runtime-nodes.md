# Studio Runtime Nodes

The `f8.pystudio` runtime is built into the Web Studio server. These operators provide authoring and presentation behavior without a desktop GUI process.

An executable project has at most one Web Studio Runtime host, with node and
service ID `studio`. Copying a selection duplicates its operators and ordinary
services such as PyEngine while reusing this host. Backend patches and runtime
compilation reject other Studio host IDs. Loading historical projects or graph
imports merges accidentally copied hosts, preserving operators, connections and
editing revisions. The existing `studio` host keeps its settings and layout;
when absent, the first historical host supplies them. Incompatible ports or
conflicting connections produce an error instead of dropping wiring.

Saved state changes synchronize only when that specific node and writable field
exist in the service's currently deployed graph for the project. Newly added or
copied operators keep their configuration in the project until deployment.
Historical deployment receipts do not enable synchronization after a server
restart or after another project replaces the builtin runtime graph. Existing
deployed operators continue to accept live edits while other nodes are still
being authored. Runtime-only commands still report rejected runtime writes.

On the graph canvas, Studio runtime operators can be placed anywhere outside another service's container. Their `serviceId` still binds them to the Studio runtime for validation and deployment. The compact Web Studio Runtime node exposes runtime settings and status; it does not contain or clip its operators. Operators owned by other services use their matching containers as visual parents. Backdrop group movement preserves these runtime bindings, including when an operator moves outside its container.

| Family | Operators and behavior |
| --- | --- |
| Presentation | Text, wave, video, audio spectrum, track overlay, 3D scene and TCode views |
| Graph control | Control panel, patch hub and value stepper |
| Expressions | Typed data and state expressions using an AST allowlist |
| Annotation | Backdrop and note nodes stored with the graph |

Video previews appear inside graph nodes and may also be inspected in the Presentation workspace. 3D and Monaco assets are part of the local production bundle; they do not load code from a CDN.

Select a visualization, Note or Backdrop to reveal its corner resize handles.
The preview or document fills the resized body, and dimensions persist in graph
layout. Notes render their saved `content` as Markdown, including headings,
lists, tables and code blocks, in both the editor and offline graph previews.
Edit their content in the inspector; no deployment is needed to read a Note.

Backdrop is a translucent canvas frame. Drag its title to move the nodes fully
enclosed by its bounds at the start of the gesture. Partial overlaps are excluded.
Services carry their operators exactly once; Studio operators keep their runtime
bindings, and nested Backdrops can move with an outer frame. Resizing a Backdrop
changes its frame without moving nodes. Group movement is one undoable layout
edit and does not increment graph revision or trigger deployment.

Studio runtime operators are explicitly registered under `extensions/f8webstudio/f8studio_server/f8studio_server/studio_runtime`. Presentation renderers are explicitly mapped under `extensions/f8webstudio/f8studio_web/src/presentation`. There is no runtime plugin discovery path.
