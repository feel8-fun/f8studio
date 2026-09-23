## When to Use

- Use `f8.pystudio` for editor-local helper nodes such as notes, local previews, control panels, and authoring-time aids.
- It is useful for annotation, inspection, manual testing, and graph readability while working inside Studio.
- Treat it as the authoring toolbox rather than as part of the deployed runtime.

## Common Wiring Patterns

- Use it for `Note`, `Control Panel`, and local visualization nodes.
- It commonly sits alongside deployable services, but with a different role: authoring support instead of runtime work.
- It is especially helpful when you need to watch, explain, and poke a graph while tuning it.

## Pitfalls / Gotchas

- Do not rely on `f8.pystudio` nodes for core runtime behavior if the graph must run correctly without Studio.
- Heavy use of local previews and visual nodes still costs local CPU and memory.
- For the broader UI workflow, pair this page with the Web Studio quickstart.
