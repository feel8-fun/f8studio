# Graph and Python Node Editing

Use `graph_read` for current node IDs, ports, edges, state values, layout, and revisions. Use `catalog_search` to find candidates, then `catalog_operator` for exact behavior, port names, state defaults, and constraints. Do not infer ports from labels or stop after listing the catalog. `catalog_create_node` only returns a template; it does not add a node.

For graph construction, call `graph_propose_changes` with current revisions and compact `nodes`, `connections`, and `stateUpdates`. Connections use port names; the server resolves specs and port IDs, builds the full patch, and validates it. A proposal may include `refreshInstalledSpec` operations for compatible old nodes; these are reviewed and approved with the requested graph changes. Then immediately call `graph_apply_proposal` with the returned proposalId. This call opens the approval UI and waits for the user. Do not end with "please approve" in chat: no approval UI exists until the apply tool is called. After approval, the tool applies the exact previewed patch. Do not claim that a template or preview changed the graph.

For advanced raw edits, `graph_preview_patch` and `graph_apply_patch` remain available. These require a full PatchRequest and real port IDs. Choose a reasonable sampling interval when the user specifies frequency but not display resolution, and mention the choice in the result.

Cross-service data edges into Studio visualization nodes request periodic upstream sampling from PyEngine or CppEngine. A visualization can therefore drive a pull-computed Phase -> Cosine -> TCode chain without wiring an execution Tick. For smooth 1 Hz Wave Viz and TCode Viz output, 20 ms upstream sampling is a reasonable default; set both visualizers' `upstreamSampleIntervalMs` to 20, and set TCode's `intervalMs` to 20. Inspect the target visualization's sampling settings before assuming an existing Tick controls it. Verify the compiled or deployed graph before claiming that samples are visible.

For a Python Script node, read its `code` field with `code_read`. Analyze proposed source with `code_analyze`, then use `code_write` with the graph revision and code SHA-256 returned by `code_read`. If either changed, read the node again and reconcile the edits. Python code is stored in the Studio graph, not in a repository file.

Validate the graph after edits. Deploy only when the task calls for running the graph, and inspect deployment results, logs, and runtime monitors before reporting observed behavior. Report any runtime sync errors.
