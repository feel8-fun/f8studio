# Graph and Python Node Editing

Use the installed catalog and current graph as the authority for available nodes, ports, and state fields.
For graph changes, build a PatchRequest using current graph and layout revisions. Preview the exact patch before applying it. The application will request human approval for application.

For a Python Script node, read its `code` field with `code_read`. Analyze proposed source with `code_analyze`, then use `code_write` with the graph revision and code SHA-256 returned by `code_read`. If either changed, read the node again and reconcile the edits. Python code is stored in the Studio graph, not in a repository file.

Validate the graph after edits. Deploy only when the task calls for running the graph, and inspect deployment results, logs, and runtime monitors before reporting observed behavior. Report any runtime sync errors.
