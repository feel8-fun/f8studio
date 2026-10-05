## When to Use

- Use this declaration when designing a C++ graph with mutually exclusive exec branches.
- The native runtime is pending. Use PyEngine's `Exec Branch` for executable graphs.

## Common Wiring Patterns

- Connect a trigger to `exec` and map `branch_a`, `branch_b`, and `branch_c` to separate paths.
- Select a port with `selectedBranch`; reserve `default` for an explicit fallback.
- Rejoin exclusive paths with `Exec Merge` when preparing a future native graph.

## Pitfalls / Gotchas

- The current C++ implementation reports `CPP_OPERATOR_UNIMPLEMENTED` and emits no exec outputs.
- The selector represents low-frequency mode configuration.
- Use `Sequence` when every branch must run in order.
