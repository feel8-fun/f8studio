## When to Use

- Use this declaration to describe a continuation shared by mutually exclusive C++ exec branches.
- The native runtime is pending. Use PyEngine's `Exec Merge` for executable graphs.

## Common Wiring Patterns

- Connect each branch's terminal exec output to a named merge input.
- Route `exec` to the common continuation.
- Select branch-specific data separately with `Data Mux`.

## Pitfalls / Gotchas

- The current C++ implementation reports `CPP_OPERATOR_UNIMPLEMENTED` and emits no exec outputs.
- This contract does not promise deduplication of simultaneous triggers.
- Merging control flow does not combine data values.
