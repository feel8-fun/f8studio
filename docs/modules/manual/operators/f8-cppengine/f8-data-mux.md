## When to Use

- `Data Mux` describes a planned C++ Engine operator for selecting one JSON-compatible data input and exposing it through `out`.
- Its native runtime is not implemented yet. Use the `f8.data_mux` operator in PyEngine when a graph needs working data selection today.

## Common Wiring Patterns

- The declared inputs are `branch_a`, `branch_b`, `branch_c`, and `default`; the declared output is `out`. Additional named data inputs may be added in the graph definition.
- Set `selectedInput` to the name of the desired input port. The declared `resolvedInput` state is intended to show which port was used after fallback.
- The declared `exec` input and output allow a selection step to sit in an execution chain once a native implementation exists.

## Pitfalls / Gotchas

- The C++ node currently reports `CPP_OPERATOR_UNIMPLEMENTED`: it emits no exec output, and requests for `out` return null. Wiring it into a deployed C++ graph will not select or forward data.
- Adding ports or changing `selectedInput` does not enable the pending runtime. Build the active path with PyEngine until native support is implemented.
