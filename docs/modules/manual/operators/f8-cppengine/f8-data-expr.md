## When to Use

- Use `Data Expr` when you want to execute a compact scalar expression over one or more numeric input payloads in the C++ engine.
- It is a good fit for minor data transforms, field extraction from JSON, simple conditionals, or unpacking one result into multiple named outputs.
- Best for logic that is too complex for a single property but doesn't require complex state management.

## Common Wiring Patterns

- **Multi-Input Reduction**: Feed it from multiple service or operator data ports. Reference the default input as `x` or use named input ports directly in your expression.
- **Output Unpacking**: Enable `Unpack Dict Outputs` when your expression returns a dictionary. The operator will automatically route values to output ports whose names match the dictionary keys.
- **Signal Gating**: Use a conditional expression (e.g., `x if x > 0.5 else 0`) to gate or filter incoming values before they move further down the graph.

## Pitfalls / Gotchas

- **Complexity Creep**: Expressions stay maintainable only while they are small. Use a native operator, `Lua Script`, or `CPython Script` when the logic needs persistent state or multiple steps.
- **Port Mapping**: Output unpacking only happens for keys that *exactly* match existing output port names. Double-check your spelling!
- **Expression Errors**: Invalid syntax and unsupported values are reported through runtime monitoring. Watch the node monitor while editing an expression.
- **Language Subset**: Python syntax, imports, and NumPy are not supported by the native scalar evaluator.
