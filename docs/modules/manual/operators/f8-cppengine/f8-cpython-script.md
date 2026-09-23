## When to Use

- Use `CPython Script` to migrate or diagnose script logic while the surrounding graph runs in the C++ engine.
- Choose it for synchronous JSON-compatible transformations that need CPython libraries unavailable to Lua.
- Prefer the Python engine's full script operator when you need its broader hook and type behavior.

## Common Wiring Patterns

- Feed scalar or JSON-compatible inputs into a short script and route its outputs back to native operators.
- Use it as a temporary comparison branch while porting an algorithm from Python to C++.
- Keep execution explicitly triggered so slow scripts do not run on every unrelated graph update.

## Pitfalls / Gotchas

- Embedded Python still uses the Python runtime and can become the throughput bottleneck in an otherwise native graph.
- Hooks are synchronous in V1; blocking I/O stalls the C++ engine execution path.
- Inputs and outputs must stay within the documented JSON-compatible boundary.
