## When to Use

- Use `Lua Script` for compact custom logic that should remain inside the native engine process.
- Choose it for low-overhead synchronous transforms and control rules that fit the documented LuaJIT hook contract.
- Use built-in native operators for common signal processing so their schemas and performance remain predictable.

## Common Wiring Patterns

- Transform numeric inputs into one or more typed outputs from an execution trigger.
- Implement small routing or clamping rules between native operators.
- Prototype logic in Lua before promoting stable high-cost work to a compiled operator.

## Pitfalls / Gotchas

- The default scaffold is illustrative; Python syntax and Python modules are unavailable.
- Script errors are reported through runtime diagnostics and do not silently fall back to pass-through behavior.
- Long loops or blocking work stall the engine because hooks execute synchronously.
