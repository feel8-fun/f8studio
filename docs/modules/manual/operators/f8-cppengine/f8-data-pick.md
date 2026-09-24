## When to Use

- Use `Data Pick` to extract one nested value from a JSON-compatible object or array.
- Choose it for small paths such as `center.y`, `pos[1]`, or `["custom-key"].score`.
- It is useful before numeric native operators that should receive a scalar instead of a full payload.

## Common Wiring Patterns

- Pick one coordinate from a tracking result and pass it to smoothing, mapping, or TCode operators.
- Extract a status field and feed it into an expression or switch branch.
- Keep the original payload on a parallel debug or presentation branch while extracting the runtime value.

## Pitfalls / Gotchas

- The path syntax is intentionally small and is not a general query language.
- Missing keys, invalid indexes, and incompatible container types produce explicit operator errors.
- Avoid repeatedly traversing large per-frame objects when an upstream service can publish the needed field directly.
