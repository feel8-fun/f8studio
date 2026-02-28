# Python Expr Service (`f8.pyexpr`)

`f8.pyexpr` is a standalone expression service for lightweight data-flow transforms.

## State Fields

- `code` (`rw`): single-line expression.
- `allowNumpy` (`rw`): enables `np.*` / `numpy.*` calls.
- `unpackDictOutputs` (`rw`): controls dict-output fanout behavior.
- `lastError` (`ro`): last compile/eval error.
- `active` (`rw`): lifecycle active flag.

## Expression Names

- `inputs`: dict of latest input values by port name.
- Identifier-safe input ports are also injected directly as variables.

## Dict Output Behavior

- `unpackDictOutputs = false` (default):
  - A dict result is emitted as one value to default output (`out`).
  - Example result: `{"a": 1}` -> emits one payload `{"a": 1}` on `out`.
- `unpackDictOutputs = true`:
  - Dict keys are treated as output port names.
  - Example result: `{"a": 1, "b": 2}` -> emits `1` on port `a`, `2` on port `b`.
  - Keys without matching output ports are ignored (deduped warning only).

