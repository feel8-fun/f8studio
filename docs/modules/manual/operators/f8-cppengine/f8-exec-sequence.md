## When to Use

- Use `Sequence` when one exec trigger needs to fan out into multiple branches in a fixed order.
- It is useful for making evaluation order explicit on the canvas.
- Put it near the top of a chain when later branches depend on work done by earlier branches in the same tick.

## Common Wiring Patterns

- **Ordered Pipeline**: Feed a `Tick` into `Sequence`, then reserve output `0` for reads, `1` for transforms, and `2` for side effects.
- **Split Side Effects**: Trigger logging or visualization on one branch before hardware output on a later branch.
- **Startup Orchestration**: Use separate outputs for reset, calibration, and normal execution steps.

## Pitfalls / Gotchas

- **Order Only**: `Sequence` controls execution order, not isolation; a slow early branch still delays later ones.
- **Branch Sprawl**: Overusing nested sequences can make graphs harder to read than separate subgraphs or service hosts.
- **Port Priority**: Lower-numbered exec outputs run first, so wire dependencies accordingly.
