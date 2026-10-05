## When to Use

- Use this declaration when designing translation and rotation signals relative to a reference bone.
- The native runtime is pending. Use PyEngine's `Relative Pose Axes` for live pose calculations.

## Common Wiring Patterns

- Feed bone poses into `referenceBone` and `targetBone`.
- Select `primaryAxis` and inspect the planned `L0/L1/L2` and `R0/R1/R2` outputs.
- Normalize, limit, and smooth geometric values before device output.

## Pitfalls / Gotchas

- The current C++ implementation reports `CPP_OPERATOR_UNIMPLEMENTED` and returns null data outputs.
- Local axes depend on the reference bone's orientation.
- Missing or stale poses require explicit downstream validity checks.
