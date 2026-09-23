## When to Use

- Use `Bone Selector` to "extract" a single named bone (e.g., `RightWrist`, `Pelvis`) from a full skeleton payload provided by an upstream ingest node.
- It acts as the primary bridge between a dense skeleton stream (containing dozens of joints) and your specific, bone-targeted control logic.
- Use it to isolate a specific joint's movement for specialized analysis or device mapping.

## Common Wiring Patterns

- **Joint Isolation**: Feed it from `Skeleton Decoder`, `VMC Decoder`, or a MediaPipe source. Pass the resulting single-bone payload into a `Bone Filter` or `Quat To Euler` operator.
- **Multi-Bone Processing**: Use multiple `Bone Selector` nodes in parallel to extract different joints (e.g., both hands) for a coordinated interaction scenario.
- **Dynamic Selection**: Use the `availableBones` output list in conjunction with a `Control Panel` to interactively switch which joint your graph is following during a tuning session.

## Pitfalls / Gotchas

- **Naming Discrepancies**: A "missing" bone is often just a naming mismatch (e.g., `Hips` vs. `Pelvis`) between the source stream and your selector configuration. Check the `availableBones` port to see valid names for your current source.
- **Stream Continuity**: This node depends on a valid, continuous skeleton payload. If the upstream detector loses the person, this node will stop producing updates.
- **Abstraction Limits**: This node only selects data; it does not perform any math or transformation. Use `Bone Filter` or `Quat To Euler` for subsequent processing.
