## When to Use

- Use `Bone Filter` when you need to stabilize and normalize a skeleton bone pose (usually from MediaPipe or a VMC stream) into a local, jitter-free control signal.
- It is essential for skeleton-driven control graphs where sensor noise or tracking jitters would otherwise cause physical actuators to "chatter."
- Ideal for calculating relative angles or distances between bones (e.g., wrist position relative to shoulder) with built-in temporal smoothing.

## Common Wiring Patterns

- **Stable Control Loop**: Feed it from a `Bone Selector` operator. Use the `filtered` (smoothed world space) or `relative` (position relative to a parent bone) outputs for `Range Map` or Euler conversion.
- **Visual Calibration**: Tune the filter properties (like `alpha` or `cutoff`) while watching a live skeleton in `f8-viz-three-d` to find the best balance between jitter reduction and following lag.
- **Gesture Normalization**: Use the `relative` output to drive gesture-recognition logic that should ignore the subject's overall position in the room.

## Pitfalls / Gotchas

- **Correct Bone Verification**: Do not attempt to tune the filter coefficients before confirming that the `Bone Selector` is actually providing the correct bone data stream.
- **Jump Reset Latency**: If your "Jump Reset" (threshold for ignoring large discontinuities) is too aggressive, it may cause the skeleton to "stick" or lag behind when the subject moves rapidly.
- **Coordinate Systems**: Ensure your world-space assumptions match the upstream source (e.g., MediaPipe's flipped Y-axis) before relying on the filtered relative positions.
