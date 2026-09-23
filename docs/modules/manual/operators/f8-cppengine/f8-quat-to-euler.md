## When to Use

- Use `Quat To Euler` when you need to convert 3D rotation data (Quaternions) into human-readable Euler angles (Pitch, Yaw, Roll) or axis-specific scalar values.
- It is most useful at the boundary between raw skeleton math and physical actuator control logic, where you need to map a specific joint's rotation to a haptic axis.
- Use it to extract "Tilt" or "Twist" intensities from an limb or a tracked object.

## Common Wiring Patterns

- **Joint-to-Actuator Mapping**: Feed it from a `Bone Selector` or `Bone Filter`. Map the resulting Euler components (X, Y, or Z) through a `Range Map` to drive device outputs.
- **Orientation Debugging**: Route the Euler angles into `f8-viz-wave` to visually analyze the range of motion before setting final control thresholds.
- **Reference Frame Alignment**: Specify the `order` (e.g., XYZ, YZX) and the unit (Degrees vs. Radians) clearly in your configuration to match your downstream control hardware requirements.

## Pitfalls / Gotchas

- **Gimbal Lock**: Euler angles can become unstable at certain orientations (Gimbal Lock). If your control values "flip" suddenly when a joint reaches a specific angle, you may need a different rotation order or a more robust quaternion-based logic upstream.
- **Rotation Order Errors**: Choosing the wrong rotation order can produce motion that looks "correct" on one axis but behaves unpredictably on others. Always verify against a visual skeleton reference.
- **Late Conversion**: Try to keep your math in Quaternion form as long as possible. Only convert to Euler at the very end of your chain to avoid mathematical artifacts during smoothing or interpolation.
