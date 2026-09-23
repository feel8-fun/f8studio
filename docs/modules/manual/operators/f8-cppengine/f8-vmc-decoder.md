## When to Use

- Use `VMC Decoder` after `UDP In` when the incoming UDP packets carry VMC OSC messages.
- Use it as the decoding stage in a `UDP In -> VMC Decoder` chain for live VMC streams.
- This split keeps OSC/VMC decoding independent from socket lifecycle, which makes graphs more composable and easier to debug.

## Common Wiring Patterns

- **Avatar Pipeline**: Connect `UDP In.packet` to `VMC Decoder.packet`, then use `selectedSkeleton` for `Bone Selector`, `Bone Filter`, or avatar-driven control logic.
- **Live VMC Debugging**: Keep a `Print` or `Text Viz` branch on `UDP In.packet` while the main branch feeds `VMC Decoder`.
- **Multi-Model Selection**: Use `availableKeys` and `selectedKey` to lock onto the intended avatar when multiple identities are present.

## Pitfalls / Gotchas

- **Binary Payload Required**: VMC is an OSC-based binary protocol. Feed this node with `UDP In.packet` or direct raw bytes so decoding reads the original payload, not text/json views.
- **Decoder Placement**: Put VMC-specific logic after `VMC Decoder`, not before; upstream nodes should stay transport-oriented.
- **Packet Contract**: Feed this node with `UDP In.packet` so OSC/VMC decoding stays isolated from socket management.
