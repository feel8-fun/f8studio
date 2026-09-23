## When to Use

- Use `UDP Out` to send values from a C++ engine graph to an external UDP listener.
- It is a simple bridge for diagnostics, remote control payloads, or interoperability with other local tools.
- This node is appropriate when delivery can be best-effort and connectionless.

## Common Wiring Patterns

- **Debug Egress**: Send intermediate values to a small local receiver for inspection outside Studio.
- **Device Bridge**: Feed mapped values or formatted text from `CPython Script`, `Lua Script`, `Data Expr`, or protocol builders into `UDP Out.value`.
- **Line-Based Text Sender**: Enable `appendNewline` or `forceText` when the receiver expects plain text records.

## Pitfalls / Gotchas

- **No Delivery Guarantee**: UDP can drop or reorder packets, so do not assume reliable transport.
- **Formatting Mismatch**: Confirm whether the receiver expects text bytes or a binary payload before toggling `forceText`.
- **Network Scope**: Sending to non-loopback targets still needs the right firewall and host configuration on both sides.
