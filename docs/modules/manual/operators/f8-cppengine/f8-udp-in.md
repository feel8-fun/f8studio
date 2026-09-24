## When to Use

- Use `UDP In` as the generic ingress node for any UDP packet stream.
- It is the right starting point when the payload format is not yet decoded, or when multiple downstream consumers need access to the same packet metadata.
- Pair it with `Skeleton Decoder` or `VMC Decoder` for motion protocols instead of using legacy protocol-specific UDP nodes.

## Common Wiring Patterns

- **Binary Motion Stream**: Connect `packet` to `Skeleton Decoder.packet` or `VMC Decoder.packet` so downstream nodes receive the raw payload plus packet metadata.
- **JSON Packet Ingest**: Wire `json` into downstream logic when the sender emits UTF-8 JSON payloads, and keep `text` attached for inspection.
- **Debug Branch**: Attach `Text Viz` or `Print` to `text`, `json`, or `packet` while keeping protocol decoding on a separate branch.

## Pitfalls / Gotchas

- **Exact Bytes vs Packet Envelope**: Use `raw` when downstream only needs payload bytes. Use `packet` when downstream also needs source/timestamp metadata or exec-context packet snapshots.
- **No Packet-Rate State**: Packet counters, byte lengths, remote address, and parse diagnostics are intentionally not published as state. Read packet-rate information from `raw`, `text`, `json`, or `packet` outputs so the Studio UI state sync does not get flooded.
- **Bind Security**: Non-loopback bind addresses stay blocked unless `allowNonLoopbackBind` is enabled explicitly.
- **Protocol Split**: `UDP In` does not decode motion payloads on its own anymore; decoding must happen in a dedicated downstream operator.
