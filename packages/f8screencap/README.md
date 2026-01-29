# f8screencap_service

Screen capture service that publishes BGRA frames into shared memory.

## Backends

- Windows: Windows Graphics Capture (WGC)
- Linux: X11 `XGetImage` (requires an X11 session / `DISPLAY`)

## Linux notes

- Wayland is not supported by this backend (use an X11 session, or add a PipeWire/portal backend in the future).
- `windowId` format: `x11:win:0x...` (X11 window id / XID).
- `listDisplays` returns X11 screens (index `0..N-1`).
- `pickRegion` is supported on X11 (drag to select, `Esc` cancels). `pickWindow/pickDisplay` are currently Windows-only.
