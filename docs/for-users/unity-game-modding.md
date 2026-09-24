# Unity Game Modding

Web Studio can install the managed F8 skeleton exporter into a supported local
Unity game, verify its UDP stream, and create a guarded skeleton-to-OSR graph.
The workflow never installs from detection alone: you preview exact writes and
confirm them before the game directory changes.

## Before You Start

- Close the game before installing or updating its loader and exporter.
- Keep a backup of a modded game directory if it contains custom loader files.
- Connect no physical output while first verifying skeleton tracking.
- The default skeleton UDP endpoint is `127.0.0.1:39540`.

## Detect And Install

1. Open the `Local` workspace in Web Studio.
2. Select the game executable or game root and run detection.
3. Review the detected Unity backend, game profile, loader, exporter, and every
   proposed destination path.
4. Resolve any blocking diagnostic before continuing. Unknown games require an
   explicit custom-profile preview.
5. Confirm the preview to apply the managed installation.

Existing custom configuration is preserved unless the preview explicitly marks
a file as managed and scheduled for update. Web Studio does not guess a loader or
write to a game directory before confirmation.

BepInEx is considered installed only when its Mono/IL2CPP loader entry DLL,
`doorstop_config.ini`, and `winhttp.dll` are all present. If detection finds only
an exporter directory, an auxiliary DLL, or part of the bootstrap, the preview
shows **Repair incomplete BepInEx** and the apply step restores the missing
official loader files before installing the exporter.

## Check UDP Skeleton Data

Start the game, enter a scene where the selected characters and bones exist,
then use **Check UDP Skeleton Data** in the dialog. This checks the local UDP
skeleton packets emitted by the exporter; it has no connection to Steam.

A valid result reports `listenerStatus=verified` and at least one decoded frame.
UDP packet count alone is not success: malformed packets are rejected, and a
chunked skeleton counts only when its complete frame has arrived. The result
lists stable skeleton keys, detected roles, bone names, exporter version, and
decoder errors when present.

If verification fails:

- Confirm the game and exporter are running.
- Confirm no other application owns UDP port `39540`.
- Check the game loader log for exporter startup errors.
- Confirm the selected profile and backend match the detected installation.
- Treat `packets_rejected` as a protocol/configuration problem, not as a healthy
  stream.

## Build The OSR Graph

Graph preview remains unavailable until a complete binary skeleton frame has
been verified. After verification:

1. Choose the reference and target roles and their role indices.
2. Choose the reference bone, target bone, and primary local axis.
3. Preview the graph plan and inspect its selectors, calibration, output range,
   smoothing, rate limits, and watchdog.
4. Confirm graph application to add it to the active Studio graph.
5. Verify skeletons in `3D Viz`, then verify raw and normalized `L0` values on
   monitor/data channels.
6. Use `TCode Viz` to inspect generated commands before selecting a serial port.

The generated path is:

```text
UDP In -> Skeleton Decoder -> stable character selectors -> bone selectors
-> Relative Pose Axes -> Envelope -> Range Map -> Smooth Filter
-> Rate Limiter -> TCode -> TCode Viz / Serial Out
```

`Serial Out` is created with `enabled=false`. A 250 ms stream watchdog gates its
execution, so stale tracking cannot continue driving output. Arm the serial node
only after the visualized skeleton, direction, output range, and rate limits are
correct for the scene and device.

## Save A Recipe

Save the verified setup as a recipe after the graph is correct. Recipe version 2
records the profile hash, exporter evidence, stable selectors, bones, axis,
calibration, graph node IDs, verification counters, safety settings, and serial
arm state. Sharing a recipe does not install a game automatically; the next user
still receives the same preview and confirmation boundaries.
