# Scene 02: GameMod Skeleton

## Script

[Download JSON](scripts/scene-02-gamemod_skeleton.json)

## Goal

Receive skeleton streams from `UDP Skeleton`, extract a distance feature in `Python Script`, map it to `TCode`, and send it through serial output with real-time visualization.

## Main Pipeline

`UDP Skeleton -> Python Script -> Envelope -> Range Map -> Rate Limiter -> TCode -> Serial Out`

## Steps

1. Import the script and start `PyEngine`.
2. Confirm `UDP Skeleton.port` matches your sender (`39540` in this script).
3. Select the target stream via `UDP Skeleton.selectedKey`.
4. Confirm timing links: `Tick 1.exec -> Python Script.exec` and `Tick 1.tickMs -> TCode 1.intervalMs`.
5. Validate value flow in `Python Script.out`, `Envelope 1.normalized`, and `Range Map 1.value`.
6. Set `Serial Out 1.port` and enable serial output.

## Key Parameters

- `Tick 1.tickMs`: `33` (~30Hz)
- `TCode 1.intervalMs`: `33` (recommended to match Tick)
- `Range Map 1.outMax`: `0.7`
- `Rate Limiter 1.maxRateUp/maxRateDown`: `2.0/2.0`

## Validation

- `3DViz` shows stable skeleton updates.
- `WaveViz` shows reasonable raw/envelope variation.
- `TCodeViz` and serial output are continuous and stable.

## Image Placeholders

- `docs/media/images/scenarios/scene-02-overview.png`
- `docs/media/images/scenarios/scene-02-3dviz.png`
