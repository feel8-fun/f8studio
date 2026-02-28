# Scene 01: CVKit Template Tracking

## Script

[Download JSON](scripts/scene-01-cvkit_template_tracking.json)

## Goal

Use `IM Player + CVKit Template Match + CVKit Tracking` to keep a target locked in video, then convert tracking motion into smooth `TCode` output.

## Main Pipeline

`IM Player -> CVKit Template Match -> CVKit Tracking -> Python Script (Tracking -> XY) -> Envelope/Filter/Map -> TCode -> Serial Out`

## Steps

1. Import the script and start required services (`IM Player`, `CVKit Template Match`, `CVKit Tracking`, `PyEngine`).
2. Select your source video in `IM Player` and verify `videoShmName` is propagated downstream.
3. Ensure `CVKit Template Match.shmName` and `CVKit Tracking.shmName` are identical.
4. Keep the control link `CVKit Tracking.isNotTracking -> CVKit Template Match.active` to avoid repeated re-init.
5. Verify numeric flow through `Tracking -> XY` and `Envelope -> 1EUR Filter -> Range Map -> Rate Limiter`.
6. Configure serial settings and enable `Serial Out`; use `TrackViz` and `WaveViz` for monitoring.

## Key Parameters

- `Tick.tickMs`: `50` (overall update cadence)
- `TCode.intervalMs`: `50` (keep aligned with Tick)
- `Range Map.outMax`: `0.7` (output ceiling)
- `Serial Out.port`: `COM4` in sample (change to your actual device)

## Validation

- `CVKit Tracking` continuously emits `tracking`.
- `WaveViz` is smooth, without major spikes.
- `TCode` and `Serial Out` stream continuously; `TrackViz` matches expected tracking behavior.

## Image Placeholders

- `docs/media/images/scenarios/scene-01-overview.png`
- `docs/media/images/scenarios/scene-01-tracking.png`
