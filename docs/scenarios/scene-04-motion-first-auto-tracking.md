# Scene 04: Motion-First Auto Tracking

## Goal

Automatically select the most violently moving object and feed it into tracker init without manual template capture.

## Topology

`video source -> detector + denseoptflow(flow shm) -> pyengine/python_script(motion_selector) -> tracking -> viz.track(optional)`

## Prerequisites

- `f8.implayer` or `f8.screencap`
- `f8.dl.detector`
- `f8.cvkit.denseoptflow`
- `f8.pyengine` with `f8.python_script`
- `f8.cvkit.tracking`
- Optional: `f8.viz.track`

## Steps

1. Add and start source + CV services: `detector`, `denseoptflow`, `tracking`.
2. Set the same SHM for all three:
   - `detector.shmName`
   - `denseoptflow.inputShmName`
   - `tracking.shmName`
3. In `tracking`, set `initSelect=highest_score`.
4. Add a `f8.pyengine` service and add `f8.python_script` node named `motion_selector`.
5. Edit `motion_selector` ports:
   - Data in: `detections`
   - Data out: `selected`
   - State: `flowShm` (string)
6. Paste script from:
   - `docs/scenarios/scripts/motion_selector.py`
7. Wire data edges:
   - `detector.detections -> motion_selector.detections`
   - `motion_selector.selected -> tracking.initBox`
   - Optional: `tracking.tracking -> viz.track.detections`
8. Set `motion_selector.flowShm = denseoptflow.flowShmName`.
9. Optional TrackViz flow overlay settings:
   - `showDenseFlow = true`
   - `showSparseFlow = true` (if sparse JSON flow is also connected)
   - `denseFlowMode = hsv | arrows`
10. Run and verify tracker starts without manual template capture.

## Runtime Defaults (from script)

- `MAX_FRAME_GAP = 2`
- `MIN_FLOW_PIXELS_IN_ROI = 64`
- `MIN_SCORE = 0.05`
- `EMIT_MIN_INTERVAL_MS = 120`

## Validation

- Tracker can initialize from selected moving target automatically.
- When two objects exist, higher motion score wins.
- Camera shake is suppressed by `roi_mean_mag - global_mean_mag` baseline.
- If tracker is active, extra init payloads are ignored by `f8.cvkit.tracking`.
- When tracker is lost, next valid selected target can re-init tracking.

## Troubleshooting

- No selection:
  - reduce `MIN_SCORE` in script
  - reduce `MIN_FLOW_PIXELS_IN_ROI`
  - increase `denseoptflow.computeScale`
- Jitter or wrong picks:
  - increase `detector.inferEveryN` stability tuning
  - increase `EMIT_MIN_INTERVAL_MS`
- Frequent misses:
  - verify `frameId` gap between detection and flow streams
  - check SHM is identical across all nodes
