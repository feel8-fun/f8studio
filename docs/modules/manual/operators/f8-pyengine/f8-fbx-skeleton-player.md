## When to Use

Play an animated FBX armature from PyEngine and connect its `skeletons` data output directly to `3D Viz.skeletons` on the Studio service. The player samples the file's animation at its original frame rate and outputs bone positions and rotations in Y-up world coordinates. Bone parent links become skeleton lines in 3D Viz.

## Common Wiring Patterns

Set `FBX Path` to an absolute path visible to the PyEngine process. Blender must be installed on the PyEngine machine. The player finds it on `PATH` or under `Program Files\Blender Foundation` on Windows; otherwise set `Blender Path` to the Blender executable. Import runs once in the background when the output is first requested. Loading and import errors are written to the PyEngine log.

Connect `Tick.elapsedSec` to `FBX Skeleton Player.timeSec`. A time input is required; without one the player does not load the file or output a pose. The incoming time is in seconds from the Tick entrypoint's start. With `Loop` enabled, the frame position is `timeSec % sequenceLength`; with `Loop` disabled, the player stops outputting poses once `timeSec >= sequenceLength`. The clip length is its frame count divided by its frame rate. Transform the incoming time to pause, seek, or change speed.

The 3D Viz cross-service sampling interval determines how often the output is observed; set `upstreamSampleIntervalMs` to about 33 for a 30 FPS clip when full frame-rate inspection is needed.

## Pitfalls / Gotchas

- The FBX path and Blender executable must be accessible on the machine running PyEngine.
- A connected time input is required before import begins or any pose is emitted.
- With looping disabled, reaching the end of the clip produces no further poses.
