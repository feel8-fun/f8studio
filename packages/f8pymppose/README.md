# f8pymppose

Feel8 MediaPipe Pose runtime service.

Service class:
- `f8.mp.pose`

Output schema:
- `f8visionDetections/1` on `detections`
- UDP-skeleton-compatible JSON list on `skeletons` (for `f8.skeleton3d`)

## Coordinate and protocol contract

- `detections` (`f8visionDetections/1`)
  - Uses MediaPipe `pose_landmarks` (image landmarks).
  - `keypoints[].x/.y` are **pixel coordinates** (not normalized) for TrackViz rendering.
  - `keypoints[].z` follows MediaPipe image-landmark depth convention.
  - `skeletonProtocol` is `mediapipe_pose_33` (payload level and detection level).

- `skeletons` (UDP-skeleton-compatible list)
  - Prefers MediaPipe `pose_world_landmarks` for 3D output.
  - World coordinates are emitted as **Y-up** (`y = -world_y`).
  - If world landmarks are unavailable, falls back to normalized image-space coordinates.
  - Bone `rot` is estimated from neighbor links defined by the `mediapipe_pose_33` skeleton graph.
