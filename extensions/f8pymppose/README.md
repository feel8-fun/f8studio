# f8pymppose

Feel8 MediaPipe Pose runtime service.

Service class:
- `f8.mp.pose`

Input:
- Zenoh latest-frame video via the typed `video` data input port.

Output schema:
- `f8visionDetections/1` on `detections`
- UDP-skeleton-compatible JSON list on `skeletons` (for `f8.skeleton3d`)

## Coordinate and protocol contract

- `detections` (`f8visionDetections/1`)
  - Uses MediaPipe `pose_landmarks` (image landmarks).
  - `keypoints[].x/.y` currently mirror the incoming image-landmark coordinates after clamping to the image bounds.
  - The service does not currently scale normalized landmark coordinates into pixel space.
  - `keypoints[].z` follows MediaPipe image-landmark depth convention.
  - `skeletonProtocol` is `mediapipe_pose_33` (payload level and detection level).

- `skeletons` (UDP-skeleton-compatible list)
  - Controlled by state `skeletonSource`:
    - `camera` (default): use MediaPipe `pose_landmarks` directly, flip Y (`y = -image_y`), and prefer `pose_world_landmarks.z` as depth when world landmarks are available.
    - `world`: use MediaPipe `pose_world_landmarks` directly and convert to **Y-up** (`y = -world_y`).
  - In `world` mode, if world landmarks are unavailable, it falls back to the `camera` mapping above.
  - Bone `rot` is estimated from neighbor links defined by the `mediapipe_pose_33` skeleton graph.
