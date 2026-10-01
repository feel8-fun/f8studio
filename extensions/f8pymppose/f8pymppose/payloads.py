from __future__ import annotations

import math
from typing import Any, Callable

from .config import DEFAULT_SKELETON_SOURCE, SkeletonSource
from .constants import (
    DETECTION_SCHEMA_VERSION,
    MEDIAPIPE_POSE_33_EDGES,
    MEDIAPIPE_POSE_33_LANDMARK_NAMES,
    POSE_MODEL_ID,
    POSE_TASK,
    SKELETON_MODEL_NAME,
    SKELETON_PROTOCOL_MEDIAPIPE_POSE_33,
    SKELETON_SCHEMA,
    SKELETON_TYPE_BINARY,
)

_IDENTITY_QUATERNION: list[float] = [1.0, 0.0, 0.0, 0.0]

PoseKeypoint = dict[str, float | None]
PosePosition = tuple[float, float, float]


def _build_neighbors_by_index() -> tuple[tuple[int, ...], ...]:
    count = len(MEDIAPIPE_POSE_33_LANDMARK_NAMES)
    neighbors: list[list[int]] = [[] for _ in range(count)]
    for edge_i, edge_j in MEDIAPIPE_POSE_33_EDGES:
        if edge_i < 0 or edge_j < 0 or edge_i >= count or edge_j >= count:
            continue
        neighbors[edge_i].append(edge_j)
        neighbors[edge_j].append(edge_i)
    return tuple(tuple(items) for items in neighbors)


_MEDIAPIPE_POSE_NEIGHBORS: tuple[tuple[int, ...], ...] = _build_neighbors_by_index()


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def _quat_from_y_axis_to_direction(dx: float, dy: float, dz: float) -> list[float]:
    base_x, base_y, base_z = 0.0, 1.0, 0.0
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 1e-8:
        return list(_IDENTITY_QUATERNION)
    tx = dx / norm
    ty = dy / norm
    tz = dz / norm

    dot = base_x * tx + base_y * ty + base_z * tz
    if dot >= 1.0 - 1e-8:
        return list(_IDENTITY_QUATERNION)
    if dot <= -1.0 + 1e-8:
        return [0.0, 1.0, 0.0, 0.0]

    cx = base_y * tz - base_z * ty
    cy = base_z * tx - base_x * tz
    cz = base_x * ty - base_y * tx
    qw = 1.0 + dot
    q_norm = math.sqrt(qw * qw + cx * cx + cy * cy + cz * cz)
    if q_norm <= 1e-8:
        return list(_IDENTITY_QUATERNION)
    return [qw / q_norm, cx / q_norm, cy / q_norm, cz / q_norm]


def _bone_orientation_quaternion(
    *,
    index: int,
    positions: list[tuple[float, float, float] | None],
) -> list[float]:
    if index < 0 or index >= len(_MEDIAPIPE_POSE_NEIGHBORS):
        return list(_IDENTITY_QUATERNION)
    origin = positions[index]
    if origin is None:
        return list(_IDENTITY_QUATERNION)
    ox, oy, oz = origin

    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    count = 0
    for neighbor_index in _MEDIAPIPE_POSE_NEIGHBORS[index]:
        if neighbor_index < 0 or neighbor_index >= len(positions):
            continue
        neighbor = positions[neighbor_index]
        if neighbor is None:
            continue
        nx, ny, nz = neighbor
        vx = nx - ox
        vy = ny - oy
        vz = nz - oz
        v_norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if v_norm <= 1e-8:
            continue
        sum_x += vx / v_norm
        sum_y += vy / v_norm
        sum_z += vz / v_norm
        count += 1

    if count <= 0:
        return list(_IDENTITY_QUATERNION)
    return _quat_from_y_axis_to_direction(sum_x, sum_y, sum_z)


def should_run_inference(last_infer_frame_id: int | None, frame_id: int, infer_every_n: int) -> bool:
    if last_infer_frame_id is None:
        return True
    return int(frame_id) - int(last_infer_frame_id) >= int(max(1, infer_every_n))


def _landmark_visibility(landmark: Any) -> float:
    raw_visibility = landmark.visibility
    if raw_visibility is None:
        return 1.0
    return float(raw_visibility)


def _hidden_pose_keypoint(*, visibility: float) -> PoseKeypoint:
    return {"x": None, "y": None, "z": None, "score": visibility}


def _image_pose_keypoint(landmark: Any, *, width: int, height: int) -> PoseKeypoint | None:
    raw_x = landmark.x
    raw_y = landmark.y
    raw_z = landmark.z
    if raw_x is None or raw_y is None:
        return None
    return {
        "x": _clamp(float(raw_x), 0.0, float(max(0, width - 1))),
        "y": _clamp(float(raw_y), 0.0, float(max(0, height - 1))),
        "z": float(raw_z) if raw_z is not None else 0.0,
        "score": 0.0,
    }


def _world_pose_keypoint(landmark: Any) -> PoseKeypoint | None:
    raw_x = landmark.x
    raw_y = landmark.y
    raw_z = landmark.z
    if raw_x is None or raw_y is None or raw_z is None:
        return None
    return {"x": float(raw_x), "y": float(raw_y), "z": float(raw_z), "score": 0.0}


def _extract_pose_keypoints_from_landmarks(
    landmarks: list[Any],
    *,
    visibility_threshold: float,
    build_visible_keypoint: Callable[[Any], PoseKeypoint | None],
) -> list[PoseKeypoint]:
    keypoints: list[PoseKeypoint] = []
    for landmark in landmarks:
        visibility = _landmark_visibility(landmark)
        if visibility < float(visibility_threshold):
            keypoints.append(_hidden_pose_keypoint(visibility=visibility))
            continue

        keypoint = build_visible_keypoint(landmark)
        if keypoint is None:
            keypoints.append(_hidden_pose_keypoint(visibility=visibility))
            continue

        keypoint["score"] = visibility
        keypoints.append(keypoint)
    return keypoints


def _pose_landmarks_from_result(result: Any) -> list[Any]:
    if result is None:
        return []
    pose_landmarks = result.pose_landmarks
    if pose_landmarks is None:
        return []

    if not isinstance(pose_landmarks, list) or not pose_landmarks:
        return []
    first_pose = pose_landmarks[0]
    if not isinstance(first_pose, list):
        return []
    return [x for x in first_pose]


def _pose_world_landmarks_from_result(result: Any) -> list[Any]:
    if result is None:
        return []
    pose_world_landmarks = result.pose_world_landmarks
    if pose_world_landmarks is None:
        return []
    if not isinstance(pose_world_landmarks, list) or not pose_world_landmarks:
        return []
    first_pose = pose_world_landmarks[0]
    if not isinstance(first_pose, list):
        return []
    return [x for x in first_pose]


def extract_pose_keypoints(
    result: Any, *, width: int, height: int, visibility_threshold: float
) -> list[dict[str, float | None]]:
    landmarks = _pose_landmarks_from_result(result)
    if not landmarks:
        return []
    return _extract_pose_keypoints_from_landmarks(
        landmarks,
        visibility_threshold=visibility_threshold,
        build_visible_keypoint=lambda landmark: _image_pose_keypoint(landmark, width=width, height=height),
    )


def extract_pose_world_keypoints(result: Any, *, visibility_threshold: float) -> list[dict[str, float | None]]:
    landmarks = _pose_world_landmarks_from_result(result)
    if not landmarks:
        return []
    return _extract_pose_keypoints_from_landmarks(
        landmarks,
        visibility_threshold=visibility_threshold,
        build_visible_keypoint=_world_pose_keypoint,
    )


def compute_bbox_from_keypoints(
    keypoints: list[dict[str, float | None]], *, width: int, height: int
) -> list[int] | None:
    xs: list[float] = []
    ys: list[float] = []
    for kp in keypoints:
        x = kp["x"]
        y = kp["y"]
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    if not xs or not ys:
        return None

    x1 = int(_clamp(min(xs), 0.0, float(max(0, width - 1))))
    y1 = int(_clamp(min(ys), 0.0, float(max(0, height - 1))))
    x2 = int(_clamp(max(xs), 0.0, float(max(0, width - 1))))
    y2 = int(_clamp(max(ys), 0.0, float(max(0, height - 1))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _person_score(keypoints: list[dict[str, float | None]]) -> float:
    scores: list[float] = []
    for kp in keypoints:
        score = kp["score"]
        x = kp["x"]
        y = kp["y"]
        if score is None or x is None or y is None:
            continue
        scores.append(float(score))
    if not scores:
        return 0.0
    return float(sum(scores) / float(len(scores)))


def build_pose_detection_payload(
    *,
    frame_id: int,
    ts_ms: int,
    width: int,
    height: int,
    keypoints: list[dict[str, float | None]],
) -> dict[str, Any]:
    bbox = compute_bbox_from_keypoints(keypoints, width=width, height=height)
    detections: list[dict[str, Any]] = []
    if bbox is not None:
        detections.append(
            {
                "id": 1,
                "cls": "person",
                "score": _person_score(keypoints),
                "bbox": bbox,
                "keypoints": keypoints,
                "skeletonProtocol": SKELETON_PROTOCOL_MEDIAPIPE_POSE_33,
            }
        )

    return {
        "schemaVersion": DETECTION_SCHEMA_VERSION,
        "frameId": int(frame_id),
        "tsMs": int(ts_ms),
        "width": int(width),
        "height": int(height),
        "model": POSE_MODEL_ID,
        "task": POSE_TASK,
        "skeletonProtocol": SKELETON_PROTOCOL_MEDIAPIPE_POSE_33,
        "detections": detections,
    }


def _world_position_from_keypoint(world_keypoint: PoseKeypoint) -> PosePosition | None:
    world_x = world_keypoint["x"]
    world_y = world_keypoint["y"]
    world_z = world_keypoint["z"]
    if world_x is None or world_y is None or world_z is None:
        return None
    return (float(world_x), -float(world_y), float(world_z))


def _camera_position_from_keypoint(image_keypoint: PoseKeypoint) -> PosePosition | None:
    image_x = image_keypoint["x"]
    image_y = image_keypoint["y"]
    image_z = image_keypoint["z"]
    if image_x is None or image_y is None or image_z is None:
        return None
    return (float(image_x), -float(image_y), float(image_z))


def _build_pose_positions(
    *,
    keypoints: list[PoseKeypoint],
    world_keypoints: list[PoseKeypoint] | None,
    skeleton_source: SkeletonSource,
) -> list[PosePosition | None]:
    landmark_count = len(MEDIAPIPE_POSE_33_LANDMARK_NAMES)
    if skeleton_source == "world" and world_keypoints is not None:
        positions = [_world_position_from_keypoint(pt) for pt in world_keypoints[:landmark_count]]
    else:
        positions = [_camera_position_from_keypoint(pt) for pt in keypoints[:landmark_count]]

    if len(positions) < landmark_count:
        positions.extend([None] * (landmark_count - len(positions)))
    return positions


def build_pose_skeleton_payload(
    *,
    frame_id: int,
    ts_ms: int,
    keypoints: list[PoseKeypoint],
    world_keypoints: list[PoseKeypoint] | None,
    skeleton_source: SkeletonSource = DEFAULT_SKELETON_SOURCE,
) -> dict[str, Any]:
    positions = _build_pose_positions(
        keypoints=keypoints,
        world_keypoints=world_keypoints,
        skeleton_source=skeleton_source,
    )

    bones: list[dict[str, Any]] = []
    for index, name in enumerate(MEDIAPIPE_POSE_33_LANDMARK_NAMES):
        position = positions[index]
        if position is None:
            continue
        x, y, z = position
        bones.append(
            {
                "name": name,
                "pos": [x, y, z],
                "rot": _bone_orientation_quaternion(index=index, positions=positions),
            }
        )
    return {
        "type": SKELETON_TYPE_BINARY,
        "schema": SKELETON_SCHEMA,
        "modelName": SKELETON_MODEL_NAME,
        "name": SKELETON_MODEL_NAME,
        "timestampMs": int(ts_ms),
        "frameId": int(frame_id),
        "boneCount": len(bones),
        "bones": bones,
        "trailer": None,
        "skeletonProtocol": SKELETON_PROTOCOL_MEDIAPIPE_POSE_33,
    }
