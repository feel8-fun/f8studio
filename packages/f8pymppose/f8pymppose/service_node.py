from __future__ import annotations

import asyncio
import logging
import math
import time
import traceback
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import ServiceNode
from f8pysdk.shm.video import VideoShmReader

from .constants import (
    DETECTION_SCHEMA_VERSION,
    MEDIAPIPE_POSE_33_EDGES,
    MEDIAPIPE_POSE_33_LANDMARK_NAMES,
    POSE_MODEL_ID,
    POSE_TASK,
    POSE_SERVICE_CLASS,
    SKELETON_MODEL_NAME,
    SKELETON_PROTOCOL_MEDIAPIPE_POSE_33,
    SKELETON_SCHEMA,
    SKELETON_TYPE_BINARY,
)

log = logging.getLogger(__name__)

_IDENTITY_QUATERNION: list[float] = [1.0, 0.0, 0.0, 0.0]


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


def _coerce_int(v: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        out = int(v)
    except (TypeError, ValueError):
        out = int(default)
    if out < minimum:
        return int(minimum)
    if out > maximum:
        return int(maximum)
    return int(out)


def _coerce_float(v: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        out = float(default)
    if out < minimum:
        return float(minimum)
    if out > maximum:
        return float(maximum)
    return float(out)


def _coerce_str(v: Any, *, default: str) -> str:
    if v is None:
        return str(default)
    text = str(v).strip()
    if not text:
        return str(default)
    return text


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


def extract_pose_keypoints(result: Any, *, width: int, height: int, visibility_threshold: float) -> list[dict[str, float | None]]:
    landmarks = _pose_landmarks_from_result(result)
    if not landmarks:
        return []

    out: list[dict[str, float | None]] = []
    for landmark in landmarks:
        raw_visibility = landmark.visibility
        visibility = float(raw_visibility) if raw_visibility is not None else 1.0
        is_visible = visibility >= float(visibility_threshold)
        raw_x = landmark.x
        raw_y = landmark.y
        raw_z = landmark.z
        if is_visible:
            if raw_x is None or raw_y is None:
                out.append({"x": None, "y": None, "z": None, "score": visibility})
                continue
            px = _clamp(float(raw_x) * float(width), 0.0, float(max(0, width - 1)))
            py = _clamp(float(raw_y) * float(height), 0.0, float(max(0, height - 1)))
            pz = float(raw_z) if raw_z is not None else 0.0
            out.append({"x": px, "y": py, "z": pz, "score": visibility})
        else:
            out.append({"x": None, "y": None, "z": None, "score": visibility})
    return out


def extract_pose_world_keypoints(result: Any, *, visibility_threshold: float) -> list[dict[str, float | None]]:
    landmarks = _pose_world_landmarks_from_result(result)
    if not landmarks:
        return []

    out: list[dict[str, float | None]] = []
    for landmark in landmarks:
        raw_visibility = landmark.visibility
        visibility = float(raw_visibility) if raw_visibility is not None else 1.0
        is_visible = visibility >= float(visibility_threshold)
        raw_x = landmark.x
        raw_y = landmark.y
        raw_z = landmark.z
        if is_visible:
            if raw_x is None or raw_y is None or raw_z is None:
                out.append({"x": None, "y": None, "z": None, "score": visibility})
                continue
            out.append({"x": float(raw_x), "y": float(raw_y), "z": float(raw_z), "score": visibility})
        else:
            out.append({"x": None, "y": None, "z": None, "score": visibility})
    return out


def compute_bbox_from_keypoints(keypoints: list[dict[str, float | None]], *, width: int, height: int) -> list[int] | None:
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


def build_pose_skeleton_payload(
    *,
    frame_id: int,
    ts_ms: int,
    keypoints: list[dict[str, float | None]],
    world_keypoints: list[dict[str, float | None]] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    norm_w = float(width) if int(width) > 0 else 1.0
    norm_h = float(height) if int(height) > 0 else 1.0
    positions: list[tuple[float, float, float] | None] = [None for _ in MEDIAPIPE_POSE_33_LANDMARK_NAMES]
    for index, _name in enumerate(MEDIAPIPE_POSE_33_LANDMARK_NAMES):
        world_position: tuple[float, float, float] | None = None
        if world_keypoints is not None and index < len(world_keypoints):
            world_keypoint = world_keypoints[index]
            world_x = world_keypoint["x"]
            world_y = world_keypoint["y"]
            world_z = world_keypoint["z"]
            if world_x is not None and world_y is not None and world_z is not None:
                world_position = (float(world_x), -float(world_y), float(world_z))
        if world_position is not None:
            positions[index] = world_position
            continue

        if index >= len(keypoints):
            continue
        image_keypoint = keypoints[index]
        image_x = image_keypoint["x"]
        image_y = image_keypoint["y"]
        image_z = image_keypoint["z"]
        if image_x is None or image_y is None or image_z is None:
            continue
        positions[index] = (float(image_x) / norm_w, float(image_y) / norm_h, float(image_z))

    bones: list[dict[str, Any]] = []
    for index, name in enumerate(MEDIAPIPE_POSE_33_LANDMARK_NAMES):
        if index >= len(positions):
            break
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


@dataclass(frozen=True)
class _PoseRuntimeConfig:
    model_complexity: str
    min_detection_confidence: float
    min_tracking_confidence: float


@dataclass(frozen=True)
class _TasksModelSpec:
    filename: str
    url: str


def _tasks_model_spec_for_complexity(model_complexity: str) -> _TasksModelSpec:
    complexity = str(model_complexity or "").strip().lower()
    if complexity == "lite":
        return _TasksModelSpec(
            filename="pose_landmarker_lite.task",
            url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        )
    if complexity == "full":
        return _TasksModelSpec(
            filename="pose_landmarker_full.task",
            url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        )
    return _TasksModelSpec(
        filename="pose_landmarker_heavy.task",
        url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    )


def _default_pose_model_dir() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append((Path.cwd() / "services" / "f8" / "mp" / "pose" / "models").resolve())
    except (OSError, RuntimeError, ValueError):
        pass
    try:
        root = Path(__file__).resolve().parents[3]
        candidates.append((root / "services" / "f8" / "mp" / "pose" / "models").resolve())
    except (OSError, RuntimeError, ValueError):
        pass
    if candidates:
        return candidates[0]
    return Path.cwd().resolve()


def _download_model_asset(*, url: str, dst_path: Path, timeout_s: float = 30.0) -> None:
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    tmp_path.replace(dst_path)


def _ensure_tasks_model_asset(model_complexity: str) -> Path:
    complexity = str(model_complexity or "").strip().lower()
    if complexity not in ("lite", "full", "heavy"):
        complexity = "full"
    spec = _tasks_model_spec_for_complexity(complexity)
    model_dir = _default_pose_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / spec.filename
    if path.is_file() and path.stat().st_size > 0:
        return path
    try:
        _download_model_asset(url=spec.url, dst_path=path)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Failed to download MediaPipe pose model: complexity={complexity}, "
            f"url={spec.url}, dst={path}, error={type(exc).__name__}: {exc}"
        ) from exc
    return path


class _TasksPoseRuntime:
    def __init__(self, *, mediapipe_module: Any, landmarker: Any) -> None:
        self._mp = mediapipe_module
        self._landmarker = landmarker

    def process(self, frame_rgb: Any, *, timestamp_ms: int) -> Any:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect_for_video(image, int(timestamp_ms))

    def close(self) -> None:
        self._landmarker.close()


def _create_tasks_pose_runtime(*, config: _PoseRuntimeConfig, mediapipe_module: Any) -> Any:
    model_asset_path = _ensure_tasks_model_asset(config.model_complexity)
    base_options = mediapipe_module.tasks.BaseOptions(model_asset_path=str(model_asset_path))
    options = mediapipe_module.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mediapipe_module.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=float(config.min_detection_confidence),
        min_pose_presence_confidence=float(config.min_detection_confidence),
        min_tracking_confidence=float(config.min_tracking_confidence),
        output_segmentation_masks=False,
    )
    landmarker = mediapipe_module.tasks.vision.PoseLandmarker.create_from_options(options)
    return _TasksPoseRuntime(mediapipe_module=mediapipe_module, landmarker=landmarker)


def _create_pose_runtime(config: _PoseRuntimeConfig) -> Any:
    import mediapipe as mp  # type: ignore[import-not-found]

    return _create_tasks_pose_runtime(config=config, mediapipe_module=mp)


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


class _RollingWindow:
    def __init__(self, *, window_ms: int) -> None:
        self.window_ms = int(window_ms)
        self._q: deque[tuple[int, float]] = deque()
        self._sum = 0.0

    def push(self, ts_ms: int, value: float) -> None:
        self._q.append((int(ts_ms), float(value)))
        self._sum += float(value)
        self.prune(ts_ms)

    def prune(self, now_ms: int) -> None:
        cutoff = int(now_ms) - int(self.window_ms)
        while self._q and int(self._q[0][0]) < cutoff:
            _, v = self._q.popleft()
            self._sum -= float(v)

    def mean(self, now_ms: int) -> float | None:
        self.prune(now_ms)
        if not self._q:
            return None
        return float(self._sum) / float(len(self._q))

    def count(self, now_ms: int) -> int:
        self.prune(now_ms)
        return int(len(self._q))


class _Telemetry:
    def __init__(self) -> None:
        self.interval_ms = 1000
        self.window_ms = 2000
        self._last_emit_ms = 0
        self._frames = _RollingWindow(window_ms=self.window_ms)
        self._infer = _RollingWindow(window_ms=self.window_ms)
        self._total = _RollingWindow(window_ms=self.window_ms)
        self._dup = _RollingWindow(window_ms=self.window_ms)

    def set_config(self, *, interval_ms: int, window_ms: int) -> None:
        self.interval_ms = int(max(0, interval_ms))
        self.window_ms = int(max(100, window_ms))
        self._frames.window_ms = self.window_ms
        self._infer.window_ms = self.window_ms
        self._total.window_ms = self.window_ms
        self._dup.window_ms = self.window_ms

    def observe_frame(self, *, ts_ms: int, infer_ms: float, total_ms: float, dup_skipped: int) -> None:
        self._frames.push(ts_ms, 1.0)
        self._infer.push(ts_ms, infer_ms)
        self._total.push(ts_ms, total_ms)
        self._dup.push(ts_ms, float(dup_skipped))

    def should_emit(self, now_ms: int) -> bool:
        if self.interval_ms <= 0:
            return False
        if self._last_emit_ms <= 0:
            return True
        return int(now_ms) - int(self._last_emit_ms) >= int(self.interval_ms)

    def mark_emitted(self, now_ms: int) -> None:
        self._last_emit_ms = int(now_ms)

    def summary(self, *, now_ms: int, node_id: str, shm_name: str) -> dict[str, Any]:
        frames = self._frames.count(now_ms)
        fps: float | None = None
        if self.window_ms > 0:
            fps = float(frames) * 1000.0 / float(self.window_ms)
        return {
            "schemaVersion": "f8mpTelemetry/1",
            "tsMs": int(now_ms),
            "nodeId": str(node_id),
            "serviceClass": POSE_SERVICE_CLASS,
            "model": {
                "id": POSE_MODEL_ID,
                "task": POSE_TASK,
                "provider": "mediapipe-cpu",
            },
            "windowMs": int(self.window_ms),
            "source": {"shmName": str(shm_name)},
            "rates": {"fps": fps},
            "timingsMsAvg": {
                "infer": self._infer.mean(now_ms),
                "total": self._total.mean(now_ms),
            },
            "frameId": {"duplicatesSkippedAvg": self._dup.mean(now_ms)},
        }


class MediaPipePoseServiceNode(ServiceNode):
    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=["detections", "skeletons", "telemetry"],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._active = True
        self._config_loaded = False
        self._task: asyncio.Task[object] | None = None

        self._shm_name = ""
        self._infer_every_n = 1
        self._model_complexity = "full"
        self._min_detection_confidence = 0.5
        self._min_tracking_confidence = 0.5
        self._visibility_threshold = 0.5

        self._shm: VideoShmReader | None = None
        self._shm_open_name = ""
        self._pose_runtime: Any | None = None

        self._last_error_signature = ""
        self._last_error_repeats = 0
        self._last_infer_frame_id: int | None = None
        self._last_processed_frame_id: int | None = None
        self._dup_skipped_since_last_processed = 0
        self._telemetry = _Telemetry()

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        loop = asyncio.get_running_loop()
        loop.create_task(self._ensure_config_loaded(), name=f"f8mppose:init:{self.node_id}")
        self._task = loop.create_task(self._loop(), name=f"f8mppose:loop:{self.node_id}")

    async def close(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._close_shm()
        self._close_pose_runtime()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del value
        del ts_ms
        name = str(field or "").strip()
        await self._ensure_config_loaded()

        if name == "shmName":
            self._shm_name = _coerce_str(await self.get_state_value("shmName"), default=self._shm_name)
            await self._maybe_reopen_shm()
            return

        if name == "inferEveryN":
            self._infer_every_n = _coerce_int(
                await self.get_state_value("inferEveryN"),
                default=self._infer_every_n,
                minimum=1,
                maximum=10000,
            )
            return

        if name == "modelComplexity":
            complexity = _coerce_str(await self.get_state_value("modelComplexity"), default=self._model_complexity).lower()
            self._model_complexity = complexity if complexity in ("lite", "full", "heavy") else "full"
            await self._reset_pose_runtime()
            return

        if name == "minDetectionConfidence":
            self._min_detection_confidence = _coerce_float(
                await self.get_state_value("minDetectionConfidence"),
                default=self._min_detection_confidence,
                minimum=0.0,
                maximum=1.0,
            )
            await self._reset_pose_runtime()
            return

        if name == "minTrackingConfidence":
            self._min_tracking_confidence = _coerce_float(
                await self.get_state_value("minTrackingConfidence"),
                default=self._min_tracking_confidence,
                minimum=0.0,
                maximum=1.0,
            )
            await self._reset_pose_runtime()
            return

        if name == "visibilityThreshold":
            self._visibility_threshold = _coerce_float(
                await self.get_state_value("visibilityThreshold"),
                default=self._visibility_threshold,
                minimum=0.0,
                maximum=1.0,
            )
            return

        if name == "telemetryIntervalMs":
            self._telemetry.set_config(
                interval_ms=_coerce_int(
                    await self.get_state_value("telemetryIntervalMs"),
                    default=self._telemetry.interval_ms,
                    minimum=0,
                    maximum=60000,
                ),
                window_ms=self._telemetry.window_ms,
            )
            return

        if name == "telemetryWindowMs":
            self._telemetry.set_config(
                interval_ms=self._telemetry.interval_ms,
                window_ms=_coerce_int(
                    await self.get_state_value("telemetryWindowMs"),
                    default=self._telemetry.window_ms,
                    minimum=100,
                    maximum=60000,
                ),
            )
            return

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._shm_name = _coerce_str(
            await self.get_state_value("shmName"),
            default=str(self._initial_state.get("shmName") or ""),
        )
        self._infer_every_n = _coerce_int(
            await self.get_state_value("inferEveryN"),
            default=int(self._initial_state.get("inferEveryN") or 1),
            minimum=1,
            maximum=10000,
        )
        complexity = _coerce_str(
            await self.get_state_value("modelComplexity"),
            default=str(self._initial_state.get("modelComplexity") or "full"),
        ).lower()
        self._model_complexity = complexity if complexity in ("lite", "full", "heavy") else "full"
        self._min_detection_confidence = _coerce_float(
            await self.get_state_value("minDetectionConfidence"),
            default=float(self._initial_state.get("minDetectionConfidence") or 0.5),
            minimum=0.0,
            maximum=1.0,
        )
        self._min_tracking_confidence = _coerce_float(
            await self.get_state_value("minTrackingConfidence"),
            default=float(self._initial_state.get("minTrackingConfidence") or 0.5),
            minimum=0.0,
            maximum=1.0,
        )
        self._visibility_threshold = _coerce_float(
            await self.get_state_value("visibilityThreshold"),
            default=float(self._initial_state.get("visibilityThreshold") or 0.5),
            minimum=0.0,
            maximum=1.0,
        )
        self._telemetry.set_config(
            interval_ms=_coerce_int(
                await self.get_state_value("telemetryIntervalMs"),
                default=int(self._initial_state.get("telemetryIntervalMs") or 1000),
                minimum=0,
                maximum=60000,
            ),
            window_ms=_coerce_int(
                await self.get_state_value("telemetryWindowMs"),
                default=int(self._initial_state.get("telemetryWindowMs") or 2000),
                minimum=100,
                maximum=60000,
            ),
        )
        self._config_loaded = True

    async def _set_last_error(self, message: str) -> None:
        await self.set_state("lastError", str(message or ""))

    async def _record_exception(self, *, where: str, exc: Exception) -> None:
        signature = f"{where}:{type(exc).__name__}:{exc}"
        if signature == self._last_error_signature:
            self._last_error_repeats += 1
        else:
            self._last_error_signature = signature
            self._last_error_repeats = 1
        if self._last_error_repeats != 1 and self._last_error_repeats % 100 != 0:
            return
        await self._set_last_error(
            f"{where} failed with {type(exc).__name__}: {exc}\n"
            f"repeat={self._last_error_repeats}\n"
            f"traceback:\n{traceback.format_exc()}"
        )

    async def _reset_pose_runtime(self) -> None:
        self._close_pose_runtime()
        self._last_error_signature = ""
        self._last_error_repeats = 0
        await self._set_last_error("")

    def _close_pose_runtime(self) -> None:
        if self._pose_runtime is None:
            return
        try:
            self._pose_runtime.close()
        except Exception as exc:
            log.exception("pose runtime close failed", exc_info=exc)
        self._pose_runtime = None

    async def _ensure_pose_runtime(self) -> None:
        if self._pose_runtime is not None:
            return
        config = _PoseRuntimeConfig(
            model_complexity=self._model_complexity,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._pose_runtime = _create_pose_runtime(config)

    async def _maybe_reopen_shm(self) -> None:
        want = self._resolve_shm_name()
        if want == self._shm_open_name:
            return
        self._close_shm()

    def _resolve_shm_name(self) -> str:
        return str(self._shm_name or "").strip()

    def _close_shm(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception as exc:
                log.exception("video shm close failed", exc_info=exc)
        self._shm = None
        self._shm_open_name = ""

    def _open_shm(self, shm_name: str) -> None:
        self._close_shm()
        shm = VideoShmReader(shm_name)
        shm.open(use_event=True)
        self._shm = shm
        self._shm_open_name = shm_name

    async def _loop(self) -> None:
        import numpy as np  # type: ignore[import-not-found]

        while True:
            try:
                await asyncio.sleep(0)
                if not self._active:
                    await asyncio.sleep(0.05)
                    continue

                await self._ensure_config_loaded()
                await self._ensure_pose_runtime()

                shm_name = self._resolve_shm_name()
                if not shm_name:
                    await asyncio.sleep(0.05)
                    continue

                if self._shm is None:
                    self._open_shm(shm_name)

                assert self._shm is not None
                t0 = time.perf_counter()
                self._shm.wait_new_frame(timeout_ms=10)
                header, payload = self._shm.read_latest_bgra()
                if header is None or payload is None:
                    continue

                frame_id_seen = int(header.frame_id)
                if self._last_processed_frame_id is not None and frame_id_seen == int(self._last_processed_frame_id):
                    self._dup_skipped_since_last_processed += 1
                    continue
                dup_skipped = int(self._dup_skipped_since_last_processed)
                self._dup_skipped_since_last_processed = 0

                if not should_run_inference(self._last_infer_frame_id, frame_id_seen, self._infer_every_n):
                    self._last_processed_frame_id = frame_id_seen
                    continue

                width = int(header.width)
                height = int(header.height)
                pitch = int(header.pitch)
                if width <= 0 or height <= 0 or pitch <= 0:
                    continue
                frame_bytes = int(pitch) * int(height)
                if len(payload) < frame_bytes:
                    continue
                self._last_processed_frame_id = frame_id_seen

                buf = np.frombuffer(payload, dtype=np.uint8)
                rows = buf.reshape((height, pitch))
                bgra = rows[:, : width * 4].reshape((height, width, 4))
                frame_bgr = bgra[:, :, 0:3]
                frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])

                t_infer0 = time.perf_counter()
                assert self._pose_runtime is not None
                result = self._pose_runtime.process(frame_rgb, timestamp_ms=int(header.ts_ms))
                keypoints = extract_pose_keypoints(
                    result,
                    width=width,
                    height=height,
                    visibility_threshold=self._visibility_threshold,
                )
                world_keypoints = extract_pose_world_keypoints(
                    result,
                    visibility_threshold=self._visibility_threshold,
                )
                payload_out = build_pose_detection_payload(
                    frame_id=frame_id_seen,
                    ts_ms=int(header.ts_ms),
                    width=width,
                    height=height,
                    keypoints=keypoints,
                )
                skeleton_payload = build_pose_skeleton_payload(
                    frame_id=frame_id_seen,
                    ts_ms=int(header.ts_ms),
                    keypoints=keypoints,
                    world_keypoints=world_keypoints,
                    width=width,
                    height=height,
                )
                await self.emit("detections", payload_out, ts_ms=int(header.ts_ms))
                await self.emit("skeletons", [skeleton_payload], ts_ms=int(header.ts_ms))
                t_infer1 = time.perf_counter()

                self._last_infer_frame_id = frame_id_seen
                now_ms = int(header.ts_ms)
                self._telemetry.observe_frame(
                    ts_ms=now_ms,
                    infer_ms=(t_infer1 - t_infer0) * 1000.0,
                    total_ms=(time.perf_counter() - t0) * 1000.0,
                    dup_skipped=dup_skipped,
                )
                if self._telemetry.should_emit(now_ms):
                    telemetry_payload = self._telemetry.summary(
                        now_ms=now_ms,
                        node_id=self.node_id,
                        shm_name=(self._shm_open_name or shm_name),
                    )
                    await self.emit("telemetry", telemetry_payload, ts_ms=now_ms)
                    self._telemetry.mark_emitted(now_ms)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_exception(where="loop", exc=exc)
                await asyncio.sleep(0.1)
