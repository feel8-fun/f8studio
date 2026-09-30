from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f8pysdk.resource_paths import model_root


@dataclass(frozen=True)
class PoseRuntimeConfig:
    model_complexity: str
    min_detection_confidence: float
    min_tracking_confidence: float


@dataclass(frozen=True)
class TasksModelSpec:
    filename: str
    url: str


def tasks_model_spec_for_complexity(model_complexity: str) -> TasksModelSpec:
    complexity = str(model_complexity or "").strip().lower()
    if complexity == "lite":
        return TasksModelSpec(
            filename="pose_landmarker_lite.task",
            url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        )
    if complexity == "full":
        return TasksModelSpec(
            filename="pose_landmarker_full.task",
            url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        )
    return TasksModelSpec(
        filename="pose_landmarker_heavy.task",
        url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    )


def _default_pose_model_dir() -> Path:
    return model_root() / "mediapipe"


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
    spec = tasks_model_spec_for_complexity(complexity)
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


class TasksPoseRuntime:
    def __init__(self, *, mediapipe_module: Any, landmarker: Any) -> None:
        self._mp = mediapipe_module
        self._landmarker = landmarker

    def process(self, frame_rgb: Any, *, timestamp_ms: int) -> Any:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect_for_video(image, int(timestamp_ms))

    def close(self) -> None:
        self._landmarker.close()


def _create_tasks_pose_runtime(*, config: PoseRuntimeConfig, mediapipe_module: Any) -> Any:
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
    return TasksPoseRuntime(mediapipe_module=mediapipe_module, landmarker=landmarker)


def create_pose_runtime(config: PoseRuntimeConfig) -> Any:
    import mediapipe as mp  # type: ignore[import-not-found]

    return _create_tasks_pose_runtime(config=config, mediapipe_module=mp)
