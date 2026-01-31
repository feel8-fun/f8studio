from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import ServiceNode
from f8pysdk.shm.video import VideoShmReader, default_video_shm_name

from .model_config import ModelSpec, build_model_index, load_model_spec
from .onnx_detectors import OnnxYoloDetector, PoseKeypoint
from .tracking import (
    TrackerKind,
    Track,
    associate_and_update_tracks,
    update_tracks_with_cv,
)
from .vision_utils import clamp_xyxy


def _default_weights_dir() -> Path:
    # Run from repo root by default (service.yml workdir is "../../../").
    # Keep this tolerant: if the path doesn't exist, user can override via state.weightsDir.
    return (Path.cwd() / "services" / "f8" / "onnx_tracker" / "weights").resolve()


def _coerce_int(v: Any, *, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        out = int(v)
    except Exception:
        out = int(default)
    if minimum is not None and out < minimum:
        out = int(minimum)
    if maximum is not None and out > maximum:
        out = int(maximum)
    return out


def _coerce_float(v: Any, *, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        out = float(v)
    except Exception:
        out = float(default)
    if minimum is not None and out < minimum:
        out = float(minimum)
    if maximum is not None and out > maximum:
        out = float(maximum)
    return out


def _coerce_str(v: Any, *, default: str = "") -> str:
    try:
        s = str(v) if v is not None else ""
    except Exception:
        s = ""
    s = s.strip()
    return s if s else default


def _coerce_bool(v: Any, *, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = _coerce_str(v).lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


class OnnxTrackerServiceNode(ServiceNode):
    """
    Runtime operator:
    - reads frames from VideoSHM (BGRA32)
    - runs ONNX YOLO detect/pose/obb at low frequency
    - uses OpenCV trackers at high frequency
    - emits results as a single data stream
    """

    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            # Service node exports a single data channel.
            data_out_ports=["detections"],
            state_fields=[s.name for s in (getattr(node, "stateFields", None) or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._active = True
        self._config_loaded = False
        self._task: asyncio.Task[object] | None = None

        self._weights_dir: Path = _default_weights_dir()
        self._model_yaml_path: str = ""
        self._model_id: str = ""
        self._ort_provider: Literal["auto", "cuda", "cpu"] = "auto"
        self._tracker_kind: TrackerKind = "kcf"
        self._detect_every_n: int = 5
        self._max_targets: int = 5
        self._iou_match: float = 0.3
        self._mismatch_iou: float = 0.2
        self._mismatch_patience: int = 3
        self._max_age: int = 30
        self._reinit_on_detect: bool = True
        self._conf_override: float = -1.0
        self._iou_override: float = -1.0
        self._source_service_id: str = ""
        self._shm_name: str = ""

        self._shm: VideoShmReader | None = None
        self._shm_open_name: str = ""

        self._model: ModelSpec | None = None
        self._detector: OnnxYoloDetector | None = None
        self._detector_yaml: Path | None = None
        self._last_error: str = ""

        self._tracks: list[Track] = []
        self._next_track_id = 1
        self._last_detect_frame_id: int | None = None
        self._need_reinit_trackers = False

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._ensure_config_loaded(), name=f"onnxtracker:init:{self.node_id}")
            self._task = loop.create_task(self._loop(), name=f"onnxtracker:loop:{self.node_id}")
        except Exception:
            pass

    async def close(self) -> None:
        t = self._task
        self._task = None
        if t is not None:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
        self._close_shm()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        self._active = bool(active)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        await self._ensure_config_loaded()

        if name == "weightsDir":
            self._weights_dir = Path(_coerce_str(await self.get_state("weightsDir"), default=str(self._weights_dir))).expanduser().resolve()
            await self._publish_model_index()
            await self._reset_detector()
        elif name == "modelId":
            self._model_id = _coerce_str(await self.get_state("modelId"), default=self._model_id)
            await self._reset_detector()
        elif name == "modelYamlPath":
            self._model_yaml_path = _coerce_str(await self.get_state("modelYamlPath"), default=self._model_yaml_path)
            await self._reset_detector()
        elif name == "ortProvider":
            v = _coerce_str(await self.get_state("ortProvider"), default=str(self._ort_provider)).lower()
            self._ort_provider = v if v in ("auto", "cuda", "cpu") else "auto"
            await self._reset_detector()
        elif name == "trackerKind":
            v = _coerce_str(await self.get_state("trackerKind"), default=str(self._tracker_kind)).lower()
            self._tracker_kind = v if v in ("none", "csrt", "kcf", "mosse") else "kcf"
            self._need_reinit_trackers = True
        elif name == "detectEveryN":
            self._detect_every_n = _coerce_int(await self.get_state("detectEveryN"), default=self._detect_every_n, minimum=1, maximum=10_000)
        elif name == "maxTargets":
            self._max_targets = _coerce_int(await self.get_state("maxTargets"), default=self._max_targets, minimum=1, maximum=1000)
        elif name == "iouMatch":
            self._iou_match = _coerce_float(await self.get_state("iouMatch"), default=self._iou_match, minimum=0.0, maximum=1.0)
        elif name == "mismatchIou":
            self._mismatch_iou = _coerce_float(await self.get_state("mismatchIou"), default=self._mismatch_iou, minimum=0.0, maximum=1.0)
        elif name == "mismatchPatience":
            self._mismatch_patience = _coerce_int(await self.get_state("mismatchPatience"), default=self._mismatch_patience, minimum=1, maximum=1000)
        elif name == "maxAge":
            self._max_age = _coerce_int(await self.get_state("maxAge"), default=self._max_age, minimum=1, maximum=100_000)
        elif name == "reinitOnDetect":
            self._reinit_on_detect = _coerce_bool(await self.get_state("reinitOnDetect"), default=self._reinit_on_detect)
        elif name == "confThreshold":
            self._conf_override = _coerce_float(await self.get_state("confThreshold"), default=self._conf_override)
            await self._reset_detector()
        elif name == "iouThreshold":
            self._iou_override = _coerce_float(await self.get_state("iouThreshold"), default=self._iou_override)
            await self._reset_detector()
        elif name == "sourceServiceId":
            self._source_service_id = _coerce_str(await self.get_state("sourceServiceId"), default=self._source_service_id)
            await self._maybe_reopen_shm()
        elif name == "shmName":
            self._shm_name = _coerce_str(await self.get_state("shmName"), default=self._shm_name)
            await self._maybe_reopen_shm()

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return

        self._weights_dir = Path(
            _coerce_str(await self.get_state("weightsDir"), default=str(self._initial_state.get("weightsDir") or _default_weights_dir()))
        ).expanduser().resolve()
        self._model_id = _coerce_str(await self.get_state("modelId"), default=str(self._initial_state.get("modelId") or ""))
        self._model_yaml_path = _coerce_str(
            await self.get_state("modelYamlPath"), default=str(self._initial_state.get("modelYamlPath") or "")
        )
        v = _coerce_str(await self.get_state("ortProvider"), default=str(self._initial_state.get("ortProvider") or "auto")).lower()
        self._ort_provider = v if v in ("auto", "cuda", "cpu") else "auto"

        v = _coerce_str(await self.get_state("trackerKind"), default=str(self._initial_state.get("trackerKind") or "kcf")).lower()
        self._tracker_kind = v if v in ("none", "csrt", "kcf", "mosse") else "kcf"

        self._detect_every_n = _coerce_int(await self.get_state("detectEveryN"), default=int(self._initial_state.get("detectEveryN") or 5), minimum=1)
        self._max_targets = _coerce_int(await self.get_state("maxTargets"), default=int(self._initial_state.get("maxTargets") or 5), minimum=1, maximum=1000)
        self._iou_match = _coerce_float(await self.get_state("iouMatch"), default=float(self._initial_state.get("iouMatch") or 0.3), minimum=0.0, maximum=1.0)
        self._mismatch_iou = _coerce_float(await self.get_state("mismatchIou"), default=float(self._initial_state.get("mismatchIou") or 0.2), minimum=0.0, maximum=1.0)
        self._mismatch_patience = _coerce_int(await self.get_state("mismatchPatience"), default=int(self._initial_state.get("mismatchPatience") or 3), minimum=1)
        self._max_age = _coerce_int(await self.get_state("maxAge"), default=int(self._initial_state.get("maxAge") or 30), minimum=1)
        self._reinit_on_detect = _coerce_bool(await self.get_state("reinitOnDetect"), default=bool(self._initial_state.get("reinitOnDetect", True)))
        self._conf_override = _coerce_float(await self.get_state("confThreshold"), default=float(self._initial_state.get("confThreshold") or -1.0))
        self._iou_override = _coerce_float(await self.get_state("iouThreshold"), default=float(self._initial_state.get("iouThreshold") or -1.0))

        self._source_service_id = _coerce_str(await self.get_state("sourceServiceId"), default=str(self._initial_state.get("sourceServiceId") or ""))
        self._shm_name = _coerce_str(await self.get_state("shmName"), default=str(self._initial_state.get("shmName") or ""))

        self._config_loaded = True
        await self._publish_model_index()

    async def _publish_model_index(self) -> None:
        idx = build_model_index(self._weights_dir)
        payload = [{"id": i.model_id, "name": i.display_name, "task": i.task, "yaml": str(i.yaml_path)} for i in idx]
        await self.set_state("availableModels", json.dumps(payload, ensure_ascii=False))

        if not self._model_id and idx:
            self._model_id = idx[0].model_id
            await self.set_state("modelId", self._model_id)

    async def _reset_detector(self) -> None:
        self._detector = None
        self._detector_yaml = None
        self._model = None
        self._last_error = ""
        await self.set_state("loadedModel", "")
        await self.set_state("lastError", "")
        await self.set_state("ortActiveProviders", "")

    async def _maybe_reopen_shm(self) -> None:
        want = self._resolve_shm_name()
        if want == self._shm_open_name:
            return
        self._close_shm()

    def _resolve_shm_name(self) -> str:
        shm = str(self._shm_name or "").strip()
        if shm:
            return shm
        sid = str(self._source_service_id or "").strip()
        if sid:
            return default_video_shm_name(sid)
        return ""

    def _close_shm(self) -> None:
        if self._shm:
            try:
                self._shm.close()
            except Exception:
                pass
        self._shm = None
        self._shm_open_name = ""

    def _open_shm(self, shm_name: str) -> bool:
        self._close_shm()
        try:
            shm = VideoShmReader(shm_name)
            shm.open(use_event=True)
            self._shm = shm
            self._shm_open_name = shm_name
            return True
        except Exception:
            self._close_shm()
            return False

    async def _ensure_detector(self) -> bool:
        if self._detector is not None:
            return True

        try:
            yaml_path = self._resolve_model_yaml()
            spec = load_model_spec(yaml_path)
            if self._conf_override >= 0:
                spec = replace(spec, conf_threshold=float(self._conf_override))
            if self._iou_override >= 0:
                spec = replace(spec, iou_threshold=float(self._iou_override))
            det = OnnxYoloDetector(spec, ort_provider=self._ort_provider)
        except Exception as exc:
            msg = str(exc)
            self._last_error = msg
            await self.set_state("lastError", msg)
            return False

        self._detector_yaml = yaml_path
        self._model = spec
        self._detector = det
        await self.set_state("loadedModel", f"{spec.model_id} ({spec.task})")
        await self.set_state("ortActiveProviders", json.dumps(det.active_providers))
        await self.set_state("lastError", "")
        return True

    def _resolve_model_yaml(self) -> Path:
        if self._model_yaml_path:
            p = Path(self._model_yaml_path).expanduser()
            return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()

        idx = build_model_index(self._weights_dir)
        if self._model_id:
            for i in idx:
                if i.model_id == self._model_id:
                    return i.yaml_path.resolve()
        if idx:
            return idx[0].yaml_path.resolve()
        raise FileNotFoundError(f"No model yamls found in: {self._weights_dir}")

    async def _loop(self) -> None:
        import numpy as np  # type: ignore

        while True:
            try:
                await asyncio.sleep(0)
                if not self._active:
                    await asyncio.sleep(0.05)
                    continue

                await self._ensure_config_loaded()

                shm_name = self._resolve_shm_name()
                if not shm_name:
                    await asyncio.sleep(0.05)
                    continue
                if self._shm is None:
                    if not self._open_shm(shm_name):
                        await asyncio.sleep(0.1)
                        continue

                if not await self._ensure_detector():
                    await asyncio.sleep(0.1)
                    continue

                assert self._shm is not None
                self._shm.wait_new_frame(timeout_ms=10)
                header, payload = self._shm.read_latest_bgra()
                if header is None or payload is None:
                    continue

                width, height, pitch = int(header.width), int(header.height), int(header.pitch)
                if width <= 0 or height <= 0 or pitch <= 0:
                    continue
                frame_bytes = int(pitch) * int(height)
                if len(payload) < frame_bytes:
                    continue

                buf = np.frombuffer(payload, dtype=np.uint8)
                try:
                    rows = buf.reshape((height, pitch))
                    bgra = rows[:, : width * 4].reshape((height, width, 4))
                except Exception:
                    continue
                frame_bgr = np.ascontiguousarray(bgra[:, :, 0:3])
                frame_size = (int(width), int(height))

                if self._need_reinit_trackers:
                    for t in self._tracks:
                        t.tracker_kind = self._tracker_kind
                        t.init_tracker(frame_bgr)
                    self._need_reinit_trackers = False

                do_detect = False
                if self._last_detect_frame_id is None:
                    do_detect = True
                else:
                    do_detect = (int(header.frame_id) - int(self._last_detect_frame_id)) >= int(self._detect_every_n)

                if do_detect:
                    det = self._detector
                    if det is None:
                        continue
                    dets, _meta = det.detect(frame_bgr)
                    self._tracks, self._next_track_id = associate_and_update_tracks(
                        tracks=self._tracks,
                        dets=dets,
                        frame_bgr=frame_bgr,
                        frame_size=frame_size,
                        tracker_kind=self._tracker_kind,
                        iou_match=self._iou_match,
                        mismatch_iou=self._mismatch_iou,
                        mismatch_patience=self._mismatch_patience,
                        max_age=self._max_age,
                        max_targets=self._max_targets,
                        reinit_on_detect=self._reinit_on_detect,
                        next_id=self._next_track_id,
                    )
                    self._last_detect_frame_id = int(header.frame_id)
                else:
                    self._tracks = update_tracks_with_cv(
                        tracks=self._tracks,
                        frame_bgr=frame_bgr,
                        frame_size=frame_size,
                        mismatch_patience=self._mismatch_patience,
                        max_age=self._max_age,
                    )

                out_tracks: list[dict[str, Any]] = []
                for t in self._tracks:
                    x1, y1, x2, y2 = clamp_xyxy(t.xyxy, size=frame_size)
                    o: dict[str, Any] = {
                        "id": int(t.track_id),
                        "cls": str(t.cls),
                        "conf": float(t.conf),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "age": int(t.age),
                        "mismatch": int(t.mismatch),
                    }
                    if t.keypoints:
                        o["keypoints"] = [{"x": float(k.x), "y": float(k.y), "s": float(k.score) if k.score is not None else None} for k in t.keypoints]
                    if t.obb:
                        o["obb"] = [[float(x), float(y)] for x, y in t.obb]
                    out_tracks.append(o)

                payload_out: dict[str, Any] = {
                    "frameId": int(header.frame_id),
                    "tsMs": int(header.ts_ms),
                    "width": int(width),
                    "height": int(height),
                    "model": (self._model.model_id if self._model else ""),
                    "task": (self._model.task if self._model else ""),
                    "tracks": out_tracks,
                }
                await self.emit("detections", payload_out, ts_ms=int(header.ts_ms))
                await self.set_state("lastFrameId", int(header.frame_id))
                await self.set_state("lastFrameTsMs", int(header.ts_ms))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                try:
                    await self.set_state("lastError", self._last_error)
                except Exception:
                    pass
                await asyncio.sleep(0.1)
