from __future__ import annotations

import asyncio
import json
import time
from collections import deque
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
    candidates: list[Path] = []
    try:
        candidates.append((Path.cwd() / "services" / "f8" / "detect_tracker" / "weights").resolve())
    except Exception:
        pass
    try:
        # Best-effort repo root guess relative to this file:
        # <root>/packages/f8pydetect_tracker/f8pydetect_tracker/detecttracker_node.py
        root = Path(__file__).resolve().parents[3]
        candidates.append((root / "services" / "f8" / "detect_tracker" / "weights").resolve())
    except Exception:
        pass
    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                return p
        except Exception:
            continue
    # Fallback to cwd-based path (even if missing) so the user can see it.
    return candidates[0] if candidates else Path.cwd().resolve()


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


class _RollingWindow:
    def __init__(self, *, window_ms: int) -> None:
        self.window_ms = int(window_ms)
        self._q: deque[tuple[int, float]] = deque()
        self._sum = 0.0

    def push(self, ts_ms: int, v: float) -> None:
        self._q.append((int(ts_ms), float(v)))
        self._sum += float(v)
        self.prune(ts_ms)

    def prune(self, now_ms: int) -> None:
        win = int(self.window_ms)
        if win <= 0:
            self._q.clear()
            self._sum = 0.0
            return
        cutoff = int(now_ms) - win
        while self._q and int(self._q[0][0]) < cutoff:
            _, v = self._q.popleft()
            self._sum -= float(v)

    def mean(self, now_ms: int) -> float | None:
        self.prune(now_ms)
        n = len(self._q)
        if n <= 0:
            return None
        return float(self._sum) / float(n)

    def count(self, now_ms: int) -> int:
        self.prune(now_ms)
        return int(len(self._q))


class _Telemetry:
    def __init__(self) -> None:
        self.interval_ms = 1000
        self.window_ms = 2000
        self._last_emit_ms = 0
        self._frames = _RollingWindow(window_ms=self.window_ms)
        self._tracks_count = _RollingWindow(window_ms=self.window_ms)
        self._dets_count = _RollingWindow(window_ms=self.window_ms)
        self._detect_frames = _RollingWindow(window_ms=self.window_ms)
        self._t_total = _RollingWindow(window_ms=self.window_ms)
        self._t_shm_wait_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_shm_read_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_pre_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_det_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_associate_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_track_update_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_emit_ms = _RollingWindow(window_ms=self.window_ms)
        self._tracks_last = 0
        self._dets_last = 0
        self._dup_skipped = _RollingWindow(window_ms=self.window_ms)

    def set_config(self, *, interval_ms: int, window_ms: int) -> None:
        self.interval_ms = max(0, int(interval_ms))
        self.window_ms = max(100, int(window_ms))
        for w in (
            self._frames,
            self._tracks_count,
            self._dets_count,
            self._detect_frames,
            self._t_total,
            self._t_shm_wait_ms,
            self._t_shm_read_ms,
            self._t_pre_ms,
            self._t_det_ms,
            self._t_associate_ms,
            self._t_track_update_ms,
            self._t_emit_ms,
            self._dup_skipped,
        ):
            w.window_ms = self.window_ms

    def observe_frame(
        self,
        *,
        ts_ms: int,
        did_detect: bool,
        tracks: int,
        dets: int,
        total_ms: float,
        shm_wait_ms: float,
        shm_read_ms: float,
        pre_ms: float,
        det_ms: float,
        associate_ms: float,
        track_update_ms: float,
        emit_ms: float,
        dup_skipped: int,
    ) -> None:
        self._frames.push(ts_ms, 1.0)
        self._tracks_last = int(tracks)
        self._dets_last = int(dets)
        self._tracks_count.push(ts_ms, float(tracks))
        self._dets_count.push(ts_ms, float(dets))
        self._detect_frames.push(ts_ms, 1.0 if did_detect else 0.0)
        self._t_total.push(ts_ms, float(total_ms))
        self._t_shm_wait_ms.push(ts_ms, float(shm_wait_ms))
        self._t_shm_read_ms.push(ts_ms, float(shm_read_ms))
        self._t_pre_ms.push(ts_ms, float(pre_ms))
        self._t_det_ms.push(ts_ms, float(det_ms))
        self._t_associate_ms.push(ts_ms, float(associate_ms))
        self._t_track_update_ms.push(ts_ms, float(track_update_ms))
        self._t_emit_ms.push(ts_ms, float(emit_ms))
        self._dup_skipped.push(ts_ms, float(int(dup_skipped)))

    def should_emit(self, now_ms: int) -> bool:
        if int(self.interval_ms) <= 0:
            return False
        last = int(self._last_emit_ms or 0)
        return last <= 0 or (int(now_ms) - last) >= int(self.interval_ms)

    def mark_emitted(self, now_ms: int) -> None:
        self._last_emit_ms = int(now_ms)

    def summary(
        self,
        *,
        now_ms: int,
        node_id: str,
        model: ModelSpec | None,
        ort_provider: str,
        shm_name: str,
        shm_has_event: bool,
        shm_wait_timeout_ms: int,
        frame_id_last_seen: int | None,
        frame_id_last_processed: int | None,
    ) -> dict[str, Any]:
        win_ms = int(self.window_ms)
        frames = self._frames.count(now_ms)
        fps = (float(frames) * 1000.0 / float(win_ms)) if win_ms > 0 else None
        det_frac = self._detect_frames.mean(now_ms)
        tracks_avg = self._tracks_count.mean(now_ms)
        dets_avg = self._dets_count.mean(now_ms)
        assoc_avg = self._t_associate_ms.mean(now_ms)
        upd_avg = self._t_track_update_ms.mean(now_ms)
        dup_skipped_avg = self._dup_skipped.mean(now_ms)
        track_avg = None
        try:
            track_avg = (float(assoc_avg or 0.0) + float(upd_avg or 0.0)) if (assoc_avg is not None or upd_avg is not None) else None
        except Exception:
            track_avg = None
        return {
            "schemaVersion": "f8telemetry/1",
            "tsMs": int(now_ms),
            "nodeId": str(node_id),
            "serviceClass": "f8.detecttracker",
            "model": {
                "id": (model.model_id if model else ""),
                "task": (model.task if model else ""),
                "provider": (model.provider if model else ""),
            },
            "windowMs": int(win_ms),
            "source": {
                "shmName": str(shm_name or ""),
                "hasEvent": bool(shm_has_event),
                "waitTimeoutMs": int(shm_wait_timeout_ms),
            },
            "frameId": {
                "lastSeen": int(frame_id_last_seen) if frame_id_last_seen is not None else None,
                "lastProcessed": int(frame_id_last_processed) if frame_id_last_processed is not None else None,
                "duplicatesSkippedAvg": float(dup_skipped_avg) if dup_skipped_avg is not None else None,
            },
            "counts": {
                "frames": int(frames),
                "tracksLast": int(self._tracks_last),
                "tracksAvg": float(tracks_avg) if tracks_avg is not None else None,
                "detsLast": int(self._dets_last),
                "detsAvg": float(dets_avg) if dets_avg is not None else None,
            },
            "rates": {
                "fps": float(fps) if fps is not None else None,
                "detectFraction": float(det_frac) if det_frac is not None else None,
            },
            "timingsMsAvg": {
                "total": self._t_total.mean(now_ms),
                "shmWait": self._t_shm_wait_ms.mean(now_ms),
                "shmRead": self._t_shm_read_ms.mean(now_ms),
                "preprocess": self._t_pre_ms.mean(now_ms),
                "detect": self._t_det_ms.mean(now_ms),
                # Back-compat: previously exported a single `track` bucket.
                "associate": assoc_avg,
                "trackUpdate": upd_avg,
                "track": track_avg,
                "emit": self._t_emit_ms.mean(now_ms),
            },
            "runtime": {"ortProvider": str(ort_provider)},
        }


class detecttrackerServiceNode(ServiceNode):
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
            data_out_ports=["detections", "telemetry"],
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
        self._last_processed_frame_id: int | None = None
        self._dup_skipped_since_last_processed: int = 0
        self._need_reinit_trackers = False
        self._telemetry = _Telemetry()

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._ensure_config_loaded(), name=f"detecttracker:init:{self.node_id}")
            self._task = loop.create_task(self._loop(), name=f"detecttracker:loop:{self.node_id}")
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
            raw = _coerce_str(await self.get_state_value("weightsDir"), default=str(self._weights_dir))
            p = Path(raw).expanduser()
            if not p.is_absolute():
                # Resolve relative to cwd first.
                p1 = (Path.cwd() / p).resolve()
                # If missing, also try resolving relative to repo root guess.
                if not p1.exists():
                    try:
                        root = Path(__file__).resolve().parents[3]
                        p2 = (root / p).resolve()
                        p = p2 if p2.exists() else p1
                    except Exception:
                        p = p1
                else:
                    p = p1
            else:
                p = p.resolve()
            self._weights_dir = p
            await self._publish_model_index()
            await self._reset_detector()
        elif name == "modelId":
            self._model_id = _coerce_str(await self.get_state_value("modelId"), default=self._model_id)
            await self._reset_detector()
        elif name == "modelYamlPath":
            self._model_yaml_path = _coerce_str(await self.get_state_value("modelYamlPath"), default=self._model_yaml_path)
            await self._reset_detector()
        elif name == "ortProvider":
            v = _coerce_str(await self.get_state_value("ortProvider"), default=str(self._ort_provider)).lower()
            self._ort_provider = v if v in ("auto", "cuda", "cpu") else "auto"
            await self._reset_detector()
        elif name == "trackerKind":
            v = _coerce_str(await self.get_state_value("trackerKind"), default=str(self._tracker_kind)).lower()
            self._tracker_kind = v if v in ("none", "csrt", "kcf", "mosse") else "kcf"
            self._need_reinit_trackers = True
        elif name == "detectEveryN":
            self._detect_every_n = _coerce_int(await self.get_state_value("detectEveryN"), default=self._detect_every_n, minimum=1, maximum=10_000)
        elif name == "maxTargets":
            self._max_targets = _coerce_int(await self.get_state_value("maxTargets"), default=self._max_targets, minimum=1, maximum=1000)
        elif name == "iouMatch":
            self._iou_match = _coerce_float(await self.get_state_value("iouMatch"), default=self._iou_match, minimum=0.0, maximum=1.0)
        elif name == "mismatchIou":
            self._mismatch_iou = _coerce_float(await self.get_state_value("mismatchIou"), default=self._mismatch_iou, minimum=0.0, maximum=1.0)
        elif name == "mismatchPatience":
            self._mismatch_patience = _coerce_int(await self.get_state_value("mismatchPatience"), default=self._mismatch_patience, minimum=1, maximum=1000)
        elif name == "maxAge":
            self._max_age = _coerce_int(await self.get_state_value("maxAge"), default=self._max_age, minimum=1, maximum=100_000)
        elif name == "reinitOnDetect":
            self._reinit_on_detect = _coerce_bool(await self.get_state_value("reinitOnDetect"), default=self._reinit_on_detect)
        elif name == "confThreshold":
            self._conf_override = _coerce_float(await self.get_state_value("confThreshold"), default=self._conf_override)
            await self._reset_detector()
        elif name == "iouThreshold":
            self._iou_override = _coerce_float(await self.get_state_value("iouThreshold"), default=self._iou_override)
            await self._reset_detector()
        elif name == "sourceServiceId":
            self._source_service_id = _coerce_str(await self.get_state_value("sourceServiceId"), default=self._source_service_id)
            await self._maybe_reopen_shm()
        elif name == "shmName":
            self._shm_name = _coerce_str(await self.get_state_value("shmName"), default=self._shm_name)
            await self._maybe_reopen_shm()
        elif name == "telemetryIntervalMs":
            self._telemetry.set_config(
                interval_ms=_coerce_int(await self.get_state_value("telemetryIntervalMs"), default=self._telemetry.interval_ms, minimum=0, maximum=60000),
                window_ms=self._telemetry.window_ms,
            )
        elif name == "telemetryWindowMs":
            self._telemetry.set_config(
                interval_ms=self._telemetry.interval_ms,
                window_ms=_coerce_int(await self.get_state_value("telemetryWindowMs"), default=self._telemetry.window_ms, minimum=100, maximum=60000),
            )

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return

        raw_weights = _coerce_str(
            await self.get_state_value("weightsDir"),
            default=str(self._initial_state.get("weightsDir") or _default_weights_dir()),
        )
        p = Path(raw_weights).expanduser()
        if not p.is_absolute():
            p1 = (Path.cwd() / p).resolve()
            if not p1.exists():
                try:
                    root = Path(__file__).resolve().parents[3]
                    p2 = (root / p).resolve()
                    p = p2 if p2.exists() else p1
                except Exception:
                    p = p1
            else:
                p = p1
        else:
            p = p.resolve()
        self._weights_dir = p
        self._model_id = _coerce_str(await self.get_state_value("modelId"), default=str(self._initial_state.get("modelId") or ""))
        self._model_yaml_path = _coerce_str(
            await self.get_state_value("modelYamlPath"), default=str(self._initial_state.get("modelYamlPath") or "")
        )
        v = _coerce_str(await self.get_state_value("ortProvider"), default=str(self._initial_state.get("ortProvider") or "auto")).lower()
        self._ort_provider = v if v in ("auto", "cuda", "cpu") else "auto"

        v = _coerce_str(await self.get_state_value("trackerKind"), default=str(self._initial_state.get("trackerKind") or "kcf")).lower()
        self._tracker_kind = v if v in ("none", "csrt", "kcf", "mosse") else "kcf"

        self._detect_every_n = _coerce_int(await self.get_state_value("detectEveryN"), default=int(self._initial_state.get("detectEveryN") or 5), minimum=1)
        self._max_targets = _coerce_int(await self.get_state_value("maxTargets"), default=int(self._initial_state.get("maxTargets") or 5), minimum=1, maximum=1000)
        self._iou_match = _coerce_float(await self.get_state_value("iouMatch"), default=float(self._initial_state.get("iouMatch") or 0.3), minimum=0.0, maximum=1.0)
        self._mismatch_iou = _coerce_float(await self.get_state_value("mismatchIou"), default=float(self._initial_state.get("mismatchIou") or 0.2), minimum=0.0, maximum=1.0)
        self._mismatch_patience = _coerce_int(await self.get_state_value("mismatchPatience"), default=int(self._initial_state.get("mismatchPatience") or 3), minimum=1)
        self._max_age = _coerce_int(await self.get_state_value("maxAge"), default=int(self._initial_state.get("maxAge") or 30), minimum=1)
        self._reinit_on_detect = _coerce_bool(await self.get_state_value("reinitOnDetect"), default=bool(self._initial_state.get("reinitOnDetect", True)))
        self._conf_override = _coerce_float(await self.get_state_value("confThreshold"), default=float(self._initial_state.get("confThreshold") or -1.0))
        self._iou_override = _coerce_float(await self.get_state_value("iouThreshold"), default=float(self._initial_state.get("iouThreshold") or -1.0))

        self._source_service_id = _coerce_str(await self.get_state_value("sourceServiceId"), default=str(self._initial_state.get("sourceServiceId") or ""))
        self._shm_name = _coerce_str(await self.get_state_value("shmName"), default=str(self._initial_state.get("shmName") or ""))
        self._telemetry.set_config(
            interval_ms=_coerce_int(
                await self.get_state_value("telemetryIntervalMs"), default=int(self._initial_state.get("telemetryIntervalMs") or 1000), minimum=0, maximum=60000
            ),
            window_ms=_coerce_int(
                await self.get_state_value("telemetryWindowMs"), default=int(self._initial_state.get("telemetryWindowMs") or 2000), minimum=100, maximum=60000
            ),
        )

        self._config_loaded = True
        await self._publish_model_index()

    async def _publish_model_index(self) -> None:
        # Build index; print diagnostics so we can debug path/env issues even
        # when state fields don't show up in the UI.
        idx = build_model_index(self._weights_dir)
        if not idx:
            # If an explicit modelYamlPath is set, prefer indexing from that file's directory.
            try:
                if self._model_yaml_path:
                    yp = Path(self._model_yaml_path).expanduser()
                    yaml_path = yp.resolve() if yp.is_absolute() else (Path.cwd() / yp).resolve()
                    if yaml_path.exists():
                        alt_dir = yaml_path.parent
                        idx2 = build_model_index(alt_dir)
                        if idx2:
                            idx = idx2
            except Exception:
                pass
            try:
                yamls = list(self._weights_dir.glob("*.yaml")) + list(self._weights_dir.glob("*.yml"))
            except Exception:
                yamls = []
            # Always publish useful diagnostics (paths/cwd are the most common root cause).
            try:
                cwd = Path.cwd().resolve()
            except Exception:
                cwd = Path(".")
            if not idx:
                msg = (
                    "Model index is empty. "
                    f"weightsDir={self._weights_dir!s} (exists={self._weights_dir.exists() if hasattr(self._weights_dir,'exists') else 'unknown'}), "
                    f"yamlCount={len(yamls)}, cwd={cwd!s}. "
                    "If you expect models here, ensure the path is correct and 'pyyaml' is installed in the detect_tracker runtime environment."
                )
                if self._model_yaml_path:
                    msg += f" modelYamlPath={self._model_yaml_path!s}"
                await self.set_state("lastError", msg)
        payload = [i.model_id for i in idx]
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
        warn_parts: list[str] = []
        warn = str(getattr(det, "provider_warning", "") or "").strip()
        if warn:
            warn_parts.append(warn)

        # If user expects CUDA but it's not available, surface a clear hint.
        prefer = str(self._ort_provider or "auto").lower()
        if prefer in ("auto", "cuda"):
            try:
                import onnxruntime as ort  # type: ignore

                available = list(ort.get_available_providers())  # type: ignore[attr-defined]
            except Exception:
                available = []
            active_l = {str(p).lower() for p in (det.active_providers or [])}
            avail_l = {str(p).lower() for p in (available or [])}
            if "cudaexecutionprovider" not in active_l and "cudaexecutionprovider" not in avail_l:
                hint = (
                    "CUDAExecutionProvider is not available in this runtime; running on "
                    f"activeProviders={det.active_providers!r}, availableProviders={available!r}. "
                    "If you expect NVIDIA GPU acceleration, ensure a CUDA-enabled ONNX Runtime build and compatible CUDA runtime are installed "
                    "(for this repo, try the optional pixi env `onnx-cuda`)."
                )
                warn_parts.append(hint)

        await self.set_state("lastError", "\n".join(warn_parts).strip())
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
                t0 = time.perf_counter()
                self._shm.wait_new_frame(timeout_ms=10)
                t_wait = time.perf_counter()
                header, payload = self._shm.read_latest_bgra()
                t1 = time.perf_counter()
                if header is None or payload is None:
                    continue
                frame_id_seen = int(header.frame_id)
                # Avoid re-processing the same SHM frame (which can inflate FPS and waste CPU
                # when the SHM producer is slower than our loop).
                if self._last_processed_frame_id is not None and frame_id_seen == int(self._last_processed_frame_id):
                    self._dup_skipped_since_last_processed += 1
                    continue
                dup_skipped = int(self._dup_skipped_since_last_processed)
                self._dup_skipped_since_last_processed = 0

                width, height, pitch = int(header.width), int(header.height), int(header.pitch)
                if width <= 0 or height <= 0 or pitch <= 0:
                    continue
                frame_bytes = int(pitch) * int(height)
                if len(payload) < frame_bytes:
                    continue
                self._last_processed_frame_id = frame_id_seen

                buf = np.frombuffer(payload, dtype=np.uint8)
                try:
                    rows = buf.reshape((height, pitch))
                    bgra = rows[:, : width * 4].reshape((height, width, 4))
                except Exception:
                    continue
                frame_bgr = np.ascontiguousarray(bgra[:, :, 0:3])
                frame_size = (int(width), int(height))
                t2 = time.perf_counter()

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
                    t_det0 = time.perf_counter()
                    dets, _meta = det.detect(frame_bgr)
                    t_det1 = time.perf_counter()
                    t_assoc0 = time.perf_counter()
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
                    t_assoc1 = time.perf_counter()
                    self._last_detect_frame_id = int(header.frame_id)
                    t3 = time.perf_counter()
                else:
                    t_det0 = t_det1 = time.perf_counter()
                    t_assoc0 = t_assoc1 = t_det1
                    t_upd0 = time.perf_counter()
                    self._tracks = update_tracks_with_cv(
                        tracks=self._tracks,
                        frame_bgr=frame_bgr,
                        frame_size=frame_size,
                        mismatch_patience=self._mismatch_patience,
                        max_age=self._max_age,
                    )
                    t_upd1 = time.perf_counter()
                    t3 = time.perf_counter()

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
                t_emit0 = time.perf_counter()
                await self.emit("detections", payload_out, ts_ms=int(header.ts_ms))
                t_emit1 = time.perf_counter()
                # Do not publish high-frequency frame counters as state.
                # These are already represented in telemetry and publishing them as state
                # can overwhelm the UI with per-frame updates.

                now_ms = int(header.ts_ms)
                self._telemetry.observe_frame(
                    ts_ms=now_ms,
                    did_detect=bool(do_detect),
                    tracks=len(self._tracks),
                    dets=(len(dets) if do_detect else 0),
                    total_ms=(time.perf_counter() - t0) * 1000.0,
                    shm_wait_ms=(t_wait - t0) * 1000.0,
                    shm_read_ms=(t1 - t_wait) * 1000.0,
                    pre_ms=(t2 - t1) * 1000.0,
                    det_ms=(t_det1 - t_det0) * 1000.0,
                    associate_ms=(t_assoc1 - t_assoc0) * 1000.0,
                    track_update_ms=((t_upd1 - t_upd0) * 1000.0 if not do_detect else 0.0),
                    emit_ms=(t_emit1 - t_emit0) * 1000.0,
                    dup_skipped=dup_skipped,
                )
                if self._telemetry.should_emit(now_ms):
                    shm_has_event = False
                    try:
                        shm_has_event = bool(getattr(self._shm, "_event", None) is not None)
                    except Exception:
                        shm_has_event = False
                    tel = self._telemetry.summary(
                        now_ms=now_ms,
                        node_id=self.node_id,
                        model=self._model,
                        ort_provider=self._ort_provider,
                        shm_name=str(self._shm_open_name or shm_name),
                        shm_has_event=shm_has_event,
                        shm_wait_timeout_ms=10,
                        frame_id_last_seen=frame_id_seen,
                        frame_id_last_processed=self._last_processed_frame_id,
                    )
                    await self.emit("telemetry", tel, ts_ms=now_ms)
                    self._telemetry.mark_emitted(now_ms)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                try:
                    await self.set_state("lastError", self._last_error)
                except Exception:
                    pass
                await asyncio.sleep(0.1)
