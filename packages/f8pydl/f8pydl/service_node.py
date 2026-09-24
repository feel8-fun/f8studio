from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from f8pysdk.codec import coerce_bool, coerce_float, coerce_int, coerce_str, parse_str_list
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode
from f8pysdk.time_utils import now_ms
from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32

from .constants import CLASSIFICATION_SCHEMA_VERSION, DETECTION_SCHEMA_VERSION
from .model_config import ModelSpec, ModelTask, build_model_index, build_model_index_with_errors, load_model_spec
from .onnx_runtime import OnnxClassifierRuntime, OnnxYoloDetectorRuntime, OnnxYowoTemporalDetectorRuntime
from .service_paths import default_weights_dir, resolve_path_from_cwd_or_repo
from .video_frame_source import (
    LatestVideoFrameSource,
    VideoFrameSourceConfig,
    video_source_metadata,
)
from .vision_utils import clamp_xyxy
from .weights_downloader import ensure_onnx_file, onnx_file_matches_sha256


def _default_weights_dir() -> Path:
    return default_weights_dir(extra_relative_candidates=("services/f8/detect_tracker/weights",))


def _resolve_path_from_cwd_or_repo(raw: str) -> Path:
    return resolve_path_from_cwd_or_repo(raw)


_DL_MODEL_METADATA_ERRORS = (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError)
_DL_ONNX_PROVIDER_ERRORS = (ImportError, RuntimeError, TypeError, ValueError)
# ONNX/runtime/video-source calls cross third-party native/runtime boundaries;
# keep the loop alive and report details through the service error channel.
_DL_DOWNLOAD_ERRORS = (Exception,)
_DL_RUNTIME_BOUNDARY_ERRORS = (Exception,)


@dataclass(frozen=True)
class _TemporalBufferedFrame:
    prepared_frame: Any
    frame_id: int
    ts_ms: int


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
        self._infer_ms = _RollingWindow(window_ms=self.window_ms)
        self._total_ms = _RollingWindow(window_ms=self.window_ms)
        self._dup_skipped = _RollingWindow(window_ms=self.window_ms)

    def set_config(self, *, interval_ms: int, window_ms: int) -> None:
        self.interval_ms = max(0, int(interval_ms))
        self.window_ms = max(100, int(window_ms))
        self._frames.window_ms = self.window_ms
        self._infer_ms.window_ms = self.window_ms
        self._total_ms.window_ms = self.window_ms
        self._dup_skipped.window_ms = self.window_ms

    def observe_frame(self, *, ts_ms: int, infer_ms: float, total_ms: float, dup_skipped: int) -> None:
        self._frames.push(ts_ms, 1.0)
        self._infer_ms.push(ts_ms, float(infer_ms))
        self._total_ms.push(ts_ms, float(total_ms))
        self._dup_skipped.push(ts_ms, float(dup_skipped))

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
        service_class: str,
        model: ModelSpec | None,
        ort_provider: str,
        frame_id_last_seen: int | None,
        frame_id_last_processed: int | None,
    ) -> dict[str, Any]:
        win_ms = int(self.window_ms)
        frames = self._frames.count(now_ms)
        fps = (float(frames) * 1000.0 / float(win_ms)) if win_ms > 0 else None
        return {
            "schemaVersion": "f8dlTelemetry/1",
            "tsMs": int(now_ms),
            "nodeId": str(node_id),
            "serviceClass": str(service_class),
            "model": {
                "id": (model.model_id if model else ""),
                "task": (model.task if model else ""),
                "provider": (model.provider if model else ""),
            },
            "windowMs": int(win_ms),
            "source": video_source_metadata(),
            "frameId": {
                "lastSeen": int(frame_id_last_seen) if frame_id_last_seen is not None else None,
                "lastProcessed": int(frame_id_last_processed) if frame_id_last_processed is not None else None,
                "duplicatesSkippedAvg": self._dup_skipped.mean(now_ms),
            },
            "rates": {"fps": float(fps) if fps is not None else None},
            "timingsMsAvg": {
                "infer": self._infer_ms.mean(now_ms),
                "total": self._total_ms.mean(now_ms),
            },
            "runtime": {"ortProvider": str(ort_provider)},
        }


class OnnxVisionServiceNode(ServiceNode):
    def __init__(
        self,
        *,
        node_id: str,
        node: Any,
        initial_state: dict[str, Any] | None,
        service_class: str,
        service_task: Literal["detector", "humandetector", "classifier"],
        output_port: Literal["detections", "classifications"],
        allowed_tasks: set[ModelTask],
    ) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=[str(output_port)],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._service_class = str(service_class)
        self._service_task = service_task
        self._output_port = output_port
        self._allowed_tasks = set(allowed_tasks)

        self._active = True
        self._config_loaded = False
        self._init_task: asyncio.Task[object] | None = None
        self._task: asyncio.Task[object] | None = None

        self._weights_dir = _default_weights_dir()
        self._model_yaml_path = ""
        self._model_id = ""
        self._ort_provider: Literal["auto", "cuda", "cpu"] = "auto"
        self._infer_every_n = 1
        self._conf_override = -1.0
        self._iou_override = -1.0
        self._top_k = 5
        self._enabled_classes: list[str] = []
        self._per_class_k = 0
        self._auto_download_weights = True
        self._download_retry_at_monotonic = 0.0

        self._video_source: LatestVideoFrameSource | None = None

        self._model: ModelSpec | None = None
        self._det_runtime: OnnxYoloDetectorRuntime | None = None
        self._temporal_det_runtime: OnnxYowoTemporalDetectorRuntime | None = None
        self._cls_runtime: OnnxClassifierRuntime | None = None
        self._runtime_yaml: Path | None = None
        self._last_error = ""
        self._last_error_signature = ""
        self._last_error_repeats = 0
        self._model_index_warning = ""
        self._temporal_frame_buffer: deque[_TemporalBufferedFrame] = deque()
        self._temporal_frame_counter = 0

        self._last_infer_frame_id: int | None = None
        self._last_processed_frame_id: int | None = None
        self._dup_skipped_since_last_processed = 0

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        self._video_source = LatestVideoFrameSource(config=VideoFrameSourceConfig.from_bus(bus))
        loop = asyncio.get_running_loop()
        self._init_task = loop.create_task(self._ensure_config_loaded(), name=f"f8dl:init:{self.node_id}")
        self._task = loop.create_task(self._loop(), name=f"f8dl:loop:{self.node_id}")

    async def close(self) -> None:
        self._active = False
        tasks: list[asyncio.Task[object]] = []
        init_task = self._init_task
        self._init_task = None
        if init_task is not None:
            tasks.append(init_task)
        t = self._task
        self._task = None
        if t is not None:
            tasks.append(t)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._close_video_source()
        self._reset_temporal_buffer()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_rungraph(self, graph: Any) -> None:
        del graph
        await self._ensure_config_loaded()
        await self._publish_model_index(force_publish=True)

    async def validate_rungraph(self, graph: Any) -> None:
        del graph
        return None

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del value
        del ts_ms
        name = str(field or "").strip()
        await self._ensure_config_loaded()

        if name == "weightsDir":
            raw = coerce_str(await self.get_state_value("weightsDir"), default=str(self._weights_dir))
            self._weights_dir = _resolve_path_from_cwd_or_repo(raw)
            await self._publish_model_index(force_publish=True)
            await self._reset_runtime()
            return

        if name == "modelId":
            self._model_id = coerce_str(await self.get_state_value("modelId"), default=self._model_id)
            await self._reset_runtime()
            return

        if name == "modelYamlPath":
            self._model_yaml_path = coerce_str(
                await self.get_state_value("modelYamlPath"), default=self._model_yaml_path
            )
            await self._reset_runtime()
            return

        if name == "ortProvider":
            v = coerce_str(await self.get_state_value("ortProvider"), default=str(self._ort_provider)).lower()
            self._ort_provider = v if v in ("auto", "cuda", "cpu") else "auto"
            await self._reset_runtime()
            return

        if name == "inferEveryN":
            self._infer_every_n = coerce_int(
                await self.get_state_value("inferEveryN"),
                default=self._infer_every_n,
                minimum=1,
                maximum=10000,
            )
            return

        if name == "confThreshold":
            self._conf_override = coerce_float(
                await self.get_state_value("confThreshold"), default=self._conf_override
            )
            await self._reset_runtime()
            return

        if name == "iouThreshold":
            self._iou_override = coerce_float(await self.get_state_value("iouThreshold"), default=self._iou_override)
            await self._reset_runtime()
            return

        if name == "topK":
            self._top_k = coerce_int(await self.get_state_value("topK"), default=self._top_k, minimum=1, maximum=100)
            return

        if name == "enabledClasses":
            self._enabled_classes = (
                parse_str_list(
                    await self.get_state_value("enabledClasses"),
                    allow_json_string=True,
                )
                or []
            )
            self._enabled_classes = self._normalize_enabled_classes(self._enabled_classes)
            return

        if name == "perClassK":
            self._per_class_k = coerce_int(
                await self.get_state_value("perClassK"),
                default=self._per_class_k,
                minimum=0,
                maximum=10000,
            )
            return

        if name == "autoDownloadWeights":
            self._auto_download_weights = coerce_bool(
                await self.get_state_value("autoDownloadWeights"),
                default=self._auto_download_weights,
            )
            return

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return

        raw_weights = coerce_str(
            await self.get_state_value("weightsDir"),
            default=str(self._initial_state.get("weightsDir") or _default_weights_dir()),
        )
        self._weights_dir = _resolve_path_from_cwd_or_repo(raw_weights)
        self._model_id = coerce_str(
            await self.get_state_value("modelId"), default=str(self._initial_state.get("modelId") or "")
        )
        self._model_yaml_path = coerce_str(
            await self.get_state_value("modelYamlPath"),
            default=str(self._initial_state.get("modelYamlPath") or ""),
        )
        v = coerce_str(
            await self.get_state_value("ortProvider"), default=str(self._initial_state.get("ortProvider") or "auto")
        ).lower()
        self._ort_provider = v if v in ("auto", "cuda", "cpu") else "auto"
        self._infer_every_n = coerce_int(
            await self.get_state_value("inferEveryN"),
            default=int(self._initial_state.get("inferEveryN") or 1),
            minimum=1,
            maximum=10000,
        )
        self._conf_override = coerce_float(
            await self.get_state_value("confThreshold"),
            default=float(self._initial_state.get("confThreshold") or -1.0),
        )
        self._iou_override = coerce_float(
            await self.get_state_value("iouThreshold"),
            default=float(self._initial_state.get("iouThreshold") or -1.0),
        )
        self._top_k = coerce_int(
            await self.get_state_value("topK"),
            default=int(self._initial_state.get("topK") or 5),
            minimum=1,
            maximum=100,
        )
        self._enabled_classes = (
            parse_str_list(
                await self.get_state_value("enabledClasses"),
                allow_json_string=True,
            )
            or []
        )
        self._enabled_classes = self._normalize_enabled_classes(self._enabled_classes)
        self._per_class_k = coerce_int(
            await self.get_state_value("perClassK"),
            default=int(self._initial_state.get("perClassK") or 0),
            minimum=0,
            maximum=10000,
        )
        self._auto_download_weights = coerce_bool(
            await self.get_state_value("autoDownloadWeights"),
            default=bool(self._initial_state.get("autoDownloadWeights", True)),
        )
        self._config_loaded = True
        await self._publish_model_index(force_publish=True)

    async def _publish_model_index(self, *, force_publish: bool = False) -> None:
        idx, errors = build_model_index_with_errors(self._weights_dir, allowed_tasks=self._allowed_tasks)
        warning = ""
        if errors:
            preview = errors[:3]
            parts: list[str] = []
            for item in preview:
                path_name = Path(str(item.get("path") or "")).name
                err_text = str(item.get("error") or "").strip()
                if path_name and err_text:
                    parts.append(f"{path_name}: {err_text}")
                elif err_text:
                    parts.append(err_text)
            warning = f"Skipped {len(errors)} invalid model yaml(s)."
            if parts:
                warning += " " + " | ".join(parts)
            remain = int(len(errors) - len(preview))
            if remain > 0:
                warning += f" | +{remain} more"
            if len(warning) > 1000:
                warning = warning[:1000] + "..."
        self._model_index_warning = warning

        if not idx:
            msg = (
                "Model index is empty. "
                f"weightsDir={self._weights_dir!s} "
                f"allowedTasks={sorted(self._allowed_tasks)!r}. "
                "Ensure model yaml task matches service task."
            )
            if warning:
                msg = f"{msg}\n{warning}"
            await self._set_last_error(msg)
        elif warning:
            await self._set_last_error(warning)
        payload = [i.model_id for i in idx]
        await self.set_state("availableModels", payload, force_publish=force_publish)
        if idx:
            available = set(payload)
            if not self._model_id or self._model_id not in available:
                self._model_id = idx[0].model_id
                await self.set_state("modelId", self._model_id, force_publish=force_publish)
            elif force_publish:
                await self.set_state("modelId", self._model_id, force_publish=True)
        else:
            self._model_id = ""
            await self.set_state("modelId", self._model_id, force_publish=force_publish)
        await self._publish_selected_model_metadata(force_publish=force_publish)

    async def _publish_selected_model_metadata(self, *, force_publish: bool = False) -> None:
        try:
            yaml_path = self._resolve_model_yaml()
            spec = load_model_spec(yaml_path)
        except _DL_MODEL_METADATA_ERRORS:
            await self.set_state("modelClasses", [], force_publish=force_publish)
            await self.set_state("enabledClasses", [], force_publish=force_publish)
            return

        await self.set_state("modelClasses", [str(x) for x in (spec.classes or [])], force_publish=force_publish)
        self._enabled_classes = self._normalize_enabled_classes(
            self._enabled_classes,
            allowed_classes=list(spec.classes or []),
        )
        await self.set_state("enabledClasses", list(self._enabled_classes), force_publish=force_publish)

    async def _set_last_error(self, message: str) -> None:
        self._last_error = str(message or "")
        if self._last_error:
            await self.report_error(
                "DL_RUNTIME",
                self._last_error,
                severity="warning" if self._last_error.lower().startswith("downloading ") else "error",
                fingerprint=f"dl:{self._last_error}",
            )
            return
        await self.clear_error()

    async def _record_exception(self, *, where: str, exc: Exception) -> None:
        signature = f"{type(exc).__name__}:{exc}"
        self._last_error_repeats = self._last_error_repeats + 1 if signature == self._last_error_signature else 1
        self._last_error_signature = signature
        if self._last_error_repeats != 1 and self._last_error_repeats % 100 != 0:
            return
        message = (
            f"{where} failed with {type(exc).__name__}: {exc}\n"
            f"repeat={self._last_error_repeats}\n"
            f"traceback:\n{traceback.format_exc()}"
        )
        await self._set_last_error(message)

    async def _reset_runtime(self) -> None:
        self._det_runtime = None
        self._temporal_det_runtime = None
        self._cls_runtime = None
        self._runtime_yaml = None
        self._model = None
        self._reset_temporal_buffer()
        self._last_infer_frame_id = None
        self._last_processed_frame_id = None
        self._dup_skipped_since_last_processed = 0
        self._last_error_signature = ""
        self._last_error_repeats = 0
        await self.set_state("loadedModel", "")
        await self.clear_error()
        await self.set_state("ortActiveProviders", "")
        await self._publish_selected_model_metadata(force_publish=True)
        if self._model_index_warning:
            await self._set_last_error(self._model_index_warning)

    def _resolve_video_stream_key(self) -> str:
        return self.input_zenoh_key("video") or ""

    def _normalize_enabled_classes(self, values: list[str], *, allowed_classes: list[str] | None = None) -> list[str]:
        if allowed_classes is not None:
            model_classes = list(allowed_classes)
            if not model_classes:
                return []
        else:
            model_classes = list(self._model.classes) if self._model is not None else []
        allowed = set(model_classes)
        out: list[str] = []
        seen: set[str] = set()
        for raw in values:
            name = coerce_str(raw)
            if not name:
                continue
            if allowed and name not in allowed:
                continue
            if name in seen:
                continue
            out.append(name)
            seen.add(name)
        return out

    def _apply_detection_filters(self, detections: list[Any]) -> list[Any]:
        enabled = set(self._enabled_classes)
        filtered: list[Any] = []
        if enabled:
            for det in detections:
                cls_name = str(det.cls)
                if cls_name in enabled:
                    filtered.append(det)
        else:
            filtered = list(detections)

        per_class_k = int(self._per_class_k)
        if per_class_k <= 0:
            return filtered

        grouped: dict[str, list[Any]] = {}
        for det in filtered:
            cls_name = str(det.cls)
            bucket = grouped.get(cls_name)
            if bucket is None:
                grouped[cls_name] = [det]
            else:
                bucket.append(det)

        picked: list[Any] = []
        for cls_name in sorted(grouped.keys()):
            bucket = grouped[cls_name]
            bucket.sort(key=lambda item: float(item.conf), reverse=True)
            picked.extend(bucket[:per_class_k])
        picked.sort(key=lambda item: float(item.conf), reverse=True)
        return picked

    def _close_video_source(self) -> None:
        source = self._video_source
        self._video_source = None
        if source is not None:
            source.close()

    def _ensure_video_source(self) -> LatestVideoFrameSource:
        source = self._video_source
        if source is not None:
            return source
        source = LatestVideoFrameSource(config=VideoFrameSourceConfig.from_bus(self._bus))
        self._video_source = source
        return source

    def _reset_temporal_buffer(self, *, maxlen: int | None = None) -> None:
        next_maxlen = maxlen if maxlen is not None else self._temporal_frame_buffer.maxlen
        self._temporal_frame_buffer = deque(maxlen=next_maxlen)
        self._temporal_frame_counter = 0

    def _append_temporal_frame(self, *, prepared_frame: Any, frame_id: int, ts_ms: int) -> None:
        self._temporal_frame_buffer.append(
            _TemporalBufferedFrame(
                prepared_frame=prepared_frame,
                frame_id=int(frame_id),
                ts_ms=int(ts_ms),
            )
        )
        self._temporal_frame_counter += 1

    def _temporal_window_ready(self) -> bool:
        runtime = self._temporal_det_runtime
        if runtime is None:
            return False
        return len(self._temporal_frame_buffer) >= int(runtime.buffer_span)

    def _should_infer_temporal(self) -> bool:
        if not self._temporal_window_ready():
            return False
        if self._last_infer_frame_id is None:
            return True
        return (int(self._temporal_frame_counter) % int(self._infer_every_n)) == 0

    def _build_temporal_sequence(self) -> Any:
        import numpy as np  # type: ignore

        runtime = self._temporal_det_runtime
        if runtime is None:
            raise RuntimeError("Temporal detector runtime is not initialized.")
        if not self._temporal_window_ready():
            raise RuntimeError("Temporal detector window is not warm.")

        frames = list(self._temporal_frame_buffer)
        last_index = len(frames) - 1
        selected: list[Any] = []
        for offset in reversed(range(int(runtime.clip_length))):
            index = last_index - offset * int(runtime.sampling_rate)
            if index < 0:
                raise RuntimeError(
                    f"Temporal detector buffer underflow for clipLength={runtime.clip_length} "
                    f"samplingRate={runtime.sampling_rate}."
                )
            selected.append(frames[index].prepared_frame)
        return np.stack(tuple(selected), axis=0)

    def _reset_video_source(self) -> None:
        source = self._video_source
        if source is not None:
            source.reset()
        self._reset_temporal_buffer()
        self._last_infer_frame_id = None
        self._last_processed_frame_id = None
        self._dup_skipped_since_last_processed = 0

    def _record_output_timing(
        self,
        *,
        port: str,
        started_at: float,
        source_ts_ms: int,
    ) -> None:
        completed_ts_ms = int(now_ms())
        process_ms = max(0.0, (time.perf_counter() - float(started_at)) * 1000.0)
        latency_ms = 0.0
        if int(source_ts_ms) > 0:
            latency_ms = max(0.0, float(completed_ts_ms - int(source_ts_ms)))
        self.record_monitor_timing(
            port=port,
            process_ms=process_ms,
            latency_ms=latency_ms,
            ts_ms=completed_ts_ms,
        )

    def _resolve_model_yaml(self) -> Path:
        if self._model_yaml_path:
            return _resolve_path_from_cwd_or_repo(self._model_yaml_path)
        idx = build_model_index(self._weights_dir, allowed_tasks=self._allowed_tasks)
        if self._model_id:
            for item in idx:
                if item.model_id == self._model_id:
                    return item.yaml_path.resolve()
        if idx:
            return idx[0].yaml_path.resolve()
        raise FileNotFoundError(
            f"No model yamls found in {self._weights_dir} for allowedTasks={sorted(self._allowed_tasks)!r}"
        )

    async def _ensure_runtime(self) -> bool:
        if self._det_runtime is not None or self._temporal_det_runtime is not None or self._cls_runtime is not None:
            return True

        yaml_path = self._resolve_model_yaml()
        spec = load_model_spec(yaml_path)
        await self._ensure_onnx_available(spec)
        if spec.task not in self._allowed_tasks:
            raise ValueError(
                f"Model task mismatch: model task={spec.task!r}, service task={self._service_task!r}, "
                f"allowed={sorted(self._allowed_tasks)!r}"
            )

        if spec.task != "yolo_cls":
            if self._conf_override >= 0:
                spec = replace(spec, conf_threshold=float(self._conf_override))
            if self._iou_override >= 0:
                spec = replace(spec, iou_threshold=float(self._iou_override))

        if spec.task == "yolo_cls":
            runtime = await self._create_classifier_runtime(spec)
            self._cls_runtime = runtime
            providers = runtime.active_providers
            warn = runtime.provider_warning
        elif spec.task == "yowo_temporal_det":
            runtime = await self._create_temporal_detector_runtime(spec)
            self._temporal_det_runtime = runtime
            self._reset_temporal_buffer(maxlen=runtime.buffer_span)
            providers = runtime.active_providers
            warn = runtime.provider_warning
        else:
            runtime = await self._create_yolo_detector_runtime(spec)
            self._det_runtime = runtime
            providers = runtime.active_providers
            warn = runtime.provider_warning

        self._runtime_yaml = yaml_path
        self._model = spec
        self._enabled_classes = self._normalize_enabled_classes(self._enabled_classes)
        await self.set_state("loadedModel", f"{spec.model_id} ({spec.task})")
        await self.set_state("ortActiveProviders", json.dumps(providers))
        await self.set_state("modelClasses", [str(x) for x in (spec.classes or [])])
        await self.set_state("enabledClasses", list(self._enabled_classes))

        warn_parts: list[str] = []
        if warn:
            warn_parts.append(str(warn))
        prefer = str(self._ort_provider or "auto").lower()
        if prefer in ("auto", "cuda"):
            try:
                import onnxruntime as ort  # type: ignore

                available = list(ort.get_available_providers())  # type: ignore[attr-defined]
            except _DL_ONNX_PROVIDER_ERRORS as exc:
                available = []
                warn_parts.append(f"Failed to query ORT available providers: {type(exc).__name__}: {exc}")
            active_l = {str(p).lower() for p in (providers or [])}
            avail_l = {str(p).lower() for p in (available or [])}
            if "cudaexecutionprovider" not in active_l and "cudaexecutionprovider" not in avail_l:
                warn_parts.append(
                    "CUDAExecutionProvider is not available in this runtime. "
                    f"activeProviders={providers!r}, availableProviders={available!r}."
                )
        if self._model_index_warning:
            warn_parts.append(self._model_index_warning)
        await self._set_last_error("\n".join([x for x in warn_parts if str(x).strip()]).strip())
        return True

    async def _create_yolo_detector_runtime(self, spec: ModelSpec) -> OnnxYoloDetectorRuntime:
        provider = self._ort_provider
        return await asyncio.to_thread(OnnxYoloDetectorRuntime, spec, ort_provider=provider)

    async def _create_temporal_detector_runtime(self, spec: ModelSpec) -> OnnxYowoTemporalDetectorRuntime:
        provider = self._ort_provider
        return await asyncio.to_thread(OnnxYowoTemporalDetectorRuntime, spec, ort_provider=provider)

    async def _create_classifier_runtime(self, spec: ModelSpec) -> OnnxClassifierRuntime:
        provider = self._ort_provider
        return await asyncio.to_thread(OnnxClassifierRuntime, spec, ort_provider=provider)

    async def _prepare_temporal_frame(self, runtime: OnnxYowoTemporalDetectorRuntime, frame_bgr: Any) -> Any:
        return await asyncio.to_thread(runtime.prepare_frame, frame_bgr)

    async def _infer_temporal_sequence(
        self,
        runtime: OnnxYowoTemporalDetectorRuntime,
        sequence: Any,
        *,
        frame_size_hw: tuple[int, int],
    ) -> tuple[list[Any], Any]:
        return await asyncio.to_thread(runtime.infer_sequence, sequence, frame_size_hw=frame_size_hw)

    async def _infer_detections(self, runtime: OnnxYoloDetectorRuntime, frame_bgr: Any) -> tuple[list[Any], Any]:
        return await asyncio.to_thread(runtime.infer, frame_bgr)

    async def _infer_classifications(
        self,
        runtime: OnnxClassifierRuntime,
        frame_bgr: Any,
        *,
        top_k: int,
    ) -> tuple[list[Any], Any]:
        return await asyncio.to_thread(runtime.infer, frame_bgr, top_k=top_k)

    async def _ensure_onnx_available(self, spec: ModelSpec) -> None:
        if spec.onnx_path.exists():
            if onnx_file_matches_sha256(spec.onnx_path, spec.onnx_sha256):
                self._download_retry_at_monotonic = 0.0
                return
            if not self._auto_download_weights:
                raise ValueError(
                    f"Model file SHA256 mismatch: {spec.onnx_path}. "
                    "Enable autoDownloadWeights or replace the .onnx file manually."
                )
        if not self._auto_download_weights:
            raise FileNotFoundError(
                f"Model file not found: {spec.onnx_path}. "
                "Enable autoDownloadWeights or place the .onnx file manually."
            )
        if not spec.onnx_url:
            raise FileNotFoundError(
                f"Model file not found: {spec.onnx_path}. " "No onnxUrl is configured in model yaml."
            )
        now = time.monotonic()
        if float(now) < float(self._download_retry_at_monotonic):
            wait_s = int(round(float(self._download_retry_at_monotonic) - float(now)))
            raise RuntimeError(f"Auto-download cooldown active; retry in {max(1, wait_s)}s.")
        await self._set_last_error(f"Downloading ONNX model: {spec.onnx_url}")
        try:
            await asyncio.to_thread(
                ensure_onnx_file,
                onnx_path=spec.onnx_path,
                onnx_url=spec.onnx_url,
                onnx_sha256=spec.onnx_sha256,
                timeout_s=300.0,
            )
            self._download_retry_at_monotonic = 0.0
        except _DL_DOWNLOAD_ERRORS as exc:
            self._download_retry_at_monotonic = time.monotonic() + 30.0
            raise RuntimeError(
                f"Auto-download failed for model={spec.model_id!r} path={spec.onnx_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    async def _loop(self) -> None:
        import numpy as np  # type: ignore

        while True:
            try:
                await asyncio.sleep(0)
                if not self._active:
                    await asyncio.sleep(0.05)
                    continue

                await self._ensure_config_loaded()

                try:
                    await self._ensure_runtime()
                except _DL_RUNTIME_BOUNDARY_ERRORS as exc:
                    await self._record_exception(where="ensure_runtime", exc=exc)
                    await asyncio.sleep(0.1)
                    continue

                temporal_runtime = self._temporal_det_runtime
                det_runtime = self._det_runtime
                cls_runtime = self._cls_runtime
                if temporal_runtime is None and det_runtime is None and cls_runtime is None:
                    await asyncio.sleep(0.05)
                    continue

                source = self._ensure_video_source()
                video_stream_key = self._resolve_video_stream_key()
                if not video_stream_key:
                    await asyncio.sleep(0.05)
                    continue

                t0 = time.perf_counter()
                try:
                    frame = source.read_latest(stream_key=video_stream_key, timeout_ms=10)
                except _DL_RUNTIME_BOUNDARY_ERRORS as exc:
                    await self._record_exception(where="open_video_source", exc=exc)
                    await asyncio.sleep(0.1)
                    continue
                if frame is None:
                    continue
                if int(frame.fmt) != VIDEO_FORMAT_BGRA32:
                    frame.release()
                    await self._set_last_error(
                        f"input video format must be BGRA32(fmt={VIDEO_FORMAT_BGRA32}), got fmt={int(frame.fmt)}"
                    )
                    await asyncio.sleep(0.05)
                    continue
                frame_id_seen = int(frame.frame_id)
                if self._last_processed_frame_id is not None and frame_id_seen == int(self._last_processed_frame_id):
                    self._dup_skipped_since_last_processed += 1
                    frame.release()
                    continue
                self._dup_skipped_since_last_processed = 0

                do_infer = False
                if self._last_infer_frame_id is None:
                    do_infer = True
                else:
                    do_infer = (int(frame.frame_id) - int(self._last_infer_frame_id)) >= int(self._infer_every_n)
                if not do_infer:
                    self._last_processed_frame_id = frame_id_seen
                    frame.release()
                    continue

                width = int(frame.width)
                height = int(frame.height)
                pitch = int(frame.pitch)
                if width <= 0 or height <= 0 or pitch <= 0:
                    frame.release()
                    continue
                frame_bytes = int(pitch) * int(height)
                if len(frame.payload) < frame_bytes:
                    frame.release()
                    continue
                self._last_processed_frame_id = frame_id_seen

                try:
                    buf = np.frombuffer(frame.payload, dtype=np.uint8)
                    rows = buf.reshape((height, pitch))
                    bgra = rows[:, : width * 4].reshape((height, width, 4))
                    frame_bgr = bgra[:, :, 0:3]

                    t_infer0 = time.perf_counter()
                    if temporal_runtime is not None:
                        prepared = await self._prepare_temporal_frame(temporal_runtime, frame_bgr)
                        self._append_temporal_frame(
                            prepared_frame=prepared,
                            frame_id=frame_id_seen,
                            ts_ms=int(frame.ts_ms),
                        )
                        if not self._temporal_window_ready():
                            continue
                        if not self._should_infer_temporal():
                            continue
                        sequence = self._build_temporal_sequence()
                        detections, _meta = await self._infer_temporal_sequence(
                            temporal_runtime,
                            sequence,
                            frame_size_hw=(height, width),
                        )
                        payload_out = self._build_detection_payload(
                            width=width,
                            height=height,
                            frame_id=frame_id_seen,
                            ts_ms=int(frame.ts_ms),
                            stream_id=video_stream_key,
                            stream_epoch=str(frame.stream_epoch),
                            detections=detections,
                        )
                        await self.emit("detections", payload_out, ts_ms=int(frame.ts_ms))
                        self._record_output_timing(
                            port="detections",
                            started_at=t0,
                            source_ts_ms=int(frame.ts_ms),
                        )
                    elif det_runtime is not None:
                        detections, _meta = await self._infer_detections(det_runtime, frame_bgr)
                        payload_out = self._build_detection_payload(
                            width=width,
                            height=height,
                            frame_id=frame_id_seen,
                            ts_ms=int(frame.ts_ms),
                            stream_id=video_stream_key,
                            stream_epoch=str(frame.stream_epoch),
                            detections=detections,
                        )
                        await self.emit("detections", payload_out, ts_ms=int(frame.ts_ms))
                        self._record_output_timing(
                            port="detections",
                            started_at=t0,
                            source_ts_ms=int(frame.ts_ms),
                        )
                    elif cls_runtime is not None:
                        topk, _meta = await self._infer_classifications(cls_runtime, frame_bgr, top_k=self._top_k)
                        payload_out = self._build_classification_payload(
                            frame_id=frame_id_seen,
                            ts_ms=int(frame.ts_ms),
                            topk=topk,
                        )
                        await self.emit("classifications", payload_out, ts_ms=int(frame.ts_ms))
                        self._record_output_timing(
                            port="classifications",
                            started_at=t0,
                            source_ts_ms=int(frame.ts_ms),
                        )
                    t_infer1 = time.perf_counter()
                finally:
                    frame.release()

                self._last_infer_frame_id = frame_id_seen
                _ = t_infer1
                _ = t0
            except asyncio.CancelledError:
                raise
            except _DL_RUNTIME_BOUNDARY_ERRORS as exc:
                await self._record_exception(where="loop", exc=exc)
                await asyncio.sleep(0.1)

    def _build_detection_payload(
        self,
        *,
        width: int,
        height: int,
        frame_id: int,
        ts_ms: int,
        detections: list[Any],
        stream_id: str = "",
        stream_epoch: str = "",
    ) -> dict[str, Any]:
        detections = self._apply_detection_filters(detections)
        skeleton_protocol = "none"
        if self._model is not None:
            skeleton_protocol = str(self._model.skeleton_protocol or "").strip() or "none"
        out: list[dict[str, Any]] = []
        frame_size = (int(width), int(height))
        for d in detections:
            x1, y1, x2, y2 = clamp_xyxy(d.xyxy, size=frame_size)
            item: dict[str, Any] = {
                "cls": str(d.cls),
                "score": float(d.conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "keypoints": [],
                "obb": [],
                "skeletonProtocol": skeleton_protocol,
            }
            if d.keypoints:
                item["keypoints"] = [
                    {
                        "x": float(k.x),
                        "y": float(k.y),
                        "score": float(k.score) if k.score is not None else None,
                    }
                    for k in d.keypoints
                ]
            if d.obb:
                item["obb"] = [[float(x), float(y)] for x, y in d.obb]
            out.append(item)
        return {
            "schemaVersion": DETECTION_SCHEMA_VERSION,
            "frameId": int(frame_id),
            "tsMs": int(ts_ms),
            "streamId": str(stream_id),
            "streamEpoch": str(stream_epoch),
            "width": int(width),
            "height": int(height),
            "model": (self._model.model_id if self._model else ""),
            "task": str(self._service_task),
            "skeletonProtocol": skeleton_protocol,
            "detections": out,
        }

    def _build_classification_payload(self, *, frame_id: int, ts_ms: int, topk: list[Any]) -> dict[str, Any]:
        topk_payload = [{"cls": str(x.cls), "score": float(x.score)} for x in topk]
        top1 = topk_payload[0] if topk_payload else {"cls": "", "score": 0.0}
        return {
            "schemaVersion": CLASSIFICATION_SCHEMA_VERSION,
            "frameId": int(frame_id),
            "tsMs": int(ts_ms),
            "model": (self._model.model_id if self._model else ""),
            "top1": top1,
            "topk": topk_payload,
        }
