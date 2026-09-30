from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from f8pysdk.codec import coerce_bool, coerce_float, coerce_int, coerce_str
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode
from f8pysdk.time_utils import now_ms
from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32

from .model_config import ModelSpec, ModelTask, build_model_index, build_model_index_with_errors, load_model_spec
from .onnx_runtime import OnnxTemporalWaveRuntime
from .service_paths import default_weights_dir, resolve_user_path
from .video_frame_source import (
    LatestVideoFrameSource,
    VideoFrameSourceConfig,
    video_source_metadata,
)
from .weights_downloader import ensure_onnx_file, onnx_file_matches_sha256

_VR_FOCUS_TOP = 0.20
_VR_FOCUS_BOTTOM = 0.0
_VR_FOCUS_LEFT = 0.10
_VR_FOCUS_RIGHT = 0.10
_DL_TCN_ONNX_PROVIDER_ERRORS = (ImportError, RuntimeError, TypeError, ValueError)
# ONNX/runtime/video-source calls cross third-party runtime boundaries; keep
# the service loop alive and report details through state.
_DL_TCN_DOWNLOAD_ERRORS = (Exception,)
_DL_TCN_RUNTIME_BOUNDARY_ERRORS = (Exception,)


def apply_vr_focus_crop(frame: Any) -> Any:
    if frame.ndim < 2:
        return frame
    height = int(frame.shape[0])
    width = int(frame.shape[1])
    if height <= 0 or width <= 0:
        return frame
    top = min(height - 1, max(0, int(round(height * _VR_FOCUS_TOP))))
    bottom = min(height, max(top + 1, height - int(round(height * _VR_FOCUS_BOTTOM))))
    left = min(width - 1, max(0, int(round(width * _VR_FOCUS_LEFT))))
    right = min(width, max(left + 1, width - int(round(width * _VR_FOCUS_RIGHT))))
    if bottom <= top or right <= left:
        return frame
    return frame[top:bottom, left:right]


def _default_weights_dir() -> Path:
    return default_weights_dir()


def _resolve_user_path(raw: str) -> Path:
    return resolve_user_path(raw)


@dataclass(frozen=True)
class AggregatedTemporalValue:
    frame_index: int
    frame_id: int
    ts_ms: int
    value: float


class DelayedAverageAggregator:
    def __init__(self) -> None:
        self._next_frame_index = 0
        self._next_emit_index = 0
        self._sum_by_index: dict[int, float] = {}
        self._count_by_index: dict[int, int] = {}
        self._frame_id_by_index: dict[int, int] = {}
        self._ts_by_index: dict[int, int] = {}

    def reset(self) -> None:
        self._next_frame_index = 0
        self._next_emit_index = 0
        self._sum_by_index.clear()
        self._count_by_index.clear()
        self._frame_id_by_index.clear()
        self._ts_by_index.clear()

    def register_frame(self, *, frame_id: int, ts_ms: int) -> int:
        idx = int(self._next_frame_index)
        self._next_frame_index = idx + 1
        self._frame_id_by_index[idx] = int(frame_id)
        self._ts_by_index[idx] = int(ts_ms)
        return idx

    def apply_window(self, *, window_end_index: int, values: list[float]) -> int:
        output_length = int(len(values))
        if output_length <= 0:
            raise ValueError("Temporal output must contain at least one value")
        start_index = int(window_end_index) - output_length + 1
        for offset, value in enumerate(values):
            target_index = start_index + int(offset)
            if target_index < 0:
                continue
            prev_sum = float(self._sum_by_index.get(target_index, 0.0))
            prev_count = int(self._count_by_index.get(target_index, 0))
            self._sum_by_index[target_index] = prev_sum + float(value)
            self._count_by_index[target_index] = prev_count + 1
        return output_length

    def pop_ready(self, *, latest_window_end_index: int, output_length: int) -> list[AggregatedTemporalValue]:
        length = max(1, int(output_length))
        ready_cutoff = int(latest_window_end_index) - (length - 1)
        if ready_cutoff < int(self._next_emit_index):
            return []
        out: list[AggregatedTemporalValue] = []
        while self._next_emit_index <= ready_cutoff:
            idx = int(self._next_emit_index)
            if idx not in self._ts_by_index:
                break
            frame_id = int(self._frame_id_by_index.get(idx, 0))
            ts_ms = int(self._ts_by_index.get(idx, 0))
            value_sum = float(self._sum_by_index.pop(idx, 0.0))
            value_count = int(self._count_by_index.pop(idx, 0))
            value = value_sum / float(value_count) if value_count > 0 else 0.0
            out.append(
                AggregatedTemporalValue(
                    frame_index=idx,
                    frame_id=frame_id,
                    ts_ms=ts_ms,
                    value=float(value),
                )
            )
            self._frame_id_by_index.pop(idx, None)
            self._ts_by_index.pop(idx, None)
            self._next_emit_index = idx + 1
        return out


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
            _, value = self._q.popleft()
            self._sum -= float(value)

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


class OnnxTcnWaveServiceNode(ServiceNode):
    def __init__(
        self,
        *,
        node_id: str,
        node: Any,
        initial_state: dict[str, Any] | None,
        service_class: str,
        allowed_tasks: set[ModelTask],
    ) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=["predictedChange"],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._service_class = str(service_class)
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
        self._output_scale = 10.0
        self._output_bias = 0.0
        self._use_vr_focus_crop = False
        self._auto_download_weights = True
        self._download_retry_at_monotonic = 0.0

        self._video_source: LatestVideoFrameSource | None = None

        self._runtime: OnnxTemporalWaveRuntime | None = None
        self._runtime_yaml: Path | None = None
        self._model: ModelSpec | None = None
        self._last_error = ""
        self._last_error_signature = ""
        self._last_error_repeats = 0
        self._model_index_warning = ""
        self._runtime_warning = ""

        self._window: deque[Any] = deque()
        self._sequence_buffer: Any = None
        self._aggregator = DelayedAverageAggregator()
        self._new_frame_counter = 0
        self._last_processed_frame_id: int | None = None
        self._last_infer_frame_id: int | None = None
        self._dup_skipped_since_last_processed = 0

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        self._video_source = LatestVideoFrameSource(config=VideoFrameSourceConfig.from_bus(bus))
        loop = asyncio.get_running_loop()
        self._init_task = loop.create_task(self._ensure_config_loaded(), name=f"f8dl-tcn:init:{self.node_id}")
        self._task = loop.create_task(self._loop(), name=f"f8dl-tcn:loop:{self.node_id}")

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
        self._window.clear()
        self._sequence_buffer = None
        self._aggregator.reset()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del value
        del ts_ms
        name = str(field or "").strip()
        await self._ensure_config_loaded()

        if name == "weightsDir":
            raw = coerce_str(await self.get_state_value("weightsDir"), default=str(self._weights_dir))
            self._weights_dir = _resolve_user_path(raw)
            await self._publish_model_index()
            await self._reset_runtime()
            return

        if name == "modelId":
            self._model_id = coerce_str(await self.get_state_value("modelId"), default=self._model_id)
            await self._reset_runtime()
            return

        if name == "modelYamlPath":
            self._model_yaml_path = coerce_str(await self.get_state_value("modelYamlPath"), default=self._model_yaml_path)
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

        if name == "outputScale":
            self._output_scale = coerce_float(
                await self.get_state_value("outputScale"),
                default=self._output_scale,
            )
            await self._reset_runtime()
            return

        if name == "outputBias":
            self._output_bias = coerce_float(
                await self.get_state_value("outputBias"),
                default=self._output_bias,
            )
            await self._reset_runtime()
            return

        if name == "useVrFocusCrop":
            self._use_vr_focus_crop = coerce_bool(
                await self.get_state_value("useVrFocusCrop"),
                default=self._use_vr_focus_crop,
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
        self._weights_dir = _resolve_user_path(raw_weights)
        self._model_id = coerce_str(await self.get_state_value("modelId"), default=str(self._initial_state.get("modelId") or ""))
        self._model_yaml_path = coerce_str(
            await self.get_state_value("modelYamlPath"),
            default=str(self._initial_state.get("modelYamlPath") or ""),
        )
        provider = coerce_str(await self.get_state_value("ortProvider"), default=str(self._initial_state.get("ortProvider") or "auto")).lower()
        self._ort_provider = provider if provider in ("auto", "cuda", "cpu") else "auto"
        self._infer_every_n = coerce_int(
            await self.get_state_value("inferEveryN"),
            default=int(self._initial_state.get("inferEveryN") or 1),
            minimum=1,
            maximum=10000,
        )
        self._output_scale = coerce_float(
            await self.get_state_value("outputScale"),
            default=float(self._initial_state.get("outputScale") or 10.0),
        )
        self._output_bias = coerce_float(
            await self.get_state_value("outputBias"),
            default=float(self._initial_state.get("outputBias") or 0.0),
        )
        self._use_vr_focus_crop = coerce_bool(
            await self.get_state_value("useVrFocusCrop"),
            default=bool(self._initial_state.get("useVrFocusCrop", False)),
        )
        self._auto_download_weights = coerce_bool(
            await self.get_state_value("autoDownloadWeights"),
            default=bool(self._initial_state.get("autoDownloadWeights", True)),
        )
        self._config_loaded = True
        await self._publish_model_index()

    async def _publish_model_index(self) -> None:
        idx, errors = build_model_index_with_errors(self._weights_dir, allowed_tasks=self._allowed_tasks)
        warning = ""
        if errors:
            preview = errors[:3]
            parts: list[str] = []
            for item in preview:
                path_name = Path(str(item.get("path") or "")).name
                error_text = str(item.get("error") or "").strip()
                if path_name and error_text:
                    parts.append(f"{path_name}: {error_text}")
                elif error_text:
                    parts.append(error_text)
            warning = f"Skipped {len(errors)} invalid model yaml(s)."
            if parts:
                warning = f"{warning} {' | '.join(parts)}"
            remain = int(len(errors) - len(preview))
            if remain > 0:
                warning = f"{warning} | +{remain} more"
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

        payload = [item.model_id for item in idx]
        await self.set_state("availableModels", payload)
        if idx:
            available = set(payload)
            if not self._model_id or self._model_id not in available:
                self._model_id = idx[0].model_id
                await self.set_state("modelId", self._model_id)
        else:
            self._model_id = ""
            await self.set_state("modelId", self._model_id)

    async def _set_last_error(self, message: str) -> None:
        self._last_error = str(message or "")
        if self._last_error:
            await self.report_error(
                "DL_TCNWAVE_RUNTIME",
                self._last_error,
                severity="warning" if self._last_error.lower().startswith("downloading ") else "error",
                fingerprint=f"dl-tcnwave:{self._last_error}",
            )
            return
        await self.clear_error()

    async def _handle_missing_video_input(self) -> None:
        message = "missing video data input"
        await self._set_last_error(message)

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
        self._runtime = None
        self._runtime_yaml = None
        self._model = None
        self._window.clear()
        self._sequence_buffer = None
        self._aggregator.reset()
        self._new_frame_counter = 0
        self._last_processed_frame_id = None
        self._last_infer_frame_id = None
        self._dup_skipped_since_last_processed = 0
        self._runtime_warning = ""
        self._last_error_signature = ""
        self._last_error_repeats = 0
        await self.set_state("loadedModel", "")
        await self.clear_error()
        await self.set_state("ortActiveProviders", "")
        if self._model_index_warning:
            await self._set_last_error(self._model_index_warning)

    def _reset_video_source(self) -> None:
        source = self._video_source
        if source is not None:
            source.reset()
        self._window.clear()
        self._sequence_buffer = None
        self._aggregator.reset()
        self._new_frame_counter = 0
        self._last_processed_frame_id = None
        self._last_infer_frame_id = None
        self._dup_skipped_since_last_processed = 0

    def _resolve_video_stream_key(self) -> str:
        return self.input_zenoh_key("video") or ""

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

    def _resolve_model_yaml(self) -> Path:
        if self._model_yaml_path:
            return _resolve_user_path(self._model_yaml_path)
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

    def _record_output_timing(self, *, started_at: float, source_ts_ms: int) -> None:
        completed_ts_ms = int(now_ms())
        process_ms = max(0.0, (time.perf_counter() - float(started_at)) * 1000.0)
        latency_ms = 0.0
        if int(source_ts_ms) > 0:
            latency_ms = max(0.0, float(completed_ts_ms - int(source_ts_ms)))
        self.record_monitor_timing(
            port="predictedChange",
            process_ms=process_ms,
            latency_ms=latency_ms,
            ts_ms=completed_ts_ms,
        )

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
                f"Model file not found: {spec.onnx_path}. "
                "No onnxUrl is configured in model yaml."
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
        except _DL_TCN_DOWNLOAD_ERRORS as exc:
            self._download_retry_at_monotonic = time.monotonic() + 30.0
            raise RuntimeError(
                f"Auto-download failed for model={spec.model_id!r} path={spec.onnx_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    async def _ensure_runtime(self) -> bool:
        if self._runtime is not None:
            return True

        yaml_path = self._resolve_model_yaml()
        spec = load_model_spec(yaml_path)
        await self._ensure_onnx_available(spec)
        if spec.task not in self._allowed_tasks:
            raise ValueError(
                f"Model task mismatch: model task={spec.task!r}, service class={self._service_class!r}, "
                f"allowed={sorted(self._allowed_tasks)!r}"
            )

        runtime = OnnxTemporalWaveRuntime(
            spec,
            ort_provider=self._ort_provider,
            output_scale=self._output_scale,
            output_bias=self._output_bias,
        )
        self._runtime = runtime
        self._runtime_yaml = yaml_path
        self._model = spec
        self._window = deque(maxlen=runtime.sequence_length)
        self._sequence_buffer = None
        self._aggregator.reset()
        self._new_frame_counter = 0
        self._last_processed_frame_id = None
        self._last_infer_frame_id = None
        self._dup_skipped_since_last_processed = 0

        providers = runtime.active_providers
        await self.set_state("loadedModel", f"{spec.model_id} ({spec.task})")
        await self.set_state("ortActiveProviders", json.dumps(providers))

        warning_parts: list[str] = []
        if runtime.provider_warning:
            warning_parts.append(str(runtime.provider_warning))
        prefer = str(self._ort_provider or "auto").lower()
        if prefer in ("auto", "cuda"):
            try:
                import onnxruntime as ort  # type: ignore

                available = list(ort.get_available_providers())  # type: ignore[attr-defined]
            except _DL_TCN_ONNX_PROVIDER_ERRORS as exc:
                available = []
                warning_parts.append(f"Failed to query ORT available providers: {type(exc).__name__}: {exc}")
            active_lower = {str(item).lower() for item in (providers or [])}
            available_lower = {str(item).lower() for item in (available or [])}
            if "cudaexecutionprovider" not in active_lower and "cudaexecutionprovider" not in available_lower:
                warning_parts.append(
                    "CUDAExecutionProvider is not available in this runtime. "
                    f"activeProviders={providers!r}, availableProviders={available!r}."
                )
        if self._model_index_warning:
            warning_parts.append(self._model_index_warning)
        self._runtime_warning = "\n".join([item for item in warning_parts if str(item).strip()]).strip()
        await self._set_last_error(self._runtime_warning)
        return True

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
                except _DL_TCN_RUNTIME_BOUNDARY_ERRORS as exc:
                    await self._record_exception(where="ensure_runtime", exc=exc)
                    await asyncio.sleep(0.1)
                    continue

                video_stream_key = self._resolve_video_stream_key()
                if not video_stream_key:
                    await self._handle_missing_video_input()
                    await asyncio.sleep(0.05)
                    continue

                runtime = self._runtime
                if runtime is None:
                    await asyncio.sleep(0.05)
                    continue
                source = self._ensure_video_source()
                t0 = time.perf_counter()
                try:
                    frame = source.read_latest(stream_key=video_stream_key, timeout_ms=10)
                except _DL_TCN_RUNTIME_BOUNDARY_ERRORS as exc:
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
                self._new_frame_counter += 1

                try:
                    buf = np.frombuffer(frame.payload, dtype=np.uint8)
                    rows = buf.reshape((height, pitch))
                    bgra = rows[:, : width * 4].reshape((height, width, 4))
                    frame_bgr = bgra[:, :, 0:3]
                    if self._use_vr_focus_crop and int(frame_bgr.shape[1]) > 1:
                        frame_bgr = apply_vr_focus_crop(frame_bgr)

                    prepared = runtime.prepare_frame(frame_bgr)
                    self._window.append(prepared)
                    frame_index = self._aggregator.register_frame(frame_id=frame_id_seen, ts_ms=int(frame.ts_ms))

                    if len(self._window) < int(runtime.sequence_length):
                        await self._set_last_error(
                            f"warming up temporal window: {len(self._window)}/{runtime.sequence_length}"
                        )
                        continue

                    do_infer = self._last_infer_frame_id is None or (
                        int(self._new_frame_counter) % int(self._infer_every_n)
                    ) == 0
                    if not do_infer:
                        continue

                    sequence_buffer = self._sequence_buffer
                    if sequence_buffer is None:
                        sequence_buffer = np.empty(
                            (
                                int(runtime.sequence_length),
                                int(runtime.channels),
                                int(runtime.input_height),
                                int(runtime.input_width),
                            ),
                            dtype=np.float32,
                        )
                        self._sequence_buffer = sequence_buffer
                    for sequence_index, prepared_frame in enumerate(self._window):
                        sequence_buffer[sequence_index] = prepared_frame
                    sequence = sequence_buffer
                    values_np = runtime.infer_sequence(sequence)
                    values = self._to_float_list(values_np.tolist())
                    output_length = self._aggregator.apply_window(window_end_index=frame_index, values=values)
                    ready = self._aggregator.pop_ready(
                        latest_window_end_index=frame_index,
                        output_length=output_length,
                    )
                    for item in ready:
                        await self.emit("predictedChange", float(item.value), ts_ms=int(item.ts_ms))
                        self._record_output_timing(
                            started_at=t0,
                            source_ts_ms=int(item.ts_ms),
                        )

                    t_infer1 = time.perf_counter()
                finally:
                    frame.release()
                self._last_infer_frame_id = frame_id_seen
                if self._runtime_warning:
                    await self._set_last_error(self._runtime_warning)
                elif self._last_error:
                    await self._set_last_error("")
                _ = t0
                _ = t_infer1
            except asyncio.CancelledError:
                raise
            except _DL_TCN_RUNTIME_BOUNDARY_ERRORS as exc:
                await self._record_exception(where="loop", exc=exc)
                await asyncio.sleep(0.1)

    @staticmethod
    def _to_float_list(values: list[Any]) -> list[float]:
        out: list[float] = []
        for value in values:
            out.append(float(value))
        return out
