from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from f8pysdk.bus import ServiceBus
from f8pysdk.codec import coerce_bool, coerce_int, coerce_str
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode
from f8pysdk.time_utils import now_ms
from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32, VIDEO_FORMAT_FLOW2_F16
from f8pysdk.video_transport import ZenohLatestVideoFrameTransport
from f8pysdk.zenoh_naming import zenoh_data_key

from .model_config import ModelSpec, ModelTask, build_model_index, build_model_index_with_errors, load_model_spec
from .onnx_runtime import OnnxNeuFlowRuntime
from .service_paths import default_weights_dir, resolve_path_from_cwd_or_repo
from .video_frame_source import (
    LatestVideoFrameSource,
    VideoFrameSourceConfig,
    video_source_metadata,
)
from .weights_downloader import ensure_onnx_file, onnx_file_matches_sha256


_MISSING_VIDEO_INPUT_GRACE_S = 2.0


_DL_OPTFLOW_ONNX_PROVIDER_ERRORS = (ImportError, RuntimeError, TypeError, ValueError)
# ONNX/runtime/video-source/Zenoh writer calls cross third-party runtime
# boundaries; keep the service loop alive and report details through state.
_DL_OPTFLOW_DOWNLOAD_ERRORS = (Exception,)
_DL_OPTFLOW_RUNTIME_BOUNDARY_ERRORS = (Exception,)
_DL_OPTFLOW_WRITER_ERRORS = (Exception,)


def _default_weights_dir() -> Path:
    return default_weights_dir()


def _resolve_path_from_cwd_or_repo(raw: str) -> Path:
    return resolve_path_from_cwd_or_repo(raw)


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


@dataclass(frozen=True)
class PreparedFlowFrame:
    frame_id: int
    width: int
    height: int
    tensor: Any


class OptflowFramePairCache:
    def __init__(self) -> None:
        self._prev: PreparedFlowFrame | None = None

    def reset(self) -> None:
        self._prev = None

    def push_and_get_pair(self, current: PreparedFlowFrame) -> tuple[PreparedFlowFrame, PreparedFlowFrame] | None:
        prev = self._prev
        self._prev = current
        if prev is None:
            return None
        if prev.width != current.width or prev.height != current.height:
            return None
        return prev, current


def pack_flow2_f16_payload(flow_hw2: Any) -> tuple[int, bytes]:
    import numpy as np  # type: ignore

    flow = np.asarray(flow_hw2, dtype=np.float32)
    if flow.ndim != 3 or int(flow.shape[2]) != 2:
        raise ValueError(f"Flow must have shape HxWx2, got {flow.shape!r}")
    height = int(flow.shape[0])
    width = int(flow.shape[1])
    pitch = int(width * 4)
    flow16 = flow.astype(np.float16)
    payload = np.ascontiguousarray(flow16.view(np.uint8)).reshape((height, pitch))
    return pitch, payload.tobytes(order="C")


class OnnxOptflowServiceNode(ServiceNode):
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
            data_out_ports=["flow"],
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
        self._compute_every_n_frames = 2
        self._auto_download_weights = True
        self._download_retry_at_monotonic = 0.0

        self._video_config = VideoFrameSourceConfig.from_bus(None)
        self._video_source: LatestVideoFrameSource | None = None

        self._flow_format = "flow2_f16"
        self._flow_key = zenoh_data_key(self.node_id, node_id=self.node_id, port_id="flow")
        self._flow_zenoh_writer: ZenohLatestVideoFrameTransport | None = None

        self._runtime: OnnxNeuFlowRuntime | None = None
        self._runtime_yaml: Path | None = None
        self._model: ModelSpec | None = None
        self._last_error = ""
        self._last_error_signature = ""
        self._last_error_repeats = 0
        self._model_index_warning = ""
        self._runtime_warning = ""
        self._last_input_stream_key = ""
        self._using_cached_input_stream_key = False
        self._missing_input_since_monotonic: float | None = None

        self._last_processed_frame_id: int | None = None
        self._last_infer_frame_id: int | None = None
        self._dup_skipped_since_last_processed = 0
        self._new_frame_counter = 0

        self._frame_cache = OptflowFramePairCache()

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        self._video_config = VideoFrameSourceConfig.from_bus(bus)
        self._video_source = LatestVideoFrameSource(config=self._video_config)
        if isinstance(bus, ServiceBus):
            self._flow_key = zenoh_data_key(bus.service_id, node_id=self.node_id, port_id="flow")
        loop = asyncio.get_running_loop()
        self._init_task = loop.create_task(self._ensure_config_loaded(), name=f"f8dl-optflow:init:{self.node_id}")
        self._task = loop.create_task(self._loop(), name=f"f8dl-optflow:loop:{self.node_id}")

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
        self._close_flow_writer()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        await self._ensure_config_loaded()

        if name == "weightsDir":
            raw = coerce_str(value, default=str(self._weights_dir))
            self._weights_dir = _resolve_path_from_cwd_or_repo(raw)
            await self._publish_model_index()
            await self._reset_runtime()
            return

        if name == "modelId":
            self._model_id = coerce_str(value, default=self._model_id)
            await self._reset_runtime()
            return

        if name == "modelYamlPath":
            self._model_yaml_path = coerce_str(value, default=self._model_yaml_path)
            await self._reset_runtime()
            return

        if name == "ortProvider":
            v = coerce_str(value, default=str(self._ort_provider)).lower()
            self._ort_provider = v if v in ("auto", "cuda", "cpu") else "auto"
            await self._reset_runtime()
            return

        if name == "computeEveryNFrames":
            self._compute_every_n_frames = coerce_int(
                value,
                default=self._compute_every_n_frames,
                minimum=1,
                maximum=120,
            )
            return

        if name == "autoDownloadWeights":
            self._auto_download_weights = coerce_bool(
                value,
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
        self._compute_every_n_frames = coerce_int(
            await self.get_state_value("computeEveryNFrames"),
            default=int(self._initial_state.get("computeEveryNFrames") or 2),
            minimum=1,
            maximum=120,
        )
        self._auto_download_weights = coerce_bool(
            await self.get_state_value("autoDownloadWeights"),
            default=bool(self._initial_state.get("autoDownloadWeights", True)),
        )
        self._config_loaded = True
        if self.input_zenoh_key("video"):
            await self._clear_missing_input_error()
        await self.set_state("flowFormat", self._flow_format)
        await self._publish_model_index()

    async def _publish_model_index(self) -> None:
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
        normalized = str(message or "")
        if normalized == self._last_error:
            return
        self._last_error = normalized
        if self._last_error:
            await self.report_error(
                "DL_OPTFLOW_RUNTIME",
                self._last_error,
                severity="warning" if self._last_error.lower().startswith("downloading ") else "error",
                fingerprint=f"dl-optflow:{self._last_error}",
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

    @staticmethod
    def _should_fallback_to_cpu(exc: Exception) -> bool:
        message = str(exc).lower()
        if "cudnn" in message:
            return True
        if "cuda" in message and ("execution_failed" in message or "non-zero status code" in message):
            return True
        return False

    async def _fallback_to_cpu_after_gpu_error(self, *, exc: Exception) -> None:
        if self._ort_provider == "cpu":
            await self._record_exception(where="loop", exc=exc)
            return
        if self._ort_provider == "cuda":
            await self._record_exception(where="loop", exc=exc)
            return
        detail = f"{type(exc).__name__}: {exc}"
        await self._reset_runtime()
        self._ort_provider = "cpu"
        await self.set_state("ortProvider", "cpu")
        await self._set_last_error(
            "GPU inference failed; switched ortProvider to cpu automatically.\n" f"reason: {detail}"
        )

    def _close_video_source(self) -> None:
        source = self._video_source
        self._video_source = None
        if source is not None:
            source.close()

    def _ensure_video_source(self) -> LatestVideoFrameSource:
        source = self._video_source
        if source is not None:
            return source
        source = LatestVideoFrameSource(config=self._video_config)
        self._video_source = source
        return source

    def _reset_input_source(self) -> None:
        source = self._video_source
        if source is not None:
            source.reset()
        self._frame_cache.reset()
        self._new_frame_counter = 0
        self._last_processed_frame_id = None
        self._last_infer_frame_id = None
        self._dup_skipped_since_last_processed = 0

    def _current_input_stream_key(self) -> str:
        return str(self.input_zenoh_key("video") or "").strip()

    def _has_rungraph(self) -> bool:
        return self.has_rungraph()

    async def _clear_missing_input_error(self) -> None:
        self._missing_input_since_monotonic = None
        if self._last_error in (
            "missing video data input",
        ):
            await self._set_last_error("")
            return
        if not self._last_error:
            await self.clear_error()

    async def _handle_missing_input_stream_key(self) -> None:
        if not self._has_rungraph():
            self._missing_input_since_monotonic = None
            await asyncio.sleep(0.05)
            return
        now = time.monotonic()
        missing_since = self._missing_input_since_monotonic
        if missing_since is None:
            self._missing_input_since_monotonic = now
            await asyncio.sleep(0.05)
            return
        if (float(now) - float(missing_since)) >= _MISSING_VIDEO_INPUT_GRACE_S:
            await self._set_last_error("missing video data input")
        await asyncio.sleep(0.05)

    def _resolve_input_stream_key(self) -> str:
        key = self._current_input_stream_key()
        if key:
            self._last_input_stream_key = key
            self._using_cached_input_stream_key = False
            return key
        cached_key = str(self._last_input_stream_key or "").strip()
        if not cached_key:
            self._using_cached_input_stream_key = False
            return ""
        now = time.monotonic()
        missing_since = self._missing_input_since_monotonic
        if missing_since is None:
            self._missing_input_since_monotonic = now
            self._using_cached_input_stream_key = True
            return cached_key
        if (float(now) - float(missing_since)) < _MISSING_VIDEO_INPUT_GRACE_S:
            self._using_cached_input_stream_key = True
            return cached_key
        self._using_cached_input_stream_key = False
        return ""

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

        runtime = OnnxNeuFlowRuntime(spec, ort_provider=self._ort_provider)
        self._runtime = runtime
        self._runtime_yaml = yaml_path
        self._model = spec
        self._frame_cache.reset()
        self._new_frame_counter = 0
        self._last_infer_frame_id = None
        providers = runtime.active_providers
        await self.set_state("loadedModel", f"{spec.model_id} ({spec.task})")
        await self.set_state("ortActiveProviders", json.dumps(providers))
        await self.set_state("flowFormat", self._flow_format)

        warn_parts: list[str] = []
        if runtime.provider_warning:
            warn_parts.append(str(runtime.provider_warning))
        prefer = str(self._ort_provider or "auto").lower()
        if prefer in ("auto", "cuda"):
            try:
                import onnxruntime as ort  # type: ignore

                available = list(ort.get_available_providers())  # type: ignore[attr-defined]
            except _DL_OPTFLOW_ONNX_PROVIDER_ERRORS as exc:
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
        self._runtime_warning = "\n".join([x for x in warn_parts if str(x).strip()]).strip()
        await self._set_last_error(self._runtime_warning)
        return True

    async def _reset_runtime(self) -> None:
        self._runtime = None
        self._runtime_yaml = None
        self._model = None
        self._frame_cache.reset()
        self._new_frame_counter = 0
        self._last_processed_frame_id = None
        self._last_infer_frame_id = None
        self._dup_skipped_since_last_processed = 0
        self._close_flow_writer()
        self._last_error_signature = ""
        self._last_error_repeats = 0
        self._runtime_warning = ""
        await self.set_state("loadedModel", "")
        await self.clear_error()
        await self.set_state("ortActiveProviders", "")
        await self.set_state("flowFormat", self._flow_format)
        if self._model_index_warning:
            await self._set_last_error(self._model_index_warning)

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
        except _DL_OPTFLOW_DOWNLOAD_ERRORS as exc:
            self._download_retry_at_monotonic = time.monotonic() + 30.0
            raise RuntimeError(
                f"Auto-download failed for model={spec.model_id!r} path={spec.onnx_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _close_flow_writer(self) -> None:
        zenoh_writer = self._flow_zenoh_writer
        self._flow_zenoh_writer = None
        if zenoh_writer is not None:
            zenoh_writer.close()

    def _ensure_flow_zenoh_writer(self) -> ZenohLatestVideoFrameTransport:
        writer = self._flow_zenoh_writer
        if writer is not None:
            return writer
        writer = ZenohLatestVideoFrameTransport.open_publisher(
            self._flow_key,
            config_path=self._video_config.config_path,
            connect=self._video_config.connect,
            listen=self._video_config.listen,
            shm_pool_bytes=self._video_config.shm_pool_bytes,
        )
        self._flow_zenoh_writer = writer
        return writer

    def _publish_flow_zenoh(self, *, width: int, height: int, pitch: int, payload: bytes, ts_ms: int) -> None:
        writer = self._ensure_flow_zenoh_writer()
        writer.publish_frame(
            width=int(width),
            height=int(height),
            pitch=int(pitch),
            payload=payload,
            fmt=VIDEO_FORMAT_FLOW2_F16,
            ts_ms=int(ts_ms),
        )

    def _record_output_timing(self, *, started_at: float, source_ts_ms: int) -> None:
        completed_ts_ms = int(now_ms())
        process_ms = max(0.0, (time.perf_counter() - float(started_at)) * 1000.0)
        latency_ms = 0.0
        if int(source_ts_ms) > 0:
            latency_ms = max(0.0, float(completed_ts_ms - int(source_ts_ms)))
        self.record_monitor_timing(
            port="flow",
            process_ms=process_ms,
            latency_ms=latency_ms,
            ts_ms=completed_ts_ms,
        )

    async def _loop(self) -> None:
        import numpy as np  # type: ignore

        while True:
            try:
                await asyncio.sleep(0)
                if not self._active:
                    await asyncio.sleep(0.05)
                    continue

                await self._ensure_config_loaded()

                input_stream_key = self._resolve_input_stream_key()
                if not input_stream_key:
                    await self._handle_missing_input_stream_key()
                    continue
                if self._missing_input_since_monotonic is not None and not self._using_cached_input_stream_key:
                    await self._clear_missing_input_error()

                try:
                    await self._ensure_runtime()
                except _DL_OPTFLOW_RUNTIME_BOUNDARY_ERRORS as exc:
                    await self._record_exception(where="ensure_runtime", exc=exc)
                    await asyncio.sleep(0.1)
                    continue

                runtime = self._runtime
                if runtime is None:
                    await asyncio.sleep(0.05)
                    continue
                source = self._ensure_video_source()
                t0 = time.perf_counter()
                try:
                    frame = source.read_latest(stream_key=input_stream_key, timeout_ms=10)
                except _DL_OPTFLOW_RUNTIME_BOUNDARY_ERRORS as exc:
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
                frame_ts_ms = int(frame.ts_ms)

                try:
                    buf = np.frombuffer(frame.payload, dtype=np.uint8)
                    rows = buf.reshape((height, pitch))
                    bgra = rows[:, : width * 4].reshape((height, width, 4))
                    frame_bgr = bgra[:, :, 0:3]

                    tensor = runtime.prepare_input(frame_bgr)
                    pair = self._frame_cache.push_and_get_pair(
                        PreparedFlowFrame(
                            frame_id=frame_id_seen,
                            width=width,
                            height=height,
                            tensor=tensor,
                        )
                    )
                    if pair is None:
                        if self._last_error and not self._runtime_warning:
                            await self._set_last_error("")
                        continue

                    if (int(self._new_frame_counter) % int(self._compute_every_n_frames)) != 0:
                        continue

                    prev_frame, current_frame = pair
                    try:
                        flow = runtime.infer_preprocessed(
                            prev_frame.tensor,
                            current_frame.tensor,
                            output_size_hw=(height, width),
                        )
                    except _DL_OPTFLOW_RUNTIME_BOUNDARY_ERRORS as exc:
                        if self._should_fallback_to_cpu(exc):
                            await self._fallback_to_cpu_after_gpu_error(exc=exc)
                            await asyncio.sleep(0.1)
                            continue
                        raise
                    flow_pitch, flow_payload = pack_flow2_f16_payload(flow)
                finally:
                    frame.release()
                try:
                    self._publish_flow_zenoh(
                        width=width,
                        height=height,
                        pitch=flow_pitch,
                        payload=flow_payload,
                        ts_ms=frame_ts_ms,
                    )
                except _DL_OPTFLOW_WRITER_ERRORS as exc:
                    await self._record_exception(where="publish_flow_zenoh", exc=exc)
                    await asyncio.sleep(0.1)
                    continue
                t_infer1 = time.perf_counter()
                self.record_monitor_processed(port="flow")
                self._record_output_timing(started_at=t0, source_ts_ms=frame_ts_ms)
                if self._runtime_warning:
                    await self._set_last_error(self._runtime_warning)
                elif self._last_error:
                    await self._set_last_error("")

                self._last_infer_frame_id = frame_id_seen
                _ = t0
                _ = t_infer1
            except asyncio.CancelledError:
                raise
            except _DL_OPTFLOW_RUNTIME_BOUNDARY_ERRORS as exc:
                await self._record_exception(where="loop", exc=exc)
                await asyncio.sleep(0.1)
