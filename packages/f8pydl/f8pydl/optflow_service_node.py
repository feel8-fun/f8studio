from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import ServiceNode
from f8pysdk.shm.video import VIDEO_FORMAT_BGRA32, VIDEO_FORMAT_FLOW2_F16, VideoShmReader, VideoShmWriter

from .model_config import ModelSpec, ModelTask, build_model_index, build_model_index_with_errors, load_model_spec
from .onnx_runtime import OnnxNeuFlowRuntime
from .weights_downloader import ensure_onnx_file


def _default_weights_dir() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append((Path.cwd() / "services" / "f8" / "dl" / "weights").resolve())
    except Exception:
        pass
    try:
        root = Path(__file__).resolve().parents[3]
        candidates.append((root / "services" / "f8" / "dl" / "weights").resolve())
    except Exception:
        pass
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_dir():
                return candidate
        except Exception:
            continue
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


def _coerce_str(v: Any, *, default: str = "") -> str:
    try:
        s = str(v) if v is not None else ""
    except Exception:
        s = ""
    s = s.strip()
    return s if s else default


def _coerce_bool(v: Any, *, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    return bool(default)


def _resolve_path_from_cwd_or_repo(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    p1 = (Path.cwd() / p).resolve()
    if p1.exists():
        return p1
    try:
        root = Path(__file__).resolve().parents[3]
        p2 = (root / p).resolve()
        if p2.exists():
            return p2
    except Exception:
        pass
    return p1


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
        shm_name: str,
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
            "source": {"shmName": str(shm_name)},
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
            data_out_ports=["telemetry"],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._service_class = str(service_class)
        self._allowed_tasks = set(allowed_tasks)

        self._active = True
        self._config_loaded = False
        self._task: asyncio.Task[object] | None = None

        self._weights_dir = _default_weights_dir()
        self._model_yaml_path = ""
        self._model_id = ""
        self._ort_provider: Literal["auto", "cuda", "cpu"] = "auto"
        self._input_shm_name = ""
        self._compute_every_n_frames = 2
        self._auto_download_weights = True
        self._download_retry_at_monotonic = 0.0

        self._shm: VideoShmReader | None = None
        self._shm_open_name = ""

        self._flow_shm_name = f"shm.{self.node_id}.flow"
        self._flow_shm_format = "flow2_f16"
        self._flow_writer: VideoShmWriter | None = None
        self._flow_writer_pitch = 0
        self._flow_writer_width = 0
        self._flow_writer_height = 0

        self._runtime: OnnxNeuFlowRuntime | None = None
        self._runtime_yaml: Path | None = None
        self._model: ModelSpec | None = None
        self._last_error = ""
        self._last_error_signature = ""
        self._last_error_repeats = 0
        self._model_index_warning = ""
        self._runtime_warning = ""

        self._last_processed_frame_id: int | None = None
        self._last_infer_frame_id: int | None = None
        self._dup_skipped_since_last_processed = 0
        self._new_frame_counter = 0

        self._frame_cache = OptflowFramePairCache()
        self._telemetry = _Telemetry()

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        loop = asyncio.get_running_loop()
        loop.create_task(self._ensure_config_loaded(), name=f"f8dl-optflow:init:{self.node_id}")
        self._task = loop.create_task(self._loop(), name=f"f8dl-optflow:loop:{self.node_id}")

    async def close(self) -> None:
        t = self._task
        self._task = None
        if t is not None:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
        self._close_shm()
        self._close_flow_writer()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del value
        del ts_ms
        name = str(field or "").strip()
        await self._ensure_config_loaded()

        if name == "weightsDir":
            raw = _coerce_str(await self.get_state_value("weightsDir"), default=str(self._weights_dir))
            self._weights_dir = _resolve_path_from_cwd_or_repo(raw)
            await self._publish_model_index()
            await self._reset_runtime()
            return

        if name == "modelId":
            self._model_id = _coerce_str(await self.get_state_value("modelId"), default=self._model_id)
            await self._reset_runtime()
            return

        if name == "modelYamlPath":
            self._model_yaml_path = _coerce_str(await self.get_state_value("modelYamlPath"), default=self._model_yaml_path)
            await self._reset_runtime()
            return

        if name == "ortProvider":
            v = _coerce_str(await self.get_state_value("ortProvider"), default=str(self._ort_provider)).lower()
            self._ort_provider = v if v in ("auto", "cuda", "cpu") else "auto"
            await self._reset_runtime()
            return

        if name == "inputShmName":
            self._input_shm_name = _coerce_str(await self.get_state_value("inputShmName"), default=self._input_shm_name)
            self._frame_cache.reset()
            self._new_frame_counter = 0
            self._last_processed_frame_id = None
            self._last_infer_frame_id = None
            self._dup_skipped_since_last_processed = 0
            await self._maybe_reopen_shm()
            return

        if name == "computeEveryNFrames":
            self._compute_every_n_frames = _coerce_int(
                await self.get_state_value("computeEveryNFrames"),
                default=self._compute_every_n_frames,
                minimum=1,
                maximum=120,
            )
            return

        if name == "autoDownloadWeights":
            self._auto_download_weights = _coerce_bool(
                await self.get_state_value("autoDownloadWeights"),
                default=self._auto_download_weights,
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

        raw_weights = _coerce_str(
            await self.get_state_value("weightsDir"),
            default=str(self._initial_state.get("weightsDir") or _default_weights_dir()),
        )
        self._weights_dir = _resolve_path_from_cwd_or_repo(raw_weights)
        self._model_id = _coerce_str(await self.get_state_value("modelId"), default=str(self._initial_state.get("modelId") or ""))
        self._model_yaml_path = _coerce_str(
            await self.get_state_value("modelYamlPath"),
            default=str(self._initial_state.get("modelYamlPath") or ""),
        )
        v = _coerce_str(await self.get_state_value("ortProvider"), default=str(self._initial_state.get("ortProvider") or "auto")).lower()
        self._ort_provider = v if v in ("auto", "cuda", "cpu") else "auto"
        self._input_shm_name = _coerce_str(
            await self.get_state_value("inputShmName"),
            default=str(self._initial_state.get("inputShmName") or ""),
        )
        self._compute_every_n_frames = _coerce_int(
            await self.get_state_value("computeEveryNFrames"),
            default=int(self._initial_state.get("computeEveryNFrames") or 2),
            minimum=1,
            maximum=120,
        )
        self._auto_download_weights = _coerce_bool(
            await self.get_state_value("autoDownloadWeights"),
            default=bool(self._initial_state.get("autoDownloadWeights", True)),
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
        await self.set_state("flowShmName", self._flow_shm_name)
        await self.set_state("flowShmFormat", self._flow_shm_format)
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
        self._last_error = str(message or "")
        await self.set_state("lastError", self._last_error)

    async def _emit_idle_telemetry(self, *, now_ms: int, shm_name: str) -> None:
        if not self._telemetry.should_emit(now_ms):
            return
        telemetry_payload = self._telemetry.summary(
            now_ms=now_ms,
            node_id=self.node_id,
            service_class=self._service_class,
            model=self._model,
            ort_provider=self._ort_provider,
            shm_name=shm_name,
            frame_id_last_seen=self._last_processed_frame_id,
            frame_id_last_processed=self._last_processed_frame_id,
        )
        await self.emit("telemetry", telemetry_payload, ts_ms=now_ms)
        self._telemetry.mark_emitted(now_ms)

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
            "GPU inference failed; switched ortProvider to cpu automatically.\n"
            f"reason: {detail}"
        )

    def _close_shm(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                self._shm = None
        self._shm = None
        self._shm_open_name = ""

    def _open_shm(self, shm_name: str) -> None:
        self._close_shm()
        shm = VideoShmReader(shm_name)
        shm.open(use_event=True)
        self._shm = shm
        self._shm_open_name = shm_name

    async def _maybe_reopen_shm(self) -> None:
        want = self._resolve_input_shm_name()
        if want == self._shm_open_name:
            return
        self._close_shm()

    def _resolve_input_shm_name(self) -> str:
        shm_name = str(self._input_shm_name or "").strip()
        if shm_name:
            return shm_name
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
        await self.set_state("flowShmName", self._flow_shm_name)
        await self.set_state("flowShmFormat", self._flow_shm_format)

        warn_parts: list[str] = []
        if runtime.provider_warning:
            warn_parts.append(str(runtime.provider_warning))
        prefer = str(self._ort_provider or "auto").lower()
        if prefer in ("auto", "cuda"):
            try:
                import onnxruntime as ort  # type: ignore

                available = list(ort.get_available_providers())  # type: ignore[attr-defined]
            except Exception as exc:
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
        await self.set_state("lastError", "")
        await self.set_state("ortActiveProviders", "")
        await self.set_state("flowShmName", self._flow_shm_name)
        await self.set_state("flowShmFormat", self._flow_shm_format)
        if self._model_index_warning:
            await self._set_last_error(self._model_index_warning)

    async def _ensure_onnx_available(self, spec: ModelSpec) -> None:
        if spec.onnx_path.exists():
            self._download_retry_at_monotonic = 0.0
            return
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
        except Exception as exc:
            self._download_retry_at_monotonic = time.monotonic() + 30.0
            raise RuntimeError(
                f"Auto-download failed for model={spec.model_id!r} path={spec.onnx_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _close_flow_writer(self) -> None:
        if self._flow_writer is not None:
            try:
                self._flow_writer.close()
            except Exception:
                self._flow_writer = None
        self._flow_writer = None
        self._flow_writer_width = 0
        self._flow_writer_height = 0
        self._flow_writer_pitch = 0

    def _ensure_flow_writer(self, *, width: int, height: int, pitch: int) -> None:
        if self._flow_writer is not None:
            if (
                int(self._flow_writer_width) == int(width)
                and int(self._flow_writer_height) == int(height)
                and int(self._flow_writer_pitch) == int(pitch)
            ):
                return
            self._close_flow_writer()
        frame_bytes = int(pitch) * int(height)
        shm_size = max(1024 * 1024, frame_bytes * 2 + 4096)
        writer = VideoShmWriter(shm_name=self._flow_shm_name, size=shm_size, slot_count=2)
        writer.open()
        self._flow_writer = writer
        self._flow_writer_width = int(width)
        self._flow_writer_height = int(height)
        self._flow_writer_pitch = int(pitch)

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
                except Exception as exc:
                    await self._record_exception(where="ensure_runtime", exc=exc)
                    await asyncio.sleep(0.1)
                    continue

                input_shm_name = self._resolve_input_shm_name()
                if not input_shm_name:
                    await self._set_last_error("missing inputShmName")
                    await self._emit_idle_telemetry(now_ms=int(time.time() * 1000), shm_name="")
                    await asyncio.sleep(0.05)
                    continue

                if self._shm is None:
                    try:
                        self._open_shm(input_shm_name)
                    except Exception as exc:
                        await self._record_exception(where="open_shm", exc=exc)
                        await asyncio.sleep(0.1)
                        continue

                assert self._runtime is not None
                assert self._shm is not None
                t0 = time.perf_counter()
                self._shm.wait_new_frame(timeout_ms=10)
                header, payload = self._shm.read_latest_frame()
                if header is None or payload is None:
                    await self._emit_idle_telemetry(now_ms=int(time.time() * 1000), shm_name=input_shm_name)
                    continue
                if int(header.fmt) != VIDEO_FORMAT_BGRA32:
                    await self._set_last_error(
                        f"input SHM format must be BGRA32(fmt={VIDEO_FORMAT_BGRA32}), got fmt={int(header.fmt)} "
                        f"for {input_shm_name!r}"
                    )
                    await self._emit_idle_telemetry(now_ms=int(header.ts_ms or time.time() * 1000), shm_name=input_shm_name)
                    await asyncio.sleep(0.05)
                    continue

                frame_id_seen = int(header.frame_id)
                if self._last_processed_frame_id is not None and frame_id_seen == int(self._last_processed_frame_id):
                    self._dup_skipped_since_last_processed += 1
                    await self._emit_idle_telemetry(now_ms=int(header.ts_ms or time.time() * 1000), shm_name=input_shm_name)
                    continue
                dup_skipped = int(self._dup_skipped_since_last_processed)
                self._dup_skipped_since_last_processed = 0

                width = int(header.width)
                height = int(header.height)
                pitch = int(header.pitch)
                if width <= 0 or height <= 0 or pitch <= 0:
                    continue
                frame_bytes = int(pitch) * int(height)
                if len(payload) < frame_bytes:
                    continue
                self._last_processed_frame_id = frame_id_seen
                self._new_frame_counter += 1

                buf = np.frombuffer(payload, dtype=np.uint8)
                rows = buf.reshape((height, pitch))
                bgra = rows[:, : width * 4].reshape((height, width, 4))
                frame_bgr = np.ascontiguousarray(bgra[:, :, 0:3])

                tensor = self._runtime.prepare_input(frame_bgr)
                pair = self._frame_cache.push_and_get_pair(
                    PreparedFlowFrame(
                        frame_id=frame_id_seen,
                        width=width,
                        height=height,
                        tensor=tensor,
                    )
                )
                if pair is None:
                    await self._set_last_error("waiting for frame pair (need at least 2 valid BGRA frames)")
                    await self._emit_idle_telemetry(now_ms=int(header.ts_ms or time.time() * 1000), shm_name=input_shm_name)
                    continue

                if (int(self._new_frame_counter) % int(self._compute_every_n_frames)) != 0:
                    await self._emit_idle_telemetry(now_ms=int(header.ts_ms or time.time() * 1000), shm_name=input_shm_name)
                    continue

                t_infer0 = time.perf_counter()
                prev_frame, current_frame = pair
                try:
                    flow = self._runtime.infer_preprocessed(
                        prev_frame.tensor,
                        current_frame.tensor,
                        output_size_hw=(height, width),
                    )
                except Exception as exc:
                    if self._should_fallback_to_cpu(exc):
                        await self._fallback_to_cpu_after_gpu_error(exc=exc)
                        await asyncio.sleep(0.1)
                        continue
                    raise
                flow_pitch, flow_payload = pack_flow2_f16_payload(flow)
                self._ensure_flow_writer(width=width, height=height, pitch=flow_pitch)
                assert self._flow_writer is not None
                self._flow_writer.write_frame(
                    width=width,
                    height=height,
                    pitch=flow_pitch,
                    payload=flow_payload,
                    fmt=VIDEO_FORMAT_FLOW2_F16,
                )
                t_infer1 = time.perf_counter()
                if self._runtime_warning:
                    await self._set_last_error(self._runtime_warning)
                elif self._last_error:
                    await self._set_last_error("")

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
                        service_class=self._service_class,
                        model=self._model,
                        ort_provider=self._ort_provider,
                        shm_name=str(self._shm_open_name or input_shm_name),
                        frame_id_last_seen=frame_id_seen,
                        frame_id_last_processed=self._last_processed_frame_id,
                    )
                    await self.emit("telemetry", telemetry_payload, ts_ms=now_ms)
                    self._telemetry.mark_emitted(now_ms)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_exception(where="loop", exc=exc)
                await asyncio.sleep(0.1)
