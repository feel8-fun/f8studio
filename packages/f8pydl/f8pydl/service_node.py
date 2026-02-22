from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import ServiceNode
from f8pysdk.shm.video import VideoShmReader

from .constants import CLASSIFICATION_SCHEMA_VERSION, DETECTION_SCHEMA_VERSION
from .model_config import ModelSpec, ModelTask, build_model_index, build_model_index_with_errors, load_model_spec
from .onnx_runtime import OnnxClassifierRuntime, OnnxYoloDetectorRuntime
from .vision_utils import clamp_xyxy


def _default_weights_dir() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append((Path.cwd() / "services" / "f8" / "dl" / "weights").resolve())
    except Exception:
        pass
    try:
        candidates.append((Path.cwd() / "services" / "f8" / "detect_tracker" / "weights").resolve())
    except Exception:
        pass
    try:
        root = Path(__file__).resolve().parents[3]
        candidates.append((root / "services" / "f8" / "dl" / "weights").resolve())
        candidates.append((root / "services" / "f8" / "detect_tracker" / "weights").resolve())
    except Exception:
        pass
    for p in candidates:
        try:
            if p.exists() and p.is_dir():
                return p
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


def _coerce_str_list(v: Any) -> list[str]:
    if isinstance(v, (list, tuple)):
        out: list[str] = []
        for item in v:
            s = _coerce_str(item)
            if s:
                out.append(s)
        return out
    if isinstance(v, str):
        raw = v.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            out2: list[str] = []
            for item in parsed:
                s = _coerce_str(item)
                if s:
                    out2.append(s)
            return out2
    return []


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
            data_out_ports=[str(output_port), "telemetry"],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._service_class = str(service_class)
        self._service_task = service_task
        self._output_port = output_port
        self._allowed_tasks = set(allowed_tasks)

        self._active = True
        self._config_loaded = False
        self._task: asyncio.Task[object] | None = None

        self._weights_dir = _default_weights_dir()
        self._model_yaml_path = ""
        self._model_id = ""
        self._ort_provider: Literal["auto", "cuda", "cpu"] = "auto"
        self._infer_every_n = 1
        self._conf_override = -1.0
        self._iou_override = -1.0
        self._top_k = 5
        self._shm_name = ""
        self._enabled_classes: list[str] = []
        self._per_class_k = 0

        self._shm: VideoShmReader | None = None
        self._shm_open_name = ""

        self._model: ModelSpec | None = None
        self._det_runtime: OnnxYoloDetectorRuntime | None = None
        self._cls_runtime: OnnxClassifierRuntime | None = None
        self._runtime_yaml: Path | None = None
        self._last_error = ""
        self._last_error_signature = ""
        self._last_error_repeats = 0
        self._model_index_warning = ""

        self._last_infer_frame_id: int | None = None
        self._last_processed_frame_id: int | None = None
        self._dup_skipped_since_last_processed = 0
        self._telemetry = _Telemetry()

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        loop = asyncio.get_running_loop()
        loop.create_task(self._ensure_config_loaded(), name=f"f8dl:init:{self.node_id}")
        self._task = loop.create_task(self._loop(), name=f"f8dl:loop:{self.node_id}")

    async def close(self) -> None:
        t = self._task
        self._task = None
        if t is not None:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
        self._close_shm()

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

        if name == "inferEveryN":
            self._infer_every_n = _coerce_int(
                await self.get_state_value("inferEveryN"),
                default=self._infer_every_n,
                minimum=1,
                maximum=10000,
            )
            return

        if name == "confThreshold":
            self._conf_override = _coerce_float(await self.get_state_value("confThreshold"), default=self._conf_override)
            await self._reset_runtime()
            return

        if name == "iouThreshold":
            self._iou_override = _coerce_float(await self.get_state_value("iouThreshold"), default=self._iou_override)
            await self._reset_runtime()
            return

        if name == "topK":
            self._top_k = _coerce_int(await self.get_state_value("topK"), default=self._top_k, minimum=1, maximum=100)
            return

        if name == "shmName":
            self._shm_name = _coerce_str(await self.get_state_value("shmName"), default=self._shm_name)
            await self._maybe_reopen_shm()
            return

        if name == "enabledClasses":
            self._enabled_classes = _coerce_str_list(await self.get_state_value("enabledClasses"))
            self._enabled_classes = self._normalize_enabled_classes(self._enabled_classes)
            return

        if name == "perClassK":
            self._per_class_k = _coerce_int(
                await self.get_state_value("perClassK"),
                default=self._per_class_k,
                minimum=0,
                maximum=10000,
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
        self._infer_every_n = _coerce_int(
            await self.get_state_value("inferEveryN"),
            default=int(self._initial_state.get("inferEveryN") or 1),
            minimum=1,
            maximum=10000,
        )
        self._conf_override = _coerce_float(
            await self.get_state_value("confThreshold"),
            default=float(self._initial_state.get("confThreshold") or -1.0),
        )
        self._iou_override = _coerce_float(
            await self.get_state_value("iouThreshold"),
            default=float(self._initial_state.get("iouThreshold") or -1.0),
        )
        self._top_k = _coerce_int(
            await self.get_state_value("topK"),
            default=int(self._initial_state.get("topK") or 5),
            minimum=1,
            maximum=100,
        )
        self._enabled_classes = _coerce_str_list(
            await self.get_state_value("enabledClasses"),
        )
        self._enabled_classes = self._normalize_enabled_classes(self._enabled_classes)
        self._per_class_k = _coerce_int(
            await self.get_state_value("perClassK"),
            default=int(self._initial_state.get("perClassK") or 0),
            minimum=0,
            maximum=10000,
        )
        self._shm_name = _coerce_str(await self.get_state_value("shmName"), default=str(self._initial_state.get("shmName") or ""))
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
        await self._publish_selected_model_metadata()

    async def _publish_selected_model_metadata(self) -> None:
        try:
            yaml_path = self._resolve_model_yaml()
            spec = load_model_spec(yaml_path)
        except Exception:
            await self.set_state("modelClasses", [])
            await self.set_state("enabledClasses", [])
            return

        await self.set_state("modelClasses", [str(x) for x in (spec.classes or [])])
        self._enabled_classes = self._normalize_enabled_classes(
            self._enabled_classes,
            allowed_classes=list(spec.classes or []),
        )
        await self.set_state("enabledClasses", list(self._enabled_classes))

    async def _set_last_error(self, message: str) -> None:
        self._last_error = str(message or "")
        await self.set_state("lastError", self._last_error)

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
        self._cls_runtime = None
        self._runtime_yaml = None
        self._model = None
        self._last_error_signature = ""
        self._last_error_repeats = 0
        await self.set_state("loadedModel", "")
        await self.set_state("lastError", "")
        await self.set_state("ortActiveProviders", "")
        await self._publish_selected_model_metadata()
        if self._model_index_warning:
            await self._set_last_error(self._model_index_warning)

    async def _maybe_reopen_shm(self) -> None:
        want = self._resolve_shm_name()
        if want == self._shm_open_name:
            return
        self._close_shm()

    def _resolve_shm_name(self) -> str:
        shm = str(self._shm_name or "").strip()
        if shm:
            return shm
        return ""

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
            name = _coerce_str(raw)
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

    def _close_shm(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                self._shm = None
        self._shm = None
        self._shm_open_name = ""

    def _open_shm(self, shm_name: str) -> bool:
        self._close_shm()
        shm = VideoShmReader(shm_name)
        shm.open(use_event=True)
        self._shm = shm
        self._shm_open_name = shm_name
        return True

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
        if self._det_runtime is not None or self._cls_runtime is not None:
            return True

        yaml_path = self._resolve_model_yaml()
        spec = load_model_spec(yaml_path)
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
            runtime = OnnxClassifierRuntime(spec, ort_provider=self._ort_provider)
            self._cls_runtime = runtime
            providers = runtime.active_providers
            warn = runtime.provider_warning
        else:
            runtime = OnnxYoloDetectorRuntime(spec, ort_provider=self._ort_provider)
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
        await self._set_last_error("\n".join([x for x in warn_parts if str(x).strip()]).strip())
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
                except Exception as exc:
                    await self._record_exception(where="ensure_runtime", exc=exc)
                    await asyncio.sleep(0.1)
                    continue

                shm_name = self._resolve_shm_name()
                if not shm_name:
                    await asyncio.sleep(0.05)
                    continue

                if self._shm is None:
                    try:
                        self._open_shm(shm_name)
                    except Exception as exc:
                        await self._record_exception(where="open_shm", exc=exc)
                        await asyncio.sleep(0.1)
                        continue

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

                do_infer = False
                if self._last_infer_frame_id is None:
                    do_infer = True
                else:
                    do_infer = (int(header.frame_id) - int(self._last_infer_frame_id)) >= int(self._infer_every_n)
                if not do_infer:
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
                frame_bgr = np.ascontiguousarray(bgra[:, :, 0:3])

                t_infer0 = time.perf_counter()
                if self._det_runtime is not None:
                    detections, _meta = self._det_runtime.infer(frame_bgr)
                    payload_out = self._build_detection_payload(
                        width=width,
                        height=height,
                        frame_id=frame_id_seen,
                        ts_ms=int(header.ts_ms),
                        detections=detections,
                    )
                    await self.emit("detections", payload_out, ts_ms=int(header.ts_ms))
                elif self._cls_runtime is not None:
                    topk, _meta = self._cls_runtime.infer(frame_bgr, top_k=self._top_k)
                    payload_out = self._build_classification_payload(
                        frame_id=frame_id_seen,
                        ts_ms=int(header.ts_ms),
                        topk=topk,
                    )
                    await self.emit("classifications", payload_out, ts_ms=int(header.ts_ms))
                else:
                    raise RuntimeError("Inference runtime is not initialized.")
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
                    tel = self._telemetry.summary(
                        now_ms=now_ms,
                        node_id=self.node_id,
                        service_class=self._service_class,
                        model=self._model,
                        ort_provider=self._ort_provider,
                        shm_name=str(self._shm_open_name or shm_name),
                        frame_id_last_seen=frame_id_seen,
                        frame_id_last_processed=self._last_processed_frame_id,
                    )
                    await self.emit("telemetry", tel, ts_ms=now_ms)
                    self._telemetry.mark_emitted(now_ms)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_exception(where="loop", exc=exc)
                await asyncio.sleep(0.1)

    def _build_detection_payload(self, *, width: int, height: int, frame_id: int, ts_ms: int, detections: list[Any]) -> dict[str, Any]:
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
