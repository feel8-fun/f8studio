from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import threading
import time
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import msgspec

from .codec import dump_json, encode_obj, validate_as
from .generated import (
    F8ComplexObjectTypeSchema,
    F8DataPortSpec,
    F8DataTypeSchema,
    F8MonitorCpu,
    F8MonitorReport,
    F8MonitorError,
    F8MonitorFrame,
    F8MonitorGpu,
    F8MonitorMemory,
    F8MonitorQueue,
    F8MonitorSnapshot,
    F8MonitorTiming,
)
from .f8_naming import data_key
from ._specs.schema import boolean_schema, complex_object_schema, integer_schema, number_schema, string_schema
from .time_utils import now_ms

if TYPE_CHECKING:
    from .service_bus.runtime import ServiceBus


MONITOR_PORT_NAME = "monitor"
_FORBIDDEN_TELEMETRY_PORT_NAME = "telemetry"
MONITOR_SNAPSHOT_SCHEMA_VERSION = "f8monitor/1"
MONITOR_REPORT_SCHEMA_VERSION = "f8monitorReport/1"
_MONITOR_SNAPSHOT_SCHEMA_DICT: dict[str, object] | None = None
MonitorErrorSeverity = Literal["info", "warning", "error", "critical"]
_MONITOR_ERROR_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "error", "critical"})
_ERROR_REPEAT_PUBLISH_INTERVAL_MS = 1000

logger = logging.getLogger(__name__)

_MONITOR_SCHEMA_PARSE_ERRORS = (TypeError, ValueError, msgspec.MsgspecError)
_MONITOR_GPU_SHUTDOWN_ERRORS = (OSError, RuntimeError)
_MONITOR_SNAPSHOT_BUILD_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError, msgspec.MsgspecError)
_MONITOR_SNAPSHOT_PUBLISH_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError, msgspec.MsgspecError)


class MonitorContractError(ValueError):
    """Raised when monitor payloads/describe contracts violate the unified schema contract."""


def _normalize_monitor_error_severity(severity: str) -> MonitorErrorSeverity:
    text = str(severity or "").strip().lower()
    if text == "info":
        return "info"
    if text == "warning":
        return "warning"
    if text == "critical":
        return "critical"
    return "error"


def _normalize_monitor_error_ts(ts_ms: int | None) -> int:
    if ts_ms is None:
        return int(now_ms())
    ts = int(ts_ms)
    if ts <= 0:
        return int(now_ms())
    return ts


def _derive_monitor_error_fingerprint(
    *,
    node_id: str,
    code: str,
    message: str,
    fingerprint: str | None,
) -> str:
    explicit = str(fingerprint or "").strip()
    if explicit:
        return explicit
    raw = f"{node_id}\0{code}\0{message}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:24]


def monitor_snapshot_value_schema() -> F8DataTypeSchema:
    cpu = complex_object_schema(
        properties={
            "processPercent": number_schema(default=0.0, minimum=0.0),
            "systemPercent": number_schema(default=0.0, minimum=0.0),
        }
    )
    memory = complex_object_schema(
        properties={
            "rssBytes": integer_schema(default=0, minimum=0),
            "vmsBytes": integer_schema(default=0, minimum=0),
        }
    )
    gpu = complex_object_schema(
        properties={
            "vendor": string_schema(default=""),
            "deviceIndex": integer_schema(default=-1),
            "utilPercent": number_schema(default=0.0, minimum=0.0),
            "memoryUsedBytes": integer_schema(default=0, minimum=0),
            "memoryTotalBytes": integer_schema(default=0, minimum=0),
            "available": boolean_schema(default=False),
        }
    )
    frame = complex_object_schema(
        properties={
            "observed": integer_schema(default=0, minimum=0),
            "processed": integer_schema(default=0, minimum=0),
            "dropped": integer_schema(default=0, minimum=0),
            "localOnlyEmits": integer_schema(default=0, minimum=0),
            "routedCrossEmits": integer_schema(default=0, minimum=0),
            "suppressedCrossPublishes": integer_schema(default=0, minimum=0),
            "callbackDeliveries": integer_schema(default=0, minimum=0),
            "bufferPullDeliveries": integer_schema(default=0, minimum=0),
        }
    )
    timing = complex_object_schema(
        properties={
            "processMsAvg": number_schema(default=0.0, minimum=0.0),
            "processMsP95": number_schema(default=0.0, minimum=0.0),
            "waitMsAvg": number_schema(default=0.0, minimum=0.0),
            "waitMsP95": number_schema(default=0.0, minimum=0.0),
            "latencyMsAvg": number_schema(default=0.0, minimum=0.0),
            "latencyMsP95": number_schema(default=0.0, minimum=0.0),
        }
    )
    queue = complex_object_schema(
        properties={
            "depth": integer_schema(default=0, minimum=0),
        }
    )
    error = complex_object_schema(
        properties={
            "countWindow": integer_schema(default=0, minimum=0),
            "lastNodeId": string_schema(default=""),
            "lastCode": string_schema(default=""),
            "lastMessage": string_schema(default=""),
            "lastSeverity": string_schema(default="error", enum=sorted(_MONITOR_ERROR_SEVERITIES)),
            "lastFingerprint": string_schema(default=""),
            "lastRepeatCount": integer_schema(default=0, minimum=0),
            "lastTsMs": integer_schema(default=0, minimum=0),
            "currentNodeId": string_schema(default=""),
            "currentCode": string_schema(default=""),
            "currentMessage": string_schema(default=""),
            "currentSeverity": string_schema(default="", enum=["", "info", "warning", "error", "critical"]),
            "currentTsMs": integer_schema(default=0, minimum=0),
        }
    )
    root = complex_object_schema(
        properties={
            "schemaVersion": string_schema(default=MONITOR_SNAPSHOT_SCHEMA_VERSION, enum=[MONITOR_SNAPSHOT_SCHEMA_VERSION]),
            "serviceId": string_schema(default=""),
            "serviceClass": string_schema(default=""),
            "nodeId": string_schema(default=""),
            "tsMs": integer_schema(default=0, minimum=0),
            "alive": boolean_schema(default=True),
            "ready": boolean_schema(default=False),
            "active": boolean_schema(default=True),
            "uptimeMs": integer_schema(default=0, minimum=0),
            "cpu": cpu,
            "memory": memory,
            "gpu": gpu,
            "frame": frame,
            "timing": timing,
            "queue": queue,
            "error": error,
        }
    )
    if isinstance(root, F8ComplexObjectTypeSchema):
        return root
    return root


def monitor_snapshot_data_port() -> F8DataPortSpec:
    return F8DataPortSpec(
        name=MONITOR_PORT_NAME,
        description="Unified runtime monitor snapshots (health/resource/perf/error).",
        valueSchema=monitor_snapshot_value_schema(),
        definitionProtected=True,
        showOnNode=False,
    )


def monitor_snapshot_schema_dict() -> dict[str, object]:
    global _MONITOR_SNAPSHOT_SCHEMA_DICT
    if _MONITOR_SNAPSHOT_SCHEMA_DICT is None:
        raw = dump_json(monitor_snapshot_value_schema(), mode="json", by_alias=True)
        _MONITOR_SNAPSHOT_SCHEMA_DICT = raw if isinstance(raw, dict) else {}
    return deepcopy(_MONITOR_SNAPSHOT_SCHEMA_DICT)


def monitor_snapshot_schema_dict_cached() -> dict[str, object]:
    global _MONITOR_SNAPSHOT_SCHEMA_DICT
    if _MONITOR_SNAPSHOT_SCHEMA_DICT is None:
        raw = dump_json(monitor_snapshot_value_schema(), mode="json", by_alias=True)
        _MONITOR_SNAPSHOT_SCHEMA_DICT = raw if isinstance(raw, dict) else {}
    return _MONITOR_SNAPSHOT_SCHEMA_DICT


def validate_monitor_snapshot_payload(payload: dict[str, Any] | F8MonitorSnapshot) -> F8MonitorSnapshot:
    if isinstance(payload, F8MonitorSnapshot):
        return validate_as(F8MonitorSnapshot, dump_json(payload, mode="json", by_alias=True))
    if not isinstance(payload, dict):
        raise MonitorContractError("monitor snapshot payload must be dict or F8MonitorSnapshot")
    return validate_as(F8MonitorSnapshot, payload)


def validate_monitor_report_payload(payload: dict[str, Any] | F8MonitorReport) -> F8MonitorReport:
    if isinstance(payload, F8MonitorReport):
        return validate_as(F8MonitorReport, dump_json(payload, mode="json", by_alias=True))
    if not isinstance(payload, dict):
        raise MonitorContractError("monitor report payload must be dict or F8MonitorReport")
    return validate_as(F8MonitorReport, payload)


def validate_describe_monitor_contract(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise MonitorContractError("describe payload must be a dict")
    target = payload
    service_obj = payload.get("service")
    if isinstance(service_obj, dict):
        target = service_obj
    ports_obj = target.get("dataOutPorts")
    if not isinstance(ports_obj, list):
        raise MonitorContractError("service.dataOutPorts must be a list")

    monitor_port: dict[str, Any] | None = None
    telemetry_ports: list[dict[str, Any]] = []
    for item in ports_obj:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name == MONITOR_PORT_NAME:
            monitor_port = item
            continue
        if name == _FORBIDDEN_TELEMETRY_PORT_NAME:
            telemetry_ports.append(item)

    if monitor_port is None:
        raise MonitorContractError("service.dataOutPorts must contain `monitor`")
    protected_raw = monitor_port.get("definitionProtected")
    if protected_raw is not None and not bool(protected_raw):
        raise MonitorContractError("`monitor` dataOutPort must set definitionProtected=true")
    monitor_schema_obj = monitor_port.get("valueSchema")
    if not isinstance(monitor_schema_obj, dict):
        raise MonitorContractError("`monitor` dataOutPort must contain object valueSchema")
    expected_schema = monitor_snapshot_schema_dict_cached()
    if monitor_schema_obj != expected_schema:
        try:
            parsed_schema = dump_json(
                validate_as(F8DataTypeSchema, monitor_schema_obj),
                mode="json",
                by_alias=True,
            )
        except _MONITOR_SCHEMA_PARSE_ERRORS as exc:
            raise MonitorContractError(f"`monitor` valueSchema is invalid: {type(exc).__name__}: {exc}") from exc
        if parsed_schema != expected_schema:
            raise MonitorContractError("`monitor` valueSchema must match F8MonitorSnapshot schema")
    if telemetry_ports:
        raise MonitorContractError("legacy `telemetry` output port is forbidden; use `monitor` only")


def _percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    idx = int(round((len(ordered) - 1) * 0.95))
    if idx < 0:
        idx = 0
    if idx >= len(ordered):
        idx = len(ordered) - 1
    return float(ordered[idx])


class _TimedValues:
    def __init__(self, *, window_ms: int) -> None:
        self._window_ms = max(1000, int(window_ms))
        self._values: deque[tuple[int, float]] = deque()

    def set_window_ms(self, window_ms: int) -> None:
        self._window_ms = max(1000, int(window_ms))

    def add(self, *, ts_ms: int, value: float) -> None:
        self._values.append((int(ts_ms), float(value)))
        self._prune(int(ts_ms))

    def _prune(self, now_ms_value: int) -> None:
        cutoff = int(now_ms_value) - int(self._window_ms)
        while self._values and int(self._values[0][0]) < cutoff:
            self._values.popleft()

    def values(self, *, now_ms_value: int) -> list[float]:
        self._prune(int(now_ms_value))
        return [float(item[1]) for item in self._values]


class _ProcessSampler:
    def __init__(self) -> None:
        self._last_wall = time.perf_counter()
        self._last_proc_cpu = time.process_time()

    def sample_process_percent(self) -> float:
        wall = time.perf_counter()
        proc_cpu = time.process_time()
        d_wall = float(wall - self._last_wall)
        d_cpu = float(proc_cpu - self._last_proc_cpu)
        self._last_wall = wall
        self._last_proc_cpu = proc_cpu
        if d_wall <= 0.0:
            return 0.0
        cpu_count = float(max(1, int(os.cpu_count() or 1)))
        percent = (d_cpu / d_wall) * 100.0 / cpu_count
        if percent < 0.0:
            return 0.0
        return float(percent)

    def sample_system_percent(self) -> float:
        try:
            load1, _, _ = os.getloadavg()
        except (AttributeError, OSError):
            return 0.0
        cpu_count = float(max(1, int(os.cpu_count() or 1)))
        percent = (float(load1) / cpu_count) * 100.0
        if percent < 0.0:
            return 0.0
        return float(percent)


class _MemorySampler:
    def __init__(self) -> None:
        self._psutil_process = None
        try:
            import psutil  # type: ignore[import-not-found]

            self._psutil_process = psutil.Process(os.getpid())
        except (ImportError, OSError, RuntimeError):
            self._psutil_process = None

    def sample(self) -> tuple[int, int]:
        if self._psutil_process is not None:
            try:
                info = self._psutil_process.memory_info()
                rss = int(info.rss)
                vms = int(info.vms)
                if rss < 0:
                    rss = 0
                if vms < 0:
                    vms = 0
                return rss, vms
            except (AttributeError, OSError, RuntimeError, ValueError):
                return 0, 0
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_kb = int(usage.ru_maxrss)
            if rss_kb < 0:
                rss_kb = 0
            return rss_kb * 1024, 0
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError):
            return 0, 0


class _GpuSampler:
    def __init__(self, *, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._nvml = None
        self._initialized = False
        if not self._enabled:
            return
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._initialized = True
        except (ImportError, OSError, RuntimeError):
            self._nvml = None
            self._initialized = False

    def close(self) -> None:
        if self._initialized and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except _MONITOR_GPU_SHUTDOWN_ERRORS as exc:
                logger.debug("monitor gpu sampler shutdown failed", exc_info=exc)
        self._initialized = False
        self._nvml = None

    def sample(self) -> F8MonitorGpu:
        if (not self._enabled) or (not self._initialized) or self._nvml is None:
            return F8MonitorGpu(
                vendor="",
                deviceIndex=None,
                utilPercent=None,
                memoryUsedBytes=None,
                memoryTotalBytes=None,
                available=False,
            )
        try:
            handle = self._nvml.nvmlDeviceGetHandleByIndex(0)
            util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            return F8MonitorGpu(
                vendor="nvidia",
                deviceIndex=0,
                utilPercent=float(util.gpu),
                memoryUsedBytes=int(mem.used),
                memoryTotalBytes=int(mem.total),
                available=True,
            )
        except (OSError, RuntimeError, ValueError):
            return F8MonitorGpu(
                vendor="nvidia",
                deviceIndex=0,
                utilPercent=None,
                memoryUsedBytes=None,
                memoryTotalBytes=None,
                available=False,
            )


@dataclass(frozen=True)
class MonitorCollectorConfig:
    enabled: bool = True
    interval_ms: int = 1000
    window_ms: int = 30000
    gpu_enabled: bool = True


class MonitorCollector:
    def __init__(self, bus: "ServiceBus", config: MonitorCollectorConfig) -> None:
        self._bus = bus
        self._enabled = bool(config.enabled)
        self._interval_ms = max(200, int(config.interval_ms))
        self._window_ms = max(1000, int(config.window_ms))
        self._process_sampler = _ProcessSampler()
        self._memory_sampler = _MemorySampler()
        self._gpu_sampler = _GpuSampler(enabled=bool(config.gpu_enabled))
        self._lock = threading.Lock()
        self._ready = False
        self._observed = 0
        self._processed = 0
        self._dropped = 0
        self._local_only_emits = 0
        self._routed_cross_emits = 0
        self._suppressed_cross_publishes = 0
        self._callback_deliveries = 0
        self._buffer_pull_deliveries = 0
        self._last_error_node_id = ""
        self._last_error_code = ""
        self._last_error_message = ""
        self._last_error_severity: MonitorErrorSeverity = "error"
        self._last_error_fingerprint = ""
        self._last_error_repeat_count = 0
        self._last_error_ts_ms: int | None = None
        self._current_error_node_id = ""
        self._current_error_code = ""
        self._current_error_message = ""
        self._current_error_severity: MonitorErrorSeverity | Literal[""] = ""
        self._current_error_fingerprint = ""
        self._current_error_ts_ms: int | None = None
        self._error_events: deque[int] = deque()
        self._wait_values = _TimedValues(window_ms=self._window_ms)
        self._process_values = _TimedValues(window_ms=self._window_ms)
        self._latency_values = _TimedValues(window_ms=self._window_ms)
        self._node_last_input_ts_ms: dict[str, int] = {}
        self._started_ts_ms = int(now_ms())
        self._task: asyncio.Task[object] | None = None
        self._publish_once_task: asyncio.Task[object] | None = None
        self._pending_error_publish_task: asyncio.Task[object] | None = None
        self._last_error_publish_fingerprint = ""
        self._last_error_publish_ts_ms = 0
        self._latest: F8MonitorSnapshot | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    def latest_snapshot(self) -> F8MonitorSnapshot | None:
        with self._lock:
            return self._latest

    def record_ready(self, ready: bool) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._ready = bool(ready)

    def record_observed(self, *, port: str) -> None:
        if (not self._enabled) or str(port) == "monitor":
            return
        with self._lock:
            self._observed += 1

    def record_processed(self, *, port: str, emit_ts_ms: int, now_ts_ms: int) -> None:
        del emit_ts_ms
        del now_ts_ms
        if (not self._enabled) or str(port) == "monitor":
            return
        with self._lock:
            self._processed += 1

    def record_timing(
        self,
        *,
        port: str,
        process_ms: float,
        latency_ms: float,
        ts_ms: int | None = None,
    ) -> None:
        if (not self._enabled) or str(port) == "monitor":
            return
        sample_ts = int(ts_ms) if ts_ms is not None else int(now_ms())
        process_value = float(process_ms)
        latency_value = float(latency_ms)
        process_ok = math.isfinite(process_value) and process_value >= 0.0
        latency_ok = math.isfinite(latency_value) and latency_value >= 0.0
        if (not process_ok) and (not latency_ok):
            return
        with self._lock:
            if process_ok:
                self._process_values.add(ts_ms=sample_ts, value=process_value)
            if latency_ok:
                self._latency_values.add(ts_ms=sample_ts, value=latency_value)

    def record_input_sample_ts(self, *, node_id: str, sample_ts_ms: int) -> None:
        if not self._enabled:
            return
        sid = str(node_id or "").strip()
        if not sid:
            return
        ts = int(sample_ts_ms)
        if ts <= 0:
            return
        with self._lock:
            existing = self._node_last_input_ts_ms.get(sid)
            if existing is None or int(ts) < int(existing):
                self._node_last_input_ts_ms[sid] = int(ts)

    def record_emit_completed(self, *, node_id: str, now_ts_ms: int) -> None:
        if not self._enabled:
            return
        sid = str(node_id or "").strip()
        if not sid:
            return
        now_ts = int(now_ts_ms)
        with self._lock:
            input_ts = self._node_last_input_ts_ms.pop(sid, None)
            if input_ts is None:
                return
            if now_ts < int(input_ts):
                return
            self._latency_values.add(ts_ms=now_ts, value=float(now_ts - int(input_ts)))

    def record_wait_ms(self, *, wait_ms: float) -> None:
        if not self._enabled or wait_ms < 0.0:
            return
        with self._lock:
            self._wait_values.add(ts_ms=int(now_ms()), value=float(wait_ms))

    def record_dropped(self, *, dropped_count: int) -> None:
        if not self._enabled or dropped_count <= 0:
            return
        with self._lock:
            self._dropped += int(dropped_count)

    def record_local_only_emit(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._local_only_emits += 1

    def record_routed_cross_emit(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._routed_cross_emits += 1

    def record_suppressed_cross_publish(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._suppressed_cross_publishes += 1

    def record_callback_delivery(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._callback_deliveries += 1

    def record_buffer_pull_delivery(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._buffer_pull_deliveries += 1

    def report_error(
        self,
        *,
        node_id: str,
        code: str,
        message: str,
        severity: str = "error",
        fingerprint: str | None = None,
        ts_ms: int | None = None,
    ) -> None:
        if not self._enabled:
            return
        node_id_s = str(node_id or "").strip() or str(self._bus.service_id)
        code_s = str(code or "").strip() or "ERROR"
        message_s = str(message or "")
        severity_s = _normalize_monitor_error_severity(severity)
        now_ts = _normalize_monitor_error_ts(ts_ms)
        fingerprint_s = _derive_monitor_error_fingerprint(
            node_id=node_id_s,
            code=code_s,
            message=message_s,
            fingerprint=fingerprint,
        )
        with self._lock:
            previous_fingerprint = str(self._last_error_fingerprint)
            if fingerprint_s == previous_fingerprint:
                self._last_error_repeat_count = max(1, int(self._last_error_repeat_count)) + 1
            else:
                self._last_error_repeat_count = 1
            self._last_error_node_id = node_id_s
            self._last_error_code = code_s
            self._last_error_message = message_s
            self._last_error_severity = severity_s
            self._last_error_fingerprint = fingerprint_s
            self._last_error_ts_ms = now_ts
            self._current_error_node_id = node_id_s
            self._current_error_code = code_s
            self._current_error_message = message_s
            self._current_error_severity = severity_s
            self._current_error_fingerprint = fingerprint_s
            self._current_error_ts_ms = now_ts
            self._error_events.append(now_ts)
        self._request_error_publish_once(
            fingerprint=fingerprint_s,
            immediate=fingerprint_s != previous_fingerprint,
        )

    def clear_error(self, *, node_id: str, fingerprint: str | None = None, ts_ms: int | None = None) -> None:
        if not self._enabled:
            return
        node_id_s = str(node_id or "").strip() or str(self._bus.service_id)
        fingerprint_s = str(fingerprint or "").strip()
        with self._lock:
            if self._current_error_node_id and self._current_error_node_id != node_id_s:
                return
            if fingerprint_s and self._current_error_fingerprint and self._current_error_fingerprint != fingerprint_s:
                return
            self._current_error_node_id = ""
            self._current_error_code = ""
            self._current_error_message = ""
            self._current_error_severity = ""
            self._current_error_fingerprint = ""
            self._current_error_ts_ms = None
        _ = ts_ms
        self._cancel_pending_error_publish()
        self._request_publish_once()

    def record_error(self, *, code: str, message: str, ts_ms: int | None = None) -> None:
        self.report_error(
            node_id=str(self._bus.service_id),
            code=code,
            message=message,
            severity="error",
            fingerprint=None,
            ts_ms=ts_ms,
        )

    def _record_monitor_internal_error(self, *, code: str, message: str, ts_ms: int) -> None:
        if not self._enabled:
            return
        node_id_s = str(self._bus.service_id)
        code_s = str(code or "").strip() or "ERROR"
        message_s = str(message or "")
        fingerprint_s = _derive_monitor_error_fingerprint(
            node_id=node_id_s,
            code=code_s,
            message=message_s,
            fingerprint=None,
        )
        with self._lock:
            if fingerprint_s == self._last_error_fingerprint:
                self._last_error_repeat_count = max(1, int(self._last_error_repeat_count)) + 1
            else:
                self._last_error_repeat_count = 1
            self._last_error_node_id = node_id_s
            self._last_error_code = code_s
            self._last_error_message = message_s
            self._last_error_severity = "error"
            self._last_error_fingerprint = fingerprint_s
            self._last_error_ts_ms = int(ts_ms)
            self._current_error_node_id = node_id_s
            self._current_error_code = code_s
            self._current_error_message = message_s
            self._current_error_severity = "error"
            self._current_error_fingerprint = fingerprint_s
            self._current_error_ts_ms = int(ts_ms)
            self._error_events.append(int(ts_ms))

    def _request_publish_once(self) -> None:
        if self._task is None:
            return
        task = self._publish_once_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._publish_once_task = loop.create_task(self._publish_once(), name="monitor_collector:publish_once")

    def _request_error_publish_once(self, *, fingerprint: str, immediate: bool) -> None:
        if self._task is None:
            return
        fingerprint_s = str(fingerprint or "").strip()
        now_ts = int(now_ms())
        interval_ms = max(0, int(_ERROR_REPEAT_PUBLISH_INTERVAL_MS))
        with self._lock:
            last_fingerprint = str(self._last_error_publish_fingerprint)
            last_ts = int(self._last_error_publish_ts_ms)
            elapsed_ms = now_ts - last_ts
            should_publish_now = (
                bool(immediate)
                or fingerprint_s != last_fingerprint
                or last_ts <= 0
                or elapsed_ms >= interval_ms
            )
            if should_publish_now:
                self._last_error_publish_fingerprint = fingerprint_s
                self._last_error_publish_ts_ms = now_ts
            else:
                delay_ms = max(1, interval_ms - max(0, elapsed_ms))

        if should_publish_now:
            self._cancel_pending_error_publish()
            self._request_publish_once()
            return
        self._ensure_delayed_error_publish(delay_ms=delay_ms)

    def _ensure_delayed_error_publish(self, *, delay_ms: int) -> None:
        task = self._pending_error_publish_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay_s = max(0.001, float(delay_ms) / 1000.0)
        self._pending_error_publish_task = loop.create_task(
            self._delayed_error_publish(delay_s=delay_s),
            name="monitor_collector:error_summary_publish",
        )

    def _cancel_pending_error_publish(self) -> None:
        task = self._pending_error_publish_task
        self._pending_error_publish_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _delayed_error_publish(self, *, delay_s: float) -> None:
        await asyncio.sleep(float(delay_s))
        with self._lock:
            self._last_error_publish_fingerprint = str(self._last_error_fingerprint)
            self._last_error_publish_ts_ms = int(now_ms())
        self._pending_error_publish_task = None
        self._request_publish_once()

    async def start(self) -> None:
        if (not self._enabled) or self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop(), name="monitor_collector")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        publish_once_task = self._publish_once_task
        self._publish_once_task = None
        pending_error_publish_task = self._pending_error_publish_task
        self._pending_error_publish_task = None
        if pending_error_publish_task is not None:
            pending_error_publish_task.cancel()
            await asyncio.gather(pending_error_publish_task, return_exceptions=True)
        if publish_once_task is not None:
            publish_once_task.cancel()
            await asyncio.gather(publish_once_task, return_exceptions=True)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._gpu_sampler.close()

    def _queue_depth(self) -> int:
        return self._bus.data_router.queue_depth()

    def _build_snapshot(self, *, ts_ms: int) -> F8MonitorSnapshot:
        process_percent = self._process_sampler.sample_process_percent()
        system_percent = self._process_sampler.sample_system_percent()
        rss_bytes, vms_bytes = self._memory_sampler.sample()
        gpu = self._gpu_sampler.sample()

        with self._lock:
            self._wait_values.set_window_ms(self._window_ms)
            self._process_values.set_window_ms(self._window_ms)
            self._latency_values.set_window_ms(self._window_ms)
            wait_values = self._wait_values.values(now_ms_value=int(ts_ms))
            process_values = self._process_values.values(now_ms_value=int(ts_ms))
            latency_values = self._latency_values.values(now_ms_value=int(ts_ms))
            error_cutoff = int(ts_ms) - int(self._window_ms)
            while self._error_events and int(self._error_events[0]) < error_cutoff:
                self._error_events.popleft()

            frame = F8MonitorFrame(
                observed=int(self._observed),
                processed=int(self._processed),
                dropped=int(self._dropped),
                localOnlyEmits=int(self._local_only_emits),
                routedCrossEmits=int(self._routed_cross_emits),
                suppressedCrossPublishes=int(self._suppressed_cross_publishes),
                callbackDeliveries=int(self._callback_deliveries),
                bufferPullDeliveries=int(self._buffer_pull_deliveries),
            )
            error = F8MonitorError(
                countWindow=int(len(self._error_events)),
                lastNodeId=str(self._last_error_node_id),
                lastCode=str(self._last_error_code),
                lastMessage=str(self._last_error_message),
                lastSeverity=str(self._last_error_severity),
                lastFingerprint=str(self._last_error_fingerprint),
                lastRepeatCount=int(self._last_error_repeat_count),
                lastTsMs=int(self._last_error_ts_ms) if self._last_error_ts_ms is not None else None,
                currentNodeId=str(self._current_error_node_id),
                currentCode=str(self._current_error_code),
                currentMessage=str(self._current_error_message),
                currentSeverity=str(self._current_error_severity),
                currentTsMs=int(self._current_error_ts_ms) if self._current_error_ts_ms is not None else None,
            )
            ready = bool(self._ready)

        wait_avg = (sum(wait_values) / float(len(wait_values))) if wait_values else None
        process_avg = (sum(process_values) / float(len(process_values))) if process_values else None
        latency_avg = (sum(latency_values) / float(len(latency_values))) if latency_values else None
        snapshot = F8MonitorSnapshot(
            schemaVersion="f8monitor/1",
            serviceId=str(self._bus.service_id),
            serviceClass=str(self._bus._service_class),
            nodeId=str(self._bus.service_id),
            tsMs=int(ts_ms),
            alive=True,
            ready=ready,
            active=bool(self._bus._active),
            uptimeMs=max(0, int(ts_ms - int(self._started_ts_ms))),
            cpu=F8MonitorCpu(
                processPercent=float(process_percent),
                systemPercent=float(system_percent),
            ),
            memory=F8MonitorMemory(
                rssBytes=max(0, int(rss_bytes)),
                vmsBytes=max(0, int(vms_bytes)),
            ),
            gpu=gpu,
            frame=frame,
            timing=F8MonitorTiming(
                processMsAvg=float(process_avg) if process_avg is not None else None,
                processMsP95=_percentile95(process_values),
                waitMsAvg=float(wait_avg) if wait_avg is not None else None,
                waitMsP95=_percentile95(wait_values),
                latencyMsAvg=float(latency_avg) if latency_avg is not None else None,
                latencyMsP95=_percentile95(latency_values),
            ),
            queue=F8MonitorQueue(depth=self._queue_depth()),
            error=error,
        )
        return validate_as(F8MonitorSnapshot, dump_json(snapshot, mode="json", by_alias=True))

    async def _publish_once(self) -> None:
        await asyncio.sleep(0)
        ts = int(now_ms())
        try:
            snapshot = self._build_snapshot(ts_ms=ts)
        except _MONITOR_SNAPSHOT_BUILD_ERRORS as exc:
            self._record_monitor_internal_error(
                code="MONITOR_BUILD_ERROR",
                message=f"{type(exc).__name__}: {exc}",
                ts_ms=ts,
            )
            return
        with self._lock:
            self._latest = snapshot
        try:
            await self._publish_snapshot(snapshot)
        except _MONITOR_SNAPSHOT_PUBLISH_ERRORS as exc:
            self._record_monitor_internal_error(
                code="MONITOR_PUBLISH_ERROR",
                message=f"{type(exc).__name__}: {exc}",
                ts_ms=ts,
            )

    async def _publish_snapshot(self, snapshot: F8MonitorSnapshot) -> None:
        ts_value = int(snapshot.tsMs)
        payload = {
            "value": dump_json(snapshot, mode="json", by_alias=True),
            "ts": ts_value,
        }
        key = data_key(
            str(self._bus.service_id),
            from_node_id=str(self._bus.service_id),
            port_id="monitor",
        )
        await self._bus._transport.publish(key, encode_obj(payload))

    async def _run_loop(self) -> None:
        interval_s = float(self._interval_ms) / 1000.0
        while True:
            await asyncio.sleep(interval_s)
            ts = int(now_ms())
            try:
                snapshot = self._build_snapshot(ts_ms=ts)
            except _MONITOR_SNAPSHOT_BUILD_ERRORS as exc:
                self._record_monitor_internal_error(
                    code="MONITOR_BUILD_ERROR",
                    message=f"{type(exc).__name__}: {exc}",
                    ts_ms=ts,
                )
                continue
            with self._lock:
                self._latest = snapshot
            try:
                await self._publish_snapshot(snapshot)
            except _MONITOR_SNAPSHOT_PUBLISH_ERRORS as exc:
                self._record_monitor_internal_error(
                    code="MONITOR_PUBLISH_ERROR",
                    message=f"{type(exc).__name__}: {exc}",
                    ts_ms=ts,
                )


__all__ = [
    "MONITOR_PORT_NAME",
    "MONITOR_REPORT_SCHEMA_VERSION",
    "MONITOR_SNAPSHOT_SCHEMA_VERSION",
    "MonitorCollector",
    "MonitorCollectorConfig",
    "MonitorContractError",
    "monitor_snapshot_data_port",
    "monitor_snapshot_schema_dict",
    "monitor_snapshot_schema_dict_cached",
    "monitor_snapshot_value_schema",
    "validate_describe_monitor_contract",
    "validate_monitor_report_payload",
    "validate_monitor_snapshot_payload",
]
