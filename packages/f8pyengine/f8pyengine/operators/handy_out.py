from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.codec import coerce_flag, parse_number, parse_int, unwrap_json_value
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS

logger = logging.getLogger(__name__)

OPERATOR_CLASS: Final[str] = "f8.handy_out"
_CONNECTION_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9]{5,64}$")
_MODE_HDSP: Final[int] = 2
_ERROR_DEDUPE_MS: Final[int] = 2000

@dataclass(frozen=True)
class _HandyConfig:
    enabled: bool
    connection_key: str
    base_url: str
    ensure_hdsp_mode: bool
    invert: bool
    min_percent: float
    max_percent: float
    default_duration_ms: int
    request_timeout_ms: int
    min_send_interval_ms: int
    immediate_response: bool
    stop_on_target: bool

@dataclass(frozen=True)
class _PendingCommand:
    position_percent: float
    duration_ms: int
    immediate_response: bool
    stop_on_target: bool

@dataclass(frozen=True)
class _QueuedRequest:
    cfg: _HandyConfig
    cmd: _PendingCommand

@dataclass(frozen=True)
class _HttpResult:
    status_code: int
    headers: dict[str, str]
    json_body: dict[str, Any] | None
    error_message: str

def _now_ms() -> int:
    return int(time.time() * 1000.0)

def _clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return float(minimum)
    if value > maximum:
        return float(maximum)
    return float(value)

def _normalize_base_url(value: str) -> str:
    return str(value).strip().rstrip("/")

class HandyOutRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._lock = asyncio.Lock()
        self._pending_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._pending_request: _QueuedRequest | None = None
        self._active = True

        self._hdsp_mode_ready = False
        self._backoff_until_ms = 0
        self._last_result = 0.0
        self._last_http_status = 0
        self._last_sent_position = 0.0
        self._last_error_message = ""
        self._last_error_signature = ""
        self._last_error_log_ts_ms = 0
        self._sent_commands = 0
        self._dropped_commands = 0
        self._last_sent_ts_ms = 0

        self._published_state_cache: dict[str, Any] = {}

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        self._start_worker()

    async def close(self) -> None:
        self._stop_event.set()
        self._pending_event.set()
        task = self._worker_task
        self._worker_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def validate_state(
        self,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        unwrapped = unwrap_json_value(value)

        if name in ("enabled", "ensureHdspMode", "invert", "immediateResponse", "stopOnTarget"):
            return coerce_flag(unwrapped, default=False)

        if name == "connectionKey":
            key = str(unwrapped or "").strip()
            if not key:
                return ""
            if not _CONNECTION_KEY_RE.fullmatch(key):
                raise ValueError("connectionKey must be 5..64 alphanumeric characters")
            return key

        if name == "baseUrl":
            base_url = _normalize_base_url(str(unwrapped or ""))
            if not base_url:
                raise ValueError("baseUrl must be non-empty")
            parsed = urlparse(base_url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("baseUrl must start with http:// or https:// and include host")
            return base_url

        if name == "minPercent":
            min_percent = self._validate_number(unwrapped, label="minPercent", minimum=0.0, maximum=100.0)
            max_percent = await self._validation_state_number("maxPercent", default=100.0)
            if min_percent > max_percent:
                raise ValueError("minPercent must be <= maxPercent")
            return min_percent

        if name == "maxPercent":
            max_percent = self._validate_number(unwrapped, label="maxPercent", minimum=0.0, maximum=100.0)
            min_percent = await self._validation_state_number("minPercent", default=0.0)
            if max_percent < min_percent:
                raise ValueError("maxPercent must be >= minPercent")
            return max_percent

        if name == "defaultDurationMs":
            return self._validate_int(unwrapped, label="defaultDurationMs", minimum=0, maximum=120000)

        if name == "requestTimeoutMs":
            return self._validate_int(unwrapped, label="requestTimeoutMs", minimum=100, maximum=120000)

        if name == "minSendIntervalMs":
            return self._validate_int(unwrapped, label="minSendIntervalMs", minimum=0, maximum=120000)

        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del value, ts_ms
        name = str(field or "").strip()
        if name in ("connectionKey", "baseUrl", "ensureHdspMode"):
            self._hdsp_mode_ready = False

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        del in_port
        if not self._active:
            await self._emit_data_ports()
            return []

        cfg = await self._read_config()
        if not cfg.enabled:
            await self._emit_data_ports()
            return []

        if not cfg.connection_key:
            await self._set_last_error_message("connectionKey is required")
            await self._emit_data_ports()
            return []

        now_ms = _now_ms()
        if now_ms < int(self._backoff_until_ms):
            await self._mark_dropped()
            await self._emit_data_ports()
            return []

        if cfg.min_send_interval_ms > 0 and self._last_sent_ts_ms > 0:
            elapsed_ms = now_ms - int(self._last_sent_ts_ms)
            if elapsed_ms < int(cfg.min_send_interval_ms):
                await self._mark_dropped()
                await self._emit_data_ports()
                return []

        raw_value = await self.pull("value", ctx_id=exec_id)
        value_num = parse_number(raw_value)
        if value_num is None:
            await self._emit_data_ports()
            return []

        normalized = _clamp(value_num, 0.0, 1.0)
        if cfg.invert:
            normalized = 1.0 - normalized

        span = float(cfg.max_percent) - float(cfg.min_percent)
        position_percent = float(cfg.min_percent) + span * float(normalized)
        position_percent = _clamp(position_percent, 0.0, 100.0)

        duration_ms = cfg.default_duration_ms
        duration_input = parse_int(await self.pull("durationMs", ctx_id=exec_id))
        if duration_input is not None:
            duration_ms = int(_clamp(float(duration_input), 0.0, 120000.0))

        immediate_response = cfg.immediate_response
        immediate_input_raw = await self.pull("immediateResponse", ctx_id=exec_id)
        if unwrap_json_value(immediate_input_raw) is not None:
            immediate_response = coerce_flag(immediate_input_raw, default=immediate_response)

        stop_on_target = cfg.stop_on_target
        stop_input_raw = await self.pull("stopOnTarget", ctx_id=exec_id)
        if unwrap_json_value(stop_input_raw) is not None:
            stop_on_target = coerce_flag(stop_input_raw, default=stop_on_target)

        queued = _QueuedRequest(
            cfg=cfg,
            cmd=_PendingCommand(
                position_percent=float(position_percent),
                duration_ms=int(duration_ms),
                immediate_response=bool(immediate_response),
                stop_on_target=bool(stop_on_target),
            ),
        )
        async with self._lock:
            self._pending_request = queued
        self._pending_event.set()
        await self._emit_data_ports()
        return []

    def _start_worker(self) -> None:
        task = self._worker_task
        if task is not None and not task.done():
            return
        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._worker_loop(), name=f"handy_out:{self.node_id}")

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._pending_event.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            self._pending_event.clear()

            async with self._lock:
                request = self._pending_request
                self._pending_request = None
            if request is None:
                continue
            if not self._active:
                continue
            await self._process_request(request)

    async def _process_request(self, request: _QueuedRequest) -> None:
        cfg = request.cfg
        cmd = request.cmd
        now_ms = _now_ms()
        if now_ms < int(self._backoff_until_ms):
            await self._mark_dropped()
            await self._emit_data_ports()
            return

        if cfg.ensure_hdsp_mode and not self._hdsp_mode_ready:
            mode_resp = await self._http_put_json(cfg=cfg, path="mode", payload={"mode": _MODE_HDSP})
            self._apply_rate_limit_headers(mode_resp.headers)
            self._last_http_status = int(mode_resp.status_code)
            if not self._is_success_response(mode_resp):
                await self._handle_error_response(context="set_mode", response=mode_resp)
                await self._publish_runtime_states()
                await self._emit_data_ports()
                return
            self._hdsp_mode_ready = True
            await self._clear_last_error()

        payload = {
            "position": float(cmd.position_percent),
            "duration": int(cmd.duration_ms),
            "immediateResponse": bool(cmd.immediate_response),
            "stopOnTarget": bool(cmd.stop_on_target),
        }
        hdsp_resp = await self._http_put_json(cfg=cfg, path="hdsp/xpt", payload=payload)
        self._apply_rate_limit_headers(hdsp_resp.headers)
        self._last_http_status = int(hdsp_resp.status_code)
        if not self._is_success_response(hdsp_resp):
            await self._handle_error_response(context="hdsp_xpt", response=hdsp_resp)
            await self._publish_runtime_states()
            await self._emit_data_ports()
            return

        result = self._extract_result_number(hdsp_resp.json_body)
        self._last_result = float(result)
        self._last_sent_position = float(cmd.position_percent)
        self._last_sent_ts_ms = _now_ms()
        self._sent_commands = int(self._sent_commands) + 1
        await self._clear_last_error()
        await self._publish_runtime_states()
        await self._emit_data_ports()

    async def _handle_error_response(self, *, context: str, response: _HttpResult) -> None:
        error_message = self._response_error_message(response)
        if not error_message:
            error_message = f"{context}: request failed"

        method_not_found = False
        body = response.json_body or {}
        error_obj = body.get("error") if isinstance(body, dict) else None
        if isinstance(error_obj, dict):
            code = parse_int(error_obj.get("code"))
            if code == 2002:
                method_not_found = True

        if method_not_found:
            self._hdsp_mode_ready = False

        await self._set_last_error_once(context=context, message=error_message)

    async def _mark_dropped(self) -> None:
        self._dropped_commands = int(self._dropped_commands) + 1
        await self._publish_runtime_states()

    async def _publish_runtime_states(self) -> None:
        message = str(self._last_error_message)
        if message:
            await self.report_error(
                "HANDY_OUT_ERROR",
                message,
                severity="error",
                fingerprint=f"handy-out:{self._last_error_signature or message}",
            )
            return
        await self.clear_error()

    async def _emit_data_ports(self) -> None:
        await self.emit("sentPosition", float(self._last_sent_position))
        await self.emit("httpStatus", int(self._last_http_status))
        await self.emit("result", float(self._last_result))
        await self.emit("error", str(self._last_error_message))

    async def _publish_state_if_changed(self, field: str, value: Any) -> None:
        prev = self._published_state_cache.get(field)
        if prev == value:
            return
        self._published_state_cache[field] = value
        await self.set_state(field, value)

    async def _read_config(self) -> _HandyConfig:
        enabled = coerce_flag(await self._read_raw_state("enabled"), default=True)
        connection_key = str(await self._read_raw_state("connectionKey") or "").strip()
        base_url = _normalize_base_url(str(await self._read_raw_state("baseUrl") or "https://www.handyfeeling.com/api/handy/v2"))
        ensure_hdsp_mode = coerce_flag(await self._read_raw_state("ensureHdspMode"), default=True)
        invert = coerce_flag(await self._read_raw_state("invert"), default=False)
        min_percent = self._state_number(await self._read_raw_state("minPercent"), default=0.0, minimum=0.0, maximum=100.0)
        max_percent = self._state_number(await self._read_raw_state("maxPercent"), default=100.0, minimum=0.0, maximum=100.0)
        if min_percent > max_percent:
            min_percent = max_percent
        default_duration_ms = self._state_int(await self._read_raw_state("defaultDurationMs"), default=100, minimum=0, maximum=120000)
        request_timeout_ms = self._state_int(await self._read_raw_state("requestTimeoutMs"), default=5000, minimum=100, maximum=120000)
        min_send_interval_ms = self._state_int(await self._read_raw_state("minSendIntervalMs"), default=0, minimum=0, maximum=120000)
        immediate_response = coerce_flag(await self._read_raw_state("immediateResponse"), default=False)
        stop_on_target = coerce_flag(await self._read_raw_state("stopOnTarget"), default=False)
        return _HandyConfig(
            enabled=enabled,
            connection_key=connection_key,
            base_url=base_url,
            ensure_hdsp_mode=ensure_hdsp_mode,
            invert=invert,
            min_percent=min_percent,
            max_percent=max_percent,
            default_duration_ms=default_duration_ms,
            request_timeout_ms=request_timeout_ms,
            min_send_interval_ms=min_send_interval_ms,
            immediate_response=immediate_response,
            stop_on_target=stop_on_target,
        )

    async def _read_raw_state(self, name: str) -> Any:
        live = await self.get_state_value(name)
        unwrapped_live = unwrap_json_value(live)
        if unwrapped_live is not None:
            return unwrapped_live
        return unwrap_json_value(self._initial_state.get(name))

    def _state_number(self, value: Any, *, default: float, minimum: float, maximum: float) -> float:
        parsed = parse_number(value)
        if parsed is None:
            return float(default)
        return _clamp(parsed, minimum, maximum)

    def _state_int(self, value: Any, *, default: int, minimum: int, maximum: int) -> int:
        parsed = parse_int(value)
        if parsed is None:
            return int(default)
        return int(_clamp(float(parsed), float(minimum), float(maximum)))

    def _validate_number(self, value: Any, *, label: str, minimum: float, maximum: float) -> float:
        parsed = parse_number(value)
        if parsed is None:
            raise ValueError(f"{label} must be a number")
        if parsed < minimum or parsed > maximum:
            raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
        return float(parsed)

    def _validate_int(self, value: Any, *, label: str, minimum: int, maximum: int) -> int:
        parsed = parse_int(value)
        if parsed is None:
            raise ValueError(f"{label} must be an integer")
        if parsed < minimum or parsed > maximum:
            raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
        return int(parsed)

    async def _validation_state_number(self, field: str, *, default: float) -> float:
        current = await self.get_state_value(field)
        current_unwrapped = unwrap_json_value(current)
        if current_unwrapped is None:
            current_unwrapped = unwrap_json_value(self._initial_state.get(field))
        parsed = parse_number(current_unwrapped)
        if parsed is None:
            return float(default)
        return float(_clamp(parsed, 0.0, 100.0))

    def _response_error_message(self, response: _HttpResult) -> str:
        if response.error_message:
            return str(response.error_message)
        body = response.json_body
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                code = err.get("code")
                name = str(err.get("name") or "").strip()
                message = str(err.get("message") or "").strip()
                if code is not None or name or message:
                    code_part = "" if code is None else f"{code}"
                    text = " ".join(part for part in (code_part, name) if part).strip()
                    if message:
                        if text:
                            return f"{text}: {message}"
                        return message
                    return text
        if int(response.status_code) >= 400:
            return f"HTTP {int(response.status_code)}"
        return ""

    def _is_success_response(self, response: _HttpResult) -> bool:
        if response.error_message:
            return False
        if int(response.status_code) >= 400:
            return False
        if self._response_error_message(response):
            return False
        return True

    def _extract_result_number(self, body: dict[str, Any] | None) -> float:
        if not isinstance(body, dict):
            return 0.0
        raw = body.get("result")
        parsed = parse_number(raw)
        if parsed is None:
            return 0.0
        return float(parsed)

    async def _set_last_error_once(self, *, context: str, message: str) -> None:
        msg = f"{context}: {str(message or '').strip()}"
        now_ms = _now_ms()
        signature = msg
        should_log = True
        if signature == self._last_error_signature and (now_ms - int(self._last_error_log_ts_ms)) < _ERROR_DEDUPE_MS:
            should_log = False
        self._last_error_signature = signature
        self._last_error_log_ts_ms = now_ms
        self._last_error_message = msg
        if should_log:
            logger.error("[%s:handy_out] %s", self.node_id, msg)

    async def _set_last_error_message(self, message: str) -> None:
        self._last_error_message = str(message or "")
        await self._publish_runtime_states()

    async def _clear_last_error(self) -> None:
        if not self._last_error_message:
            return
        self._last_error_message = ""

    def _apply_rate_limit_headers(self, headers: dict[str, str]) -> None:
        remaining_raw = headers.get("x-ratelimit-remaining", "")
        reset_raw = headers.get("x-ratelimit-reset", "")
        remaining = parse_int(remaining_raw)
        reset_ms = parse_int(reset_raw)
        if remaining is None or reset_ms is None:
            return
        if remaining <= 0 and reset_ms > 0:
            self._backoff_until_ms = _now_ms() + int(reset_ms)

    async def _http_put_json(self, *, cfg: _HandyConfig, path: str, payload: dict[str, Any]) -> _HttpResult:
        timeout_s = float(max(100, int(cfg.request_timeout_ms))) / 1000.0
        try:
            return await asyncio.to_thread(
                self._http_put_json_sync,
                cfg.base_url,
                cfg.connection_key,
                path,
                payload,
                timeout_s,
            )
        except TimeoutError as exc:
            return _HttpResult(status_code=0, headers={}, json_body=None, error_message=f"TimeoutError: {exc}")
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            return _HttpResult(status_code=0, headers={}, json_body=None, error_message=f"URLError: {reason}")
        except ValueError as exc:
            return _HttpResult(status_code=0, headers={}, json_body=None, error_message=f"ValueError: {exc}")
        except OSError as exc:
            return _HttpResult(status_code=0, headers={}, json_body=None, error_message=f"OSError: {exc}")
        except Exception as exc:
            logger.exception("[%s:handy_out] unexpected request error path=%s", self.node_id, path, exc_info=exc)
            return _HttpResult(status_code=0, headers={}, json_body=None, error_message=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _http_put_json_sync(
        base_url: str,
        connection_key: str,
        path: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> _HttpResult:
        url = f"{str(base_url).rstrip('/')}/{str(path).lstrip('/')}"
        body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-Connection-Key": str(connection_key),
        }
        req = Request(url=url, data=body_bytes, headers=headers, method="PUT")
        response_headers: dict[str, str] = {}
        response_status = 0
        response_body = b""

        try:
            with urlopen(req, timeout=float(timeout_s)) as resp:
                response_status = int(resp.getcode() or 0)
                response_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
                response_body = bytes(resp.read() or b"")
        except HTTPError as exc:
            response_status = int(exc.code or 0)
            response_headers = {str(k).lower(): str(v) for k, v in (exc.headers.items() if exc.headers else [])}
            response_body = bytes(exc.read() or b"")

        body_text = response_body.decode("utf-8", errors="replace").strip()
        json_body: dict[str, Any] | None = None
        if body_text:
            try:
                parsed = json.loads(body_text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                json_body = parsed
        return _HttpResult(
            status_code=response_status,
            headers=response_headers,
            json_body=json_body,
            error_message="",
        )

HandyOutRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.output",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Handy Out",
    description="Drive The Handy via HDSP using normalized 0..1 input values.",
    tags=["io", "handy", "hdsp", "device", "output"],
    execInPorts=exec_port_specs(["exec"]),
    dataInPorts=[
        F8DataPortSpec(
            name="value",
            description="Normalized position input (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
        ),
        F8DataPortSpec(
            name="durationMs",
            description="Optional duration override for /hdsp/xpt.",
            valueSchema=number_schema(default=100, minimum=0),
            definitionProtected=False,
        ),
        F8DataPortSpec(
            name="immediateResponse",
            description="Optional immediate response override for /hdsp/xpt.",
            valueSchema=boolean_schema(default=False),
            definitionProtected=False,
        ),
        F8DataPortSpec(
            name="stopOnTarget",
            description="Optional stopOnTarget override for /hdsp/xpt.",
            valueSchema=boolean_schema(default=False),
            definitionProtected=False,
        ),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="sentPosition", description="Last sent position percent (0..100).", valueSchema=number_schema(default=0.0)),
        F8DataPortSpec(name="httpStatus", description="Last HTTP status code.", valueSchema=integer_schema(default=0, minimum=0)),
        F8DataPortSpec(name="result", description="Last RPC result code.", valueSchema=number_schema(default=0.0)),
        F8DataPortSpec(name="error", description="Last runtime error.", valueSchema=string_schema(default="")),
    ],
    stateFields=[
        F8StateSpec(
            name="enabled",
            label="Enabled",
            description="Enable/disable Handy output.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="connectionKey",
            label="Connection Key",
            description="The Handy X-Connection-Key value.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="baseUrl",
            label="Base URL",
            description="Handy API base URL. Reset to default when exporting publish JSON.",
            valueSchema=string_schema(default="https://www.handyfeeling.com/api/handy/v2"),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="ensureHdspMode",
            label="Ensure HDSP Mode",
            description="Automatically set mode=HDSP before sending position commands.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="invert",
            label="Invert",
            description="Invert 0..1 input mapping before percent conversion.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="minPercent",
            label="Min Percent",
            description="Mapped output minimum in percent.",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="maxPercent",
            label="Max Percent",
            description="Mapped output maximum in percent.",
            valueSchema=number_schema(default=100.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="defaultDurationMs",
            label="Default Duration (ms)",
            description="Default /hdsp/xpt duration when durationMs input is not provided.",
            valueSchema=integer_schema(default=100, minimum=0, maximum=120000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="requestTimeoutMs",
            label="Request Timeout (ms)",
            description="HTTP request timeout for Handy API calls.",
            valueSchema=integer_schema(default=5000, minimum=100, maximum=120000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="minSendIntervalMs",
            label="Min Send Interval (ms)",
            description="Minimum interval between sent commands (0 means follow tick rate).",
            valueSchema=integer_schema(default=0, minimum=0, maximum=120000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="immediateResponse",
            label="Immediate Response",
            description="Default immediateResponse value for /hdsp/xpt.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="stopOnTarget",
            label="Stop On Target",
            description="Default stopOnTarget value for /hdsp/xpt.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)

def register_operator(registry: Registry) -> Registry:
    registry.register_operator(HandyOutRuntimeNode.SPEC, HandyOutRuntimeNode, overwrite=True)
    return registry
