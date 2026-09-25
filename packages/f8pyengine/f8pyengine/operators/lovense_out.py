from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import asyncio
import json
import logging
import ssl
import time
from dataclasses import dataclass
from typing import Any, Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    boolean_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.codec import coerce_flag, parse_int, parse_number, unwrap_json_value
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS

logger = logging.getLogger(__name__)

OPERATOR_CLASS: Final[str] = "f8.lovense_out"
_ERROR_DEDUPE_MS: Final[int] = 2000

@dataclass(frozen=True)
class _LovenseOutConfig:
    enabled: bool
    command_url: str
    platform_name: str
    request_timeout_ms: int
    min_send_interval_ms: int
    verify_tls: bool
    default_toy: str

@dataclass(frozen=True)
class _HttpResult:
    status_code: int
    headers: dict[str, str]
    body: Any
    raw_body: str
    error_message: str

def _now_ms() -> int:
    return int(time.time() * 1000.0)

def _clamp(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return float(minimum)
    if value > maximum:
        return float(maximum)
    return float(value)

def _normalize_url(value: Any) -> str:
    return str(unwrap_json_value(value) or "").strip()

def _validate_command_url(value: str) -> str:
    command_url = str(value or "").strip()
    if not command_url:
        raise ValueError("commandUrl must be non-empty")
    parsed = urlparse(command_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("commandUrl must start with http:// or https:// and include host")
    return command_url

class LovenseOutRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._active = True
        self._lock = asyncio.Lock()

        self._last_error_message = ""
        self._last_error_signature = ""
        self._last_error_log_ts_ms = 0

        self._last_http_status = 0
        self._last_result_code = 0
        self._last_response: Any = None
        self._last_request: dict[str, Any] = {}
        self._sent_commands = 0
        self._dropped_commands = 0
        self._last_sent_ts_ms = 0
        self._available_toys: list[str] = []
        self._did_initial_toy_refresh = False

        self._published_state_cache: dict[str, Any] = {}

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)
        if not self._active:
            self._did_initial_toy_refresh = False
            return

        if self._did_initial_toy_refresh:
            return

        try:
            cfg = await self._read_config()
        except ValueError as exc:
            await self._set_last_error_once(context="config", message=str(exc))
            await self._publish_runtime_states()
            return

        if not cfg.enabled:
            return

        refreshed = await self._refresh_toys(cfg=cfg, error_context="autoRefreshToys")
        self._did_initial_toy_refresh = bool(refreshed)

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

        if name in ("enabled", "verifyTls", "stop"):
            return coerce_flag(unwrapped, default=False)

        if name == "stopPrevious":
            return coerce_flag(unwrapped, default=True)

        if name == "commandUrl":
            return _validate_command_url(_normalize_url(unwrapped))

        if name == "platformName":
            platform_name = str(unwrapped or "").strip()
            if not platform_name:
                raise ValueError("platformName must be non-empty")
            return platform_name

        if name == "requestTimeoutMs":
            timeout_ms = parse_int(unwrapped)
            if timeout_ms is None or timeout_ms < 100 or timeout_ms > 120000:
                raise ValueError("requestTimeoutMs must be in [100, 120000]")
            return int(timeout_ms)

        if name == "minSendIntervalMs":
            min_interval_ms = parse_int(unwrapped)
            if min_interval_ms is None or min_interval_ms < 0 or min_interval_ms > 120000:
                raise ValueError("minSendIntervalMs must be in [0, 120000]")
            return int(min_interval_ms)

        if name in (
            "vibrate",
            "rotate",
            "pump",
            "thrusting",
            "fingering",
            "suction",
            "depth",
            "oscillate",
            "all",
            "strokeMin",
            "strokeMax",
        ):
            return self._validate_optional_unit_interval(unwrapped, label=name)

        if name == "timeSec":
            time_sec = parse_number(unwrapped)
            if time_sec is None:
                raise ValueError("timeSec must be a number")
            if time_sec < 0.0 or time_sec > 86400.0:
                raise ValueError("timeSec must be in [0, 86400]")
            return float(time_sec)

        if name in ("loopRunningSec", "loopPauseSec"):
            if self._is_none_like(unwrapped):
                return None
            value_sec = parse_number(unwrapped)
            if value_sec is None:
                raise ValueError(f"{name} must be a number or null")
            if value_sec < 0.0 or value_sec > 86400.0:
                raise ValueError(f"{name} must be in [0, 86400]")
            return float(value_sec)

        if name in ("toy", "defaultToy"):
            if self._is_none_like(unwrapped):
                return ""
            return str(unwrapped).strip()

        return value

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        if not self._active:
            return []

        try:
            cfg = await self._read_config()
        except ValueError as exc:
            await self._set_last_error_once(context="config", message=str(exc))
            await self._publish_runtime_states()
            return []

        if not cfg.enabled:
            return []

        trigger_port = str(in_port or "sendPositionCmd").strip().lower()

        if trigger_port == "sendfunctioncmd":
            await self._handle_apply_exec(cfg)
            return []

        if trigger_port == "sendpositioncmd":
            await self._handle_position_exec(exec_id=exec_id, cfg=cfg)
            return []

        await self._set_last_error_once(
            context="execPort",
            message=f"unsupported exec in port: {in_port!r}; expected sendPositionCmd or sendFunctionCmd",
        )
        await self._publish_runtime_states()
        return []

    async def _handle_position_exec(self, *, exec_id: str | int, cfg: _LovenseOutConfig) -> None:
        raw_position = await self.pull("position", ctx_id=exec_id)
        position = parse_number(raw_position)
        if position is None:
            return

        async with self._lock:
            now_ms = _now_ms()
            if cfg.min_send_interval_ms > 0 and self._last_sent_ts_ms > 0:
                elapsed = now_ms - int(self._last_sent_ts_ms)
                if elapsed < int(cfg.min_send_interval_ms):
                    self._dropped_commands = int(self._dropped_commands) + 1
                    await self._publish_runtime_states()
                    return

            position_clamped = _clamp(position, 0.0, 1.0)
            position_int = int(round(position_clamped * 100.0))
            payload: dict[str, Any] = {
                "command": "Position",
                "value": str(position_int),
                "apiVer": 1,
            }
            toy = await self._resolve_target_toy(default_toy=cfg.default_toy)
            if toy is not None:
                payload["toy"] = toy

            await self._send_payload(cfg=cfg, payload=payload)

    async def _handle_apply_exec(self, cfg: _LovenseOutConfig) -> None:
        try:
            payload = await self._build_apply_payload(default_toy=cfg.default_toy)
        except ValueError as exc:
            await self._set_last_error_once(context="sendFunctionCmd", message=str(exc))
            await self._publish_runtime_states()
            return

        if payload is None:
            return

        async with self._lock:
            await self._send_payload(cfg=cfg, payload=payload)

    async def _refresh_toys(self, *, cfg: _LovenseOutConfig, error_context: str) -> bool:
        async with self._lock:
            payload: dict[str, Any] = {
                "command": "GetToys",
                "apiVer": 1,
            }
            self._last_request = dict(payload)
            response = await self._http_post_json(cfg=cfg, payload=payload)
            self._last_http_status = int(response.status_code)
            self._last_response = response.body if response.body is not None else response.raw_body
            self._last_result_code = self._extract_result_code(response)

            if not self._is_success_response(response):
                await self._set_last_error_once(context=error_context, message=self._response_error_message(response))
                await self._publish_runtime_states()
                return False

            await self._clear_last_error()
            self._available_toys = self._parse_available_toy_ids(response.body)
            await self._publish_runtime_states()
            return True

    async def _build_apply_payload(self, *, default_toy: str) -> dict[str, Any] | None:
        stop_enabled = coerce_flag(await self._read_raw_state("stop"), default=False)
        if stop_enabled:
            payload: dict[str, Any] = {
                "command": "Function",
                "action": "Stop",
                "timeSec": 0,
                "apiVer": 1,
            }
            toy = await self._resolve_target_toy(default_toy=default_toy)
            if toy is not None:
                payload["toy"] = toy
            return payload

        actions: list[str] = []

        vibrate = await self._read_optional_level_state("vibrate")
        if vibrate is not None:
            actions.append(f"Vibrate:{self._scale_level(vibrate, max_level=20)}")

        rotate = await self._read_optional_level_state("rotate")
        if rotate is not None:
            actions.append(f"Rotate:{self._scale_level(rotate, max_level=20)}")

        pump = await self._read_optional_level_state("pump")
        if pump is not None:
            actions.append(f"Pump:{self._scale_level(pump, max_level=3)}")

        thrusting = await self._read_optional_level_state("thrusting")
        if thrusting is not None:
            actions.append(f"Thrusting:{self._scale_level(thrusting, max_level=20)}")

        fingering = await self._read_optional_level_state("fingering")
        if fingering is not None:
            actions.append(f"Fingering:{self._scale_level(fingering, max_level=20)}")

        suction = await self._read_optional_level_state("suction")
        if suction is not None:
            actions.append(f"Suction:{self._scale_level(suction, max_level=20)}")

        depth = await self._read_optional_level_state("depth")
        if depth is not None:
            actions.append(f"Depth:{self._scale_level(depth, max_level=3)}")

        oscillate = await self._read_optional_level_state("oscillate")
        if oscillate is not None:
            actions.append(f"Oscillate:{self._scale_level(oscillate, max_level=20)}")

        all_level = await self._read_optional_level_state("all")
        if all_level is not None:
            actions.append(f"All:{self._scale_level(all_level, max_level=20)}")

        stroke_min = await self._read_optional_level_state("strokeMin")
        stroke_max = await self._read_optional_level_state("strokeMax")
        if stroke_min is not None or stroke_max is not None:
            if stroke_min is None or stroke_max is None:
                raise ValueError("strokeMin and strokeMax must both be provided")
            stroke_min_pct = self._scale_level(stroke_min, max_level=100)
            stroke_max_pct = self._scale_level(stroke_max, max_level=100)
            lower = min(stroke_min_pct, stroke_max_pct)
            upper = max(stroke_min_pct, stroke_max_pct)
            actions.append(f"Stroke:{lower}-{upper}")

        if not actions:
            return None

        time_sec = await self._read_required_time_sec()
        stop_previous = coerce_flag(await self._read_raw_state("stopPrevious"), default=True)

        payload = {
            "command": "Function",
            "action": ",".join(actions),
            "timeSec": float(time_sec),
            "stopPrevious": 1 if stop_previous else 0,
            "apiVer": 1,
        }

        loop_running_sec = await self._read_optional_loop_sec("loopRunningSec")
        if loop_running_sec is not None:
            payload["loopRunningSec"] = float(loop_running_sec)

        loop_pause_sec = await self._read_optional_loop_sec("loopPauseSec")
        if loop_pause_sec is not None:
            payload["loopPauseSec"] = float(loop_pause_sec)

        toy = await self._resolve_target_toy(default_toy=default_toy)
        if toy is not None:
            payload["toy"] = toy

        return payload

    async def _send_payload(self, *, cfg: _LovenseOutConfig, payload: dict[str, Any]) -> None:
        self._last_request = dict(payload)
        response = await self._http_post_json(cfg=cfg, payload=payload)
        self._last_http_status = int(response.status_code)
        self._last_response = response.body if response.body is not None else response.raw_body
        self._last_result_code = self._extract_result_code(response)

        if not self._is_success_response(response):
            await self._set_last_error_once(context="request", message=self._response_error_message(response))
            await self._publish_runtime_states()
            return

        await self._clear_last_error()
        self._sent_commands = int(self._sent_commands) + 1
        self._last_sent_ts_ms = _now_ms()
        await self._publish_runtime_states()

    async def _publish_runtime_states(self) -> None:
        await self._publish_state_if_changed("availableToys", list(self._available_toys))

    async def _publish_state_if_changed(self, field: str, value: Any) -> None:
        prev = self._published_state_cache.get(field)
        if prev == value:
            return
        self._published_state_cache[field] = value
        await self.set_state(field, value)

    async def _read_config(self) -> _LovenseOutConfig:
        enabled = coerce_flag(await self._read_raw_state("enabled"), default=True)
        command_url = _normalize_url(await self._read_raw_state("commandUrl"))
        if not command_url:
            command_url = "https://127-0-0-1.lovense.club:30010/command"
        command_url = _validate_command_url(command_url)

        platform_name = str(await self._read_raw_state("platformName") or "Feel8 Studio").strip()
        if not platform_name:
            platform_name = "Feel8 Studio"

        request_timeout_ms = parse_int(await self._read_raw_state("requestTimeoutMs"))
        if request_timeout_ms is None:
            request_timeout_ms = 5000
        request_timeout_ms = max(100, min(120000, int(request_timeout_ms)))

        min_send_interval_ms = parse_int(await self._read_raw_state("minSendIntervalMs"))
        if min_send_interval_ms is None:
            min_send_interval_ms = 100
        min_send_interval_ms = max(0, min(120000, int(min_send_interval_ms)))

        verify_tls = coerce_flag(await self._read_raw_state("verifyTls"), default=True)
        default_toy = str(await self._read_raw_state("defaultToy") or "").strip()

        return _LovenseOutConfig(
            enabled=enabled,
            command_url=command_url,
            platform_name=platform_name,
            request_timeout_ms=request_timeout_ms,
            min_send_interval_ms=min_send_interval_ms,
            verify_tls=verify_tls,
            default_toy=default_toy,
        )

    async def _read_raw_state(self, name: str) -> Any:
        live = await self.get_state_value(name)
        unwrapped_live = unwrap_json_value(live)
        if unwrapped_live is not None:
            return unwrapped_live
        return unwrap_json_value(self._initial_state.get(name))

    async def _read_optional_level_state(self, field: str) -> float | None:
        raw = await self._read_raw_state(field)
        if self._is_none_like(raw):
            return None
        parsed = parse_number(raw)
        if parsed is None:
            return None
        return _clamp(parsed, 0.0, 1.0)

    async def _read_optional_loop_sec(self, field: str) -> float | None:
        raw = await self._read_raw_state(field)
        if self._is_none_like(raw):
            return None
        parsed = parse_number(raw)
        if parsed is None:
            return None
        parsed = _clamp(parsed, 0.0, 86400.0)
        if parsed <= 0.0:
            return None
        return float(parsed)

    async def _read_required_time_sec(self) -> float:
        raw = await self._read_raw_state("timeSec")
        parsed = parse_number(raw)
        if parsed is None:
            return 0.0
        return float(_clamp(parsed, 0.0, 86400.0))

    async def _resolve_target_toy(self, *, default_toy: str) -> str | None:
        raw_toy = await self._read_raw_state("toy")
        if self._is_none_like(raw_toy):
            fallback = str(default_toy or "").strip()
            return fallback or None
        toy = str(raw_toy).strip()
        if not toy:
            fallback = str(default_toy or "").strip()
            return fallback or None
        return toy

    def _scale_level(self, normalized: float, *, max_level: int) -> int:
        value = int(round(float(normalized) * float(max_level)))
        if value < 0:
            return 0
        if value > int(max_level):
            return int(max_level)
        return value

    def _validate_optional_unit_interval(self, value: Any, *, label: str) -> float | None:
        if self._is_none_like(value):
            return None
        parsed = parse_number(value)
        if parsed is None:
            raise ValueError(f"{label} must be a number in [0, 1] or null")
        if parsed < 0.0 or parsed > 1.0:
            raise ValueError(f"{label} must be in [0, 1]")
        return float(parsed)

    @staticmethod
    def _is_none_like(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    def _extract_result_code(self, response: _HttpResult) -> int:
        body = response.body
        if isinstance(body, dict):
            code = parse_int(body.get("code"))
            if code is not None:
                return int(code)
        if response.status_code > 0:
            return int(response.status_code)
        return 0

    def _response_error_message(self, response: _HttpResult) -> str:
        if response.error_message:
            return str(response.error_message)

        body = response.body
        if isinstance(body, dict):
            code = parse_int(body.get("code"))
            type_text = str(body.get("type") or "").strip().lower()
            message_text = str(body.get("message") or "").strip()
            if code is not None and code != 200:
                if message_text:
                    return f"{code}: {message_text}"
                return str(code)
            if type_text == "error":
                if message_text:
                    return message_text
                return "Lovense API type=error"

        if int(response.status_code) >= 400:
            if response.raw_body:
                return f"HTTP {int(response.status_code)}: {response.raw_body}"
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

    def _parse_available_toy_ids(self, body: Any) -> list[str]:
        if not isinstance(body, dict):
            return []
        data = body.get("data")
        if not isinstance(data, dict):
            return []
        toys_value = data.get("toys")
        parsed_toys: Any = toys_value
        if isinstance(toys_value, str):
            toys_text = toys_value.strip()
            if not toys_text:
                return []
            try:
                parsed_toys = json.loads(toys_text)
            except json.JSONDecodeError:
                return []

        toy_ids: list[str] = []
        if isinstance(parsed_toys, dict):
            for toy_key, toy_info in parsed_toys.items():
                toy_id = ""
                if isinstance(toy_info, dict):
                    raw_id = toy_info.get("id")
                    if isinstance(raw_id, str):
                        toy_id = raw_id.strip()
                if not toy_id and isinstance(toy_key, str):
                    toy_id = toy_key.strip()
                if toy_id and toy_id not in toy_ids:
                    toy_ids.append(toy_id)
        return toy_ids

    async def _set_last_error_once(self, *, context: str, message: str) -> None:
        msg = f"{context}: {str(message or '').strip()}"
        now_ms = _now_ms()
        should_log = True
        if msg == self._last_error_signature and (now_ms - int(self._last_error_log_ts_ms)) < _ERROR_DEDUPE_MS:
            should_log = False
        self._last_error_signature = msg
        self._last_error_log_ts_ms = now_ms
        self._last_error_message = msg
        if should_log:
            logger.error("[%s:lovense_out] %s", self.node_id, msg)

    async def _clear_last_error(self) -> None:
        if not self._last_error_message:
            return
        self._last_error_message = ""

    async def _http_post_json(self, *, cfg: _LovenseOutConfig, payload: dict[str, Any]) -> _HttpResult:
        timeout_s = float(max(100, int(cfg.request_timeout_ms))) / 1000.0
        try:
            return await asyncio.to_thread(
                self._http_post_json_sync,
                cfg.command_url,
                cfg.platform_name,
                payload,
                timeout_s,
                cfg.verify_tls,
            )
        except TimeoutError as exc:
            return _HttpResult(status_code=0, headers={}, body=None, raw_body="", error_message=f"TimeoutError: {exc}")
        except URLError as exc:
            reason = exc.reason
            return _HttpResult(status_code=0, headers={}, body=None, raw_body="", error_message=f"URLError: {reason}")
        except ValueError as exc:
            return _HttpResult(status_code=0, headers={}, body=None, raw_body="", error_message=f"ValueError: {exc}")
        except OSError as exc:
            return _HttpResult(status_code=0, headers={}, body=None, raw_body="", error_message=f"OSError: {exc}")
        except Exception as exc:
            logger.exception("[%s:lovense_out] unexpected request error", self.node_id, exc_info=exc)
            return _HttpResult(
                status_code=0,
                headers={},
                body=None,
                raw_body="",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _http_post_json_sync(
        command_url: str,
        platform_name: str,
        payload: dict[str, Any],
        timeout_s: float,
        verify_tls: bool,
    ) -> _HttpResult:
        payload_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "X-platform": str(platform_name),
        }
        request = Request(url=command_url, data=payload_bytes, headers=headers, method="POST")

        response_headers: dict[str, str] = {}
        response_status = 0
        response_body = b""

        ssl_context: ssl.SSLContext | None = None
        parsed_url = urlparse(command_url)
        if parsed_url.scheme == "https" and not verify_tls:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        try:
            if ssl_context is None:
                with urlopen(request, timeout=float(timeout_s)) as response:
                    response_status = int(response.getcode() or 0)
                    response_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
                    response_body = bytes(response.read() or b"")
            else:
                with urlopen(request, timeout=float(timeout_s), context=ssl_context) as response:
                    response_status = int(response.getcode() or 0)
                    response_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
                    response_body = bytes(response.read() or b"")
        except HTTPError as exc:
            response_status = int(exc.code or 0)
            if exc.headers is not None:
                response_headers = {str(k).lower(): str(v) for k, v in exc.headers.items()}
            response_body = bytes(exc.read() or b"")

        body_text = response_body.decode("utf-8", errors="replace").strip()
        parsed_body: Any = None
        if body_text:
            try:
                parsed_body = json.loads(body_text)
            except json.JSONDecodeError:
                parsed_body = None

        return _HttpResult(
            status_code=response_status,
            headers=response_headers,
            body=parsed_body,
            raw_body=body_text,
            error_message="",
        )

LovenseOutRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.output",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Lovense Out",
    description="Send Lovense Local API commands with split channels: sendPositionCmd->Position, sendFunctionCmd->Function.",
    tags=["io", "lovense", "http", "output", "device"],
    execInPorts=exec_port_specs(["sendPositionCmd", "sendFunctionCmd"]),
    dataInPorts=[
        F8DataPortSpec(
            name="position",
            description="Normalized position input (0..1). Sent on sendPositionCmd as Lovense Position command.",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            definitionProtected=True,
        ),
    ],
    dataOutPorts=[],
    stateFields=[
        F8StateSpec(
            name="enabled",
            label="Enabled",
            description="Enable/disable Lovense output.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="commandUrl",
            label="Command URL",
            description="Lovense Local API /command URL. Reset to default when exporting publish JSON.",
            valueSchema=string_schema(default="https://127-0-0-1.lovense.club:30010/command"),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="platformName",
            label="Platform Name",
            description="Value for X-platform request header.",
            valueSchema=string_schema(default="Feel8 Studio"),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="requestTimeoutMs",
            label="Request Timeout (ms)",
            description="HTTP timeout for Lovense requests.",
            valueSchema=integer_schema(default=5000, minimum=100, maximum=120000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="verifyTls",
            label="Verify TLS",
            description="Verify HTTPS certificate when using https:// commandUrl.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="minSendIntervalMs",
            label="Min Send Interval (ms)",
            description="Minimum interval between Position commands sent by sendPositionCmd (0 disables throttling).",
            valueSchema=integer_schema(default=100, minimum=0, maximum=120000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="vibrate",
            label="Vibrate",
            description="Normalized Function Vibrate level (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="rotate",
            label="Rotate",
            description="Normalized Function Rotate level (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="pump",
            label="Pump",
            description="Normalized Function Pump level (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="thrusting",
            label="Thrusting",
            description="Normalized Function Thrusting level (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="fingering",
            label="Fingering",
            description="Normalized Function Fingering level (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="suction",
            label="Suction",
            description="Normalized Function Suction level (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="depth",
            label="Depth",
            description="Normalized Function Depth level (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="oscillate",
            label="Oscillate",
            description="Normalized Function Oscillate level (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="all",
            label="All",
            description="Normalized Function All level (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="strokeMin",
            label="Stroke Min",
            description="Normalized Function Stroke min (0..1). Requires strokeMax.",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="strokeMax",
            label="Stroke Max",
            description="Normalized Function Stroke max (0..1). Requires strokeMin.",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="stop",
            label="Stop",
            description="When true, sendFunctionCmd sends Function Stop.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="timeSec",
            label="Time Sec",
            description="Function timeSec.",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=86400.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="loopRunningSec",
            label="Loop Running Sec",
            description="Optional Function loopRunningSec (omit when empty or <=0).",
            valueSchema=number_schema(minimum=0.0, maximum=86400.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="loopPauseSec",
            label="Loop Pause Sec",
            description="Optional Function loopPauseSec (omit when empty or <=0).",
            valueSchema=number_schema(minimum=0.0, maximum=86400.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="stopPrevious",
            label="Stop Previous",
            description="Function stopPrevious (true->1, false->0).",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="toy",
            label="Toy",
            description="Optional target toy id. Empty uses defaultToy.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            valueRequired=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select, optionsFromState="availableToys"),
            showOnNode=True,
        ),
        F8StateSpec(
            name="defaultToy",
            label="Default Toy",
            description="Fallback toy id when toy is empty.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="availableToys",
            label="Available Toys",
            description="Discovered toy IDs from GetToys response.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)

def register_operator(registry: Registry) -> Registry:
    registry.register_operator(LovenseOutRuntimeNode.SPEC, LovenseOutRuntimeNode, overwrite=True)
    return registry
