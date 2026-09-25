from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, cast

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
    complex_object_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.codec import coerce_flag, coerce_float, coerce_int, coerce_str
from f8pysdk.codec import unwrap_json_value
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS

logger = logging.getLogger(__name__)
try:
    from buttplug import ButtplugError as _ButtplugError
except ImportError:
    _ButtplugError = RuntimeError

_BUTTPLUG_RUNTIME_ERRORS: tuple[type[BaseException], ...] = (
    _ButtplugError,
    OSError,
    RuntimeError,
    TimeoutError,
)
_BUTTPLUG_COMMAND_BUILD_ERRORS: tuple[type[BaseException], ...] = (
    *_BUTTPLUG_RUNTIME_ERRORS,
    TypeError,
    ValueError,
)
_STATE_READ_ERRORS = (LookupError, OSError, RuntimeError, TypeError, ValueError)

OPERATOR_CLASS = "f8.buttplug_out"

_OUTPUT_VIBRATE = "Vibrate"
_OUTPUT_ROTATE = "Rotate"
_OUTPUT_OSCILLATE = "Oscillate"
_OUTPUT_POSITION = "Position"
_OUTPUT_POSITION_WITH_DURATION = "HwPositionWithDuration"
_POSITION_CLAMP_MIN = 0.0001
_POSITION_CLAMP_MAX = 0.9999

def _range2_schema():
    return array_schema(items=integer_schema())

def _feature_output_info_schema():
    return complex_object_schema(
        properties={
            "featureIndex": integer_schema(),
            "description": string_schema(),
            "stepRange": _range2_schema(),
            "durationRange": _range2_schema(),
        }
    )

def _feature_input_info_schema():
    return complex_object_schema(
        properties={
            "featureIndex": integer_schema(),
            "description": string_schema(),
            "valueRanges": array_schema(items=_range2_schema()),
            "commands": array_schema(items=string_schema()),
        }
    )

def _device_info_schema():
    return complex_object_schema(
        properties={
            "index": integer_schema(),
            "name": string_schema(),
            "displayName": string_schema(),
            "messageTimingGapMs": integer_schema(),
            "outputs": complex_object_schema(
                properties={
                    "Vibrate": array_schema(items=_feature_output_info_schema()),
                    "Rotate": array_schema(items=_feature_output_info_schema()),
                    "Oscillate": array_schema(items=_feature_output_info_schema()),
                    "Position": array_schema(items=_feature_output_info_schema()),
                    "HwPositionWithDuration": array_schema(items=_feature_output_info_schema()),
                }
            ),
            "inputs": complex_object_schema(
                properties={
                    "Battery": array_schema(items=_feature_input_info_schema()),
                    "RSSI": array_schema(items=_feature_input_info_schema()),
                    "Button": array_schema(items=_feature_input_info_schema()),
                }
            ),
        }
    )

class _FeatureOutputDefinitionLike(Protocol):
    value: tuple[int, int]
    duration: tuple[int, int] | None

class _FeatureInputDefinitionLike(Protocol):
    value: list[tuple[int, int]]
    command: list[str]

class _DeviceFeatureLike(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def description(self) -> str | None: ...

    @property
    def outputs(self) -> dict[str, _FeatureOutputDefinitionLike] | None: ...

    @property
    def inputs(self) -> dict[str, _FeatureInputDefinitionLike] | None: ...

    def has_output(self, output_type: str) -> bool: ...

    async def run_output(self, command: Any) -> None: ...

class _DeviceLike(Protocol):
    @property
    def index(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def display_name(self) -> str | None: ...

    @property
    def message_timing_gap(self) -> int: ...

    @property
    def features(self) -> dict[int, _DeviceFeatureLike]: ...

    def has_output(self, output_type: str) -> bool: ...

    async def run_output(self, command: Any) -> None: ...

    async def stop(self, inputs: bool = True, outputs: bool = True) -> None: ...

class _ButtplugClientLike(Protocol):
    @property
    def connected(self) -> bool: ...

    @property
    def scanning(self) -> bool: ...

    @property
    def devices(self) -> dict[int, _DeviceLike]: ...

    @property
    def on_device_added(self) -> Callable[[_DeviceLike], None] | Callable[[_DeviceLike], Awaitable[None]] | None: ...

    @on_device_added.setter
    def on_device_added(
        self,
        callback: Callable[[_DeviceLike], None] | Callable[[_DeviceLike], Awaitable[None]] | None,
    ) -> None: ...

    @property
    def on_device_removed(self) -> Callable[[_DeviceLike], None] | Callable[[_DeviceLike], Awaitable[None]] | None: ...

    @on_device_removed.setter
    def on_device_removed(
        self,
        callback: Callable[[_DeviceLike], None] | Callable[[_DeviceLike], Awaitable[None]] | None,
    ) -> None: ...

    @property
    def on_scanning_finished(self) -> Callable[[], None] | Callable[[], Awaitable[None]] | None: ...

    @on_scanning_finished.setter
    def on_scanning_finished(self, callback: Callable[[], None] | Callable[[], Awaitable[None]] | None) -> None: ...

    @property
    def on_server_disconnect(self) -> Callable[[], None] | Callable[[], Awaitable[None]] | None: ...

    @on_server_disconnect.setter
    def on_server_disconnect(self, callback: Callable[[], None] | Callable[[], Awaitable[None]] | None) -> None: ...

    @property
    def on_error(self) -> Callable[[Exception], None] | Callable[[Exception], Awaitable[None]] | None: ...

    @on_error.setter
    def on_error(
        self, callback: Callable[[Exception], None] | Callable[[Exception], Awaitable[None]] | None
    ) -> None: ...

    async def connect(self, url: str) -> None: ...

    async def disconnect(self) -> None: ...

    async def start_scanning(self) -> None: ...

    async def stop_scanning(self) -> None: ...

@dataclass(frozen=True)
class _ButtplugSymbols:
    buttplug_client_cls: type
    device_output_command_cls: type
    output_type_enum: type

@dataclass(frozen=True)
class _OutConfig:
    enabled: bool
    ws_url: str
    auto_connect: bool
    auto_scan_on_connect: bool
    scan_duration_ms: int
    reconnect_interval_ms: int

def _now_ms() -> int:
    return int(time.time() * 1000.0)

def _device_token(*, index: int, name: str) -> str:
    return f"{int(index)}|{str(name)}"

def _device_index_from_token(token: str) -> int | None:
    raw = str(token or "").strip()
    if not raw:
        return None
    parts = raw.split("|", 1)
    left = parts[0].strip()
    if not left:
        return None
    try:
        return int(left)
    except ValueError:
        return None

class ButtplugOutRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._client: _ButtplugClientLike | None = None
        self._client_url: str = ""
        self._worker_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._active = True
        self._tick_lock = asyncio.Lock()

        self._rescan_requested = False
        self._force_reconnect = False
        self._last_connect_attempt_ms = 0

        self._sent_commands = 0
        self._last_command_ts_ms = 0

        self._last_error_message = ""
        self._last_error_signature = ""
        self._last_error_logged_ms = 0

        self._published_state_cache: dict[str, Any] = {}

        self._symbols: _ButtplugSymbols | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        self._start_worker()

    async def close(self) -> None:
        self._stop_event.set()
        task = self._worker_task
        self._worker_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._disconnect_client()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)
        if not self._active:
            stop_on_deactivate = await self._read_bool_state("stopOnDeactivate", default=True)
            if stop_on_deactivate:
                await self._stop_target_device_outputs()
            await self._disconnect_client()

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()

        if name in (
            "enabled",
            "autoConnect",
            "autoScanOnConnect",
            "rescan",
            "stopOnDeactivate",
        ):
            return coerce_flag(value, default=False)

        if name == "wsUrl":
            out = coerce_str(value, default="ws://127.0.0.1:12345")
            if not (out.startswith("ws://") or out.startswith("wss://")):
                raise ValueError("wsUrl must start with ws:// or wss://")
            return out

        if name == "selectedDevice":
            return coerce_str(value, default="")

        if name in ("scanDurationMs",):
            return coerce_int(value, default=5000, minimum=100, maximum=120000)

        if name in ("reconnectIntervalMs",):
            return coerce_int(value, default=2000, minimum=100, maximum=120000)

        if name in (
            "defaultPositionDurationMs",
            "vibrateFeatureIndex",
            "rotateFeatureIndex",
            "oscillateFeatureIndex",
            "positionFeatureIndex",
        ):
            if name == "defaultPositionDurationMs":
                return coerce_int(value, default=500, minimum=0, maximum=120000)
            return coerce_int(value, default=-1, minimum=-1, maximum=4096)

        if name in ("vibrate", "oscillate"):
            v = unwrap_json_value(value)
            if v is None or (isinstance(v, str) and not v.strip()):
                return None
            return coerce_float(v, default=0.0, minimum=0.0, maximum=1.0)

        if name in ("rotate",):
            v = unwrap_json_value(value)
            if v is None or (isinstance(v, str) and not v.strip()):
                return None
            return coerce_float(v, default=0.0, minimum=-1.0, maximum=1.0)

        if name in ("stop",):
            return coerce_flag(value, default=False)

        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "rescan":
            if coerce_flag(value, default=False):
                self._rescan_requested = True
                await self._publish_state_if_changed("rescan", False)
            return

        if name in ("wsUrl",):
            self._force_reconnect = True
            return

        if name in ("selectedDevice",):
            await self._publish_device_snapshot()
            return

        if name in ("enabled",):
            enabled = coerce_flag(value, default=True)
            if not enabled:
                await self._disconnect_client()
            return

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        if not self._active:
            return []
        await self._tick_once()

        target = await self._resolve_target_device(update_selection=True)
        if target is None:
            return []

        trigger_port = str(in_port or "sendPositionCmd").strip().lower()
        if trigger_port == "sendfunctioncmd":
            await self._handle_send_function_cmd(target=target)
            return []

        if trigger_port == "sendpositioncmd":
            await self._handle_send_position_cmd(target=target, exec_id=exec_id)
            return []

        await self._set_last_error_message(
            f"unsupported exec in port: {in_port!r}; expected sendPositionCmd or sendFunctionCmd"
        )
        return []

    def _start_worker(self) -> None:
        task = self._worker_task
        if task is not None and not task.done():
            return
        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._worker_loop(), name=f"buttplug_out:{self.node_id}")

    async def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            await self._tick_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

    async def _tick_once(self) -> None:
        async with self._tick_lock:
            if not self._active:
                await self._publish_runtime_status()
                await self._publish_device_snapshot()
                return
            cfg = await self._read_out_config()
            await self._reconcile_connection(cfg)
            if self._rescan_requested:
                self._rescan_requested = False
                await self._run_scan_cycle(cfg)
            await self._publish_runtime_status()
            await self._publish_device_snapshot()

    async def _read_out_config(self) -> _OutConfig:
        enabled = await self._read_bool_state("enabled", default=True)
        ws_url = await self._read_str_state("wsUrl", default="ws://127.0.0.1:12345")
        auto_connect = await self._read_bool_state("autoConnect", default=True)
        auto_scan_on_connect = await self._read_bool_state("autoScanOnConnect", default=True)
        scan_duration_ms = await self._read_int_state("scanDurationMs", default=5000, minimum=100, maximum=120000)
        reconnect_interval_ms = await self._read_int_state(
            "reconnectIntervalMs",
            default=2000,
            minimum=100,
            maximum=120000,
        )
        return _OutConfig(
            enabled=enabled,
            ws_url=ws_url,
            auto_connect=auto_connect,
            auto_scan_on_connect=auto_scan_on_connect,
            scan_duration_ms=scan_duration_ms,
            reconnect_interval_ms=reconnect_interval_ms,
        )

    def _load_buttplug_symbols(self) -> _ButtplugSymbols:
        if self._symbols is not None:
            return self._symbols
        try:
            from buttplug import ButtplugClient, DeviceOutputCommand, OutputType
        except ImportError as exc:
            raise RuntimeError("buttplug package is required (pip install buttplug>=1.0.0)") from exc
        self._symbols = _ButtplugSymbols(
            buttplug_client_cls=ButtplugClient,
            device_output_command_cls=DeviceOutputCommand,
            output_type_enum=OutputType,
        )
        return self._symbols

    def _create_client(self) -> _ButtplugClientLike:
        symbols = self._load_buttplug_symbols()
        client_obj = symbols.buttplug_client_cls("Feel8 Buttplug Out")
        return cast(_ButtplugClientLike, client_obj)

    def _build_output_command(self, *, output_name: str, value: float, duration_ms: int | None) -> Any:
        symbols = self._load_buttplug_symbols()
        output_enum = symbols.output_type_enum

        if output_name == _OUTPUT_VIBRATE:
            output_type = output_enum.VIBRATE
        elif output_name == _OUTPUT_ROTATE:
            output_type = output_enum.ROTATE
        elif output_name == _OUTPUT_OSCILLATE:
            output_type = output_enum.OSCILLATE
        elif output_name == _OUTPUT_POSITION:
            output_type = output_enum.POSITION
        elif output_name == _OUTPUT_POSITION_WITH_DURATION:
            output_type = output_enum.POSITION_WITH_DURATION
        else:
            raise ValueError(f"unsupported output name: {output_name}")

        if duration_ms is None:
            return symbols.device_output_command_cls(output_type, float(value))
        return symbols.device_output_command_cls(output_type, float(value), duration=int(duration_ms))

    async def _reconcile_connection(self, cfg: _OutConfig) -> None:
        client = self._client

        if not cfg.enabled:
            if client is not None and client.connected:
                await self._disconnect_client()
            return

        if client is not None and client.connected and cfg.ws_url != self._client_url:
            await self._disconnect_client()

        if not cfg.auto_connect:
            return

        if self._client is not None and self._client.connected and not self._force_reconnect:
            return

        now = _now_ms()
        last_attempt = int(self._last_connect_attempt_ms)
        if last_attempt > 0 and (now - last_attempt) < int(cfg.reconnect_interval_ms):
            return

        self._last_connect_attempt_ms = now
        self._force_reconnect = False

        try:
            if self._client is None:
                self._client = self._create_client()
                self._bind_client_callbacks(self._client)

            if not self._client.connected:
                await self._client.connect(cfg.ws_url)
                self._client_url = str(cfg.ws_url)
                await self._clear_last_error()
                if cfg.auto_scan_on_connect:
                    await self._run_scan_cycle(cfg)
        except _BUTTPLUG_RUNTIME_ERRORS as exc:
            await self._set_last_error_once("connect_failed", exc)

    async def _disconnect_client(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            if client.connected:
                await client.disconnect()
        except _BUTTPLUG_RUNTIME_ERRORS as exc:
            await self._set_last_error_once("disconnect_failed", exc)

    def _bind_client_callbacks(self, client: _ButtplugClientLike) -> None:
        client.on_device_added = self._on_device_added
        client.on_device_removed = self._on_device_removed
        client.on_scanning_finished = self._on_scanning_finished
        client.on_server_disconnect = self._on_server_disconnect
        client.on_error = self._on_client_error

    async def _on_device_added(self, device: _DeviceLike) -> None:
        del device
        await self._publish_device_snapshot()

    async def _on_device_removed(self, device: _DeviceLike) -> None:
        del device
        await self._publish_device_snapshot()

    async def _on_scanning_finished(self) -> None:
        await self._publish_runtime_status()
        await self._publish_device_snapshot()

    async def _on_server_disconnect(self) -> None:
        await self._publish_runtime_status()
        await self._publish_device_snapshot()

    async def _on_client_error(self, exc: Exception) -> None:
        await self._set_last_error_once("client_error", exc)

    async def _run_scan_cycle(self, cfg: _OutConfig) -> None:
        client = self._client
        if client is None or not client.connected:
            return
        try:
            if not client.scanning:
                await client.start_scanning()
            await self._publish_runtime_status()
            sleep_s = float(max(100, int(cfg.scan_duration_ms))) / 1000.0
            await asyncio.sleep(sleep_s)
            if client.connected and client.scanning:
                await client.stop_scanning()
            await self._publish_runtime_status()
            await self._publish_device_snapshot()
        except _BUTTPLUG_RUNTIME_ERRORS as exc:
            await self._set_last_error_once("scan_failed", exc)

    async def _resolve_target_device(self, *, update_selection: bool) -> _DeviceLike | None:
        client = self._client
        if client is None or not client.connected:
            return None

        devices = client.devices
        if not devices:
            return None

        selected_token = await self._read_str_state("selectedDevice", default="")
        selected_index = _device_index_from_token(selected_token)

        target: _DeviceLike | None = None
        if selected_index is not None:
            target = devices.get(int(selected_index))

        if target is None:
            keys = sorted(devices.keys())
            if not keys:
                return None
            target = devices[int(keys[0])]
            if update_selection:
                token = _device_token(index=target.index, name=target.name)
                await self._publish_state_if_changed("selectedDevice", token)

        if update_selection:
            await self._publish_selected_device_info(target)

        return target

    async def _dispatch_output_value(
        self,
        *,
        device: _DeviceLike,
        output_name: str,
        feature_state_name: str,
        value: float,
        duration_ms: int | None,
        fallback_output_name: str | None = None,
    ) -> None:
        feature_index = await self._read_int_state(feature_state_name, default=-1, minimum=-1, maximum=4096)

        selected_output_name = output_name
        if (
            not device.has_output(output_name)
            and fallback_output_name is not None
            and device.has_output(fallback_output_name)
        ):
            selected_output_name = fallback_output_name

        if selected_output_name == output_name and not device.has_output(output_name):
            return

        try:
            command = self._build_output_command(
                output_name=selected_output_name,
                value=float(value),
                duration_ms=duration_ms if selected_output_name == _OUTPUT_POSITION_WITH_DURATION else None,
            )
        except _BUTTPLUG_COMMAND_BUILD_ERRORS as exc:
            await self._set_last_error_once("build_command_failed", exc)
            return

        try:
            if feature_index < 0:
                await device.run_output(command)
                await self._mark_command_sent()
                return

            feature = device.features.get(int(feature_index))
            if feature is None:
                await self._set_last_error_message(
                    f"feature index not found for {selected_output_name}: {feature_index}"
                )
                return
            if not feature.has_output(selected_output_name):
                await self._set_last_error_message(f"feature {feature_index} does not support {selected_output_name}")
                return

            await feature.run_output(command)
            await self._mark_command_sent()
        except _BUTTPLUG_RUNTIME_ERRORS as exc:
            await self._set_last_error_once(f"send_{selected_output_name}_failed", exc)

    async def _handle_send_position_cmd(self, *, target: _DeviceLike, exec_id: str | int) -> None:
        raw_position = unwrap_json_value(await self.pull("position", ctx_id=exec_id))
        if raw_position is None:
            return
        if isinstance(raw_position, str) and not raw_position.strip():
            return
        if isinstance(raw_position, bool):
            return
        try:
            value_raw = float(raw_position)
        except (TypeError, ValueError):
            return
        if value_raw != value_raw:
            return
        if value_raw in (float("inf"), float("-inf")):
            return
        value = coerce_float(
            value_raw,
            default=_POSITION_CLAMP_MIN,
            minimum=_POSITION_CLAMP_MIN,
            maximum=_POSITION_CLAMP_MAX,
        )
        if value is None:
            return

        position_duration = await self._read_position_duration_ms()
        await self._dispatch_output_value(
            device=target,
            output_name=_OUTPUT_POSITION_WITH_DURATION,
            feature_state_name="positionFeatureIndex",
            value=value,
            duration_ms=position_duration,
            fallback_output_name=_OUTPUT_POSITION,
        )

    async def _handle_send_function_cmd(self, *, target: _DeviceLike) -> None:
        stop_raw = await self._read_raw_state("stop")
        stop_flag = coerce_flag(stop_raw, default=False)
        if stop_flag:
            try:
                await target.stop(inputs=False, outputs=True)
                await self._mark_command_sent()
            except _BUTTPLUG_RUNTIME_ERRORS as exc:
                await self._set_last_error_once("device_stop_failed", exc)
            return

        vibrate = await self._read_optional_level_state("vibrate", minimum=0.0, maximum=1.0)
        if vibrate is not None:
            await self._dispatch_output_value(
                device=target,
                output_name=_OUTPUT_VIBRATE,
                feature_state_name="vibrateFeatureIndex",
                value=vibrate,
                duration_ms=None,
            )

        rotate = await self._read_optional_level_state("rotate", minimum=-1.0, maximum=1.0)
        if rotate is not None:
            await self._dispatch_output_value(
                device=target,
                output_name=_OUTPUT_ROTATE,
                feature_state_name="rotateFeatureIndex",
                value=rotate,
                duration_ms=None,
            )

        oscillate = await self._read_optional_level_state("oscillate", minimum=0.0, maximum=1.0)
        if oscillate is not None:
            await self._dispatch_output_value(
                device=target,
                output_name=_OUTPUT_OSCILLATE,
                feature_state_name="oscillateFeatureIndex",
                value=oscillate,
                duration_ms=None,
            )

    async def _read_position_duration_ms(self) -> int:
        return await self._read_int_state("defaultPositionDurationMs", default=500, minimum=0, maximum=120000)

    async def _read_optional_level_state(self, name: str, *, minimum: float, maximum: float) -> float | None:
        raw = await self._read_raw_state(name)
        if raw is None:
            return None
        if isinstance(raw, str) and not raw.strip():
            return None
        return coerce_float(raw, default=0.0, minimum=minimum, maximum=maximum)

    async def _stop_target_device_outputs(self) -> None:
        target = await self._resolve_target_device(update_selection=False)
        if target is None:
            return
        try:
            await target.stop(inputs=False, outputs=True)
            await self._mark_command_sent()
        except _BUTTPLUG_RUNTIME_ERRORS as exc:
            await self._set_last_error_once("stop_on_deactivate_failed", exc)

    async def _mark_command_sent(self) -> None:
        self._sent_commands = int(self._sent_commands) + 1
        self._last_command_ts_ms = _now_ms()

    async def _publish_runtime_status(self) -> None:
        client = self._client
        connected = bool(client.connected) if client is not None else False
        scanning = bool(client.scanning) if client is not None else False
        await self._publish_state_if_changed("connected", connected)
        await self._publish_state_if_changed("scanning", scanning)

    async def _publish_device_snapshot(self) -> None:
        client = self._client
        if client is None or not client.connected:
            await self._publish_state_if_changed("availableDevices", [])
            await self._publish_state_if_changed("deviceInfos", [])
            await self._publish_selected_device_info(None)
            return

        devices = client.devices
        available_devices: list[str] = []
        device_infos: list[dict[str, Any]] = []

        for device_index in sorted(devices.keys()):
            device = devices[int(device_index)]
            available_devices.append(_device_token(index=device.index, name=device.name))
            device_infos.append(self._build_device_info(device))

        await self._publish_state_if_changed("availableDevices", available_devices)
        await self._publish_state_if_changed("deviceInfos", device_infos)

        target = await self._resolve_target_device(update_selection=False)
        await self._publish_selected_device_info(target)

    def _build_device_info(self, device: _DeviceLike) -> dict[str, Any]:
        outputs: dict[str, list[dict[str, Any]]] = {}
        inputs: dict[str, list[dict[str, Any]]] = {}

        for feature_index in sorted(device.features.keys()):
            feature = device.features[int(feature_index)]
            feature_desc = str(feature.description or "")

            output_defs = feature.outputs
            if output_defs is not None:
                for output_name, output_def in output_defs.items():
                    if output_name not in outputs:
                        outputs[output_name] = []
                    step_range = [int(output_def.value[0]), int(output_def.value[1])]
                    duration_range: list[int] = []
                    if output_def.duration is not None:
                        duration_range = [int(output_def.duration[0]), int(output_def.duration[1])]
                    outputs[output_name].append(
                        {
                            "featureIndex": int(feature.index),
                            "description": feature_desc,
                            "stepRange": step_range,
                            "durationRange": duration_range,
                        }
                    )

            input_defs = feature.inputs
            if input_defs is not None:
                for input_name, input_def in input_defs.items():
                    if input_name not in inputs:
                        inputs[input_name] = []
                    value_ranges: list[list[int]] = []
                    for item in list(input_def.value or []):
                        value_ranges.append([int(item[0]), int(item[1])])
                    commands = [str(c) for c in list(input_def.command or [])]
                    inputs[input_name].append(
                        {
                            "featureIndex": int(feature.index),
                            "description": feature_desc,
                            "valueRanges": value_ranges,
                            "commands": commands,
                        }
                    )

        return {
            "index": int(device.index),
            "name": str(device.name),
            "displayName": str(device.display_name or ""),
            "messageTimingGapMs": int(device.message_timing_gap),
            "outputs": outputs,
            "inputs": inputs,
        }

    async def _publish_selected_device_info(self, device: _DeviceLike | None) -> None:
        if device is None:
            await self._publish_state_if_changed("selectedDeviceInfo", None)
            return
        await self._publish_state_if_changed("selectedDeviceInfo", self._build_device_info(device))

    async def _publish_state_if_changed(self, field: str, value: Any) -> None:
        prev = self._published_state_cache.get(field)
        if prev == value:
            return
        self._published_state_cache[field] = value
        await self.set_state(field, value)

    async def _set_last_error_once(self, context: str, exc: BaseException) -> None:
        message = f"{context}: {type(exc).__name__}: {exc}"
        signature = f"{context}|{type(exc).__name__}|{exc}"
        now = _now_ms()

        should_log = True
        if signature == self._last_error_signature and (now - int(self._last_error_logged_ms)) < 2000:
            should_log = False

        self._last_error_signature = signature
        self._last_error_logged_ms = now

        await self._set_last_error_message(message)
        if should_log:
            logger.error("[%s:buttplug_out] %s", self.node_id, message, exc_info=(type(exc), exc, exc.__traceback__))

    async def _set_last_error_message(self, message: str) -> None:
        self._last_error_message = str(message or "")

    async def _clear_last_error(self) -> None:
        if not self._last_error_message:
            return
        self._last_error_message = ""

    async def _read_raw_state(self, name: str) -> Any:
        live: Any
        try:
            live = await self.get_state_value(name)
        except _STATE_READ_ERRORS as exc:
            await self._set_last_error_once(f"read_state_{name}_failed", exc)
            live = None
        if live is not None:
            return unwrap_json_value(live)
        return unwrap_json_value(self._initial_state.get(name))

    async def _read_bool_state(self, name: str, *, default: bool) -> bool:
        return coerce_flag(await self._read_raw_state(name), default=default)

    async def _read_int_state(
        self, name: str, *, default: int, minimum: int | None = None, maximum: int | None = None
    ) -> int:
        return coerce_int(await self._read_raw_state(name), default=default, minimum=minimum, maximum=maximum)

    async def _read_str_state(self, name: str, *, default: str) -> str:
        return coerce_str(await self._read_raw_state(name), default=default)

ButtplugOutRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.output",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Buttplug Out",
    description="Connect to Intiface/Buttplug with split channels: sendPositionCmd->position, sendFunctionCmd->state.",
    tags=["io", "buttplug", "intiface", "haptics", "device"],
    execInPorts=exec_port_specs(["sendPositionCmd", "sendFunctionCmd"]),
    dataInPorts=[
        F8DataPortSpec(
            name="position",
            description="Position-channel target (0.0001..0.9999) used by sendPositionCmd.",
            valueSchema=number_schema(minimum=_POSITION_CLAMP_MIN, maximum=_POSITION_CLAMP_MAX),
            definitionProtected=True,
        ),
    ],
    dataOutPorts=[],
    stateFields=[
        F8StateSpec(
            name="enabled",
            label="Enabled",
            description="Enable connection and output control.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="wsUrl",
            label="WebSocket URL",
            description="Buttplug server websocket URL. Reset to default when exporting publish JSON.",
            valueSchema=string_schema(default="ws://127.0.0.1:12345"),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="autoConnect",
            label="Auto Connect",
            description="Automatically connect while enabled.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="autoScanOnConnect",
            label="Auto Scan On Connect",
            description="Start and stop scan once after connect.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="scanDurationMs",
            label="Scan Duration (ms)",
            description="Scan duration before stop when scan is triggered.",
            valueSchema=integer_schema(default=5000, minimum=100, maximum=120000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="reconnectIntervalMs",
            label="Reconnect Interval (ms)",
            description="Reconnect throttle interval.",
            valueSchema=integer_schema(default=2000, minimum=100, maximum=120000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="selectedDevice",
            label="Selected Device",
            description='Target token: "index|name".',
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            valueRequired=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select, optionsFromState="availableDevices"),
            showOnNode=True,
        ),
        F8StateSpec(
            name="rescan",
            label="Rescan",
            description="Set true to trigger one scan cycle; runtime resets it to false.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="vibrateFeatureIndex",
            label="Vibrate Feature Index",
            description="Feature index for vibrate (-1 = all).",
            valueSchema=integer_schema(default=-1, minimum=-1, maximum=4096),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="rotateFeatureIndex",
            label="Rotate Feature Index",
            description="Feature index for rotate (-1 = all).",
            valueSchema=integer_schema(default=-1, minimum=-1, maximum=4096),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="oscillateFeatureIndex",
            label="Oscillate Feature Index",
            description="Feature index for oscillate (-1 = all).",
            valueSchema=integer_schema(default=-1, minimum=-1, maximum=4096),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="positionFeatureIndex",
            label="Position Feature Index",
            description="Feature index for position (-1 = all).",
            valueSchema=integer_schema(default=-1, minimum=-1, maximum=4096),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="defaultPositionDurationMs",
            label="Default Position Duration (ms)",
            description="Default duration for position output.",
            valueSchema=integer_schema(default=500, minimum=0, maximum=120000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="vibrate",
            label="Vibrate",
            description="Function-channel vibrate intensity (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="rotate",
            label="Rotate",
            description="Function-channel rotate speed (-1..1).",
            valueSchema=number_schema(minimum=-1.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="oscillate",
            label="Oscillate",
            description="Function-channel oscillate intensity (0..1).",
            valueSchema=number_schema(minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="stop",
            label="Stop",
            description="When true, sendFunctionCmd stops output on selected device.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="stopOnDeactivate",
            label="Stop On Deactivate",
            description="Send stop command when service deactivates.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="connected",
            label="Connected",
            description="True when websocket is connected.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="scanning",
            label="Scanning",
            description="True while scanning is active.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="availableDevices",
            label="Available Devices",
            description="Device tokens for selection UI.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="deviceInfos",
            label="Device Infos",
            description="Full discovered device infos.",
            valueSchema=array_schema(items=_device_info_schema()),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="selectedDeviceInfo",
            label="Selected Device Info",
            description="Current selected device info object.",
            valueSchema=_device_info_schema(),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)

def register_operator(registry: Registry) -> Registry:
    registry.register_operator(ButtplugOutRuntimeNode.SPEC, ButtplugOutRuntimeNode, overwrite=True)
    return registry
