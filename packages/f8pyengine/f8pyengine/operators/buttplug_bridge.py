from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, cast

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    array_schema,
    boolean_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.json_unwrap import unwrap_json_value as _unwrap_json_value
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

logger = logging.getLogger(__name__)

OPERATOR_CLASS = "f8.buttplug_bridge"

_OUTPUT_VIBRATE = "Vibrate"
_OUTPUT_ROTATE = "Rotate"
_OUTPUT_OSCILLATE = "Oscillate"
_OUTPUT_POSITION = "Position"
_OUTPUT_POSITION_WITH_DURATION = "HwPositionWithDuration"
_POSITION_CLAMP_MIN = 0.0001
_POSITION_CLAMP_MAX = 0.9999


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
    def on_error(self, callback: Callable[[Exception], None] | Callable[[Exception], Awaitable[None]] | None) -> None: ...

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
class _BridgeConfig:
    enabled: bool
    ws_url: str
    auto_connect: bool
    auto_scan_on_connect: bool
    scan_duration_ms: int
    reconnect_interval_ms: int


def _now_ms() -> int:
    return int(time.time() * 1000.0)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    v = _unwrap_json_value(value)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    text = str(v or "").strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


def _coerce_int(value: Any, *, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    v = _unwrap_json_value(value)
    out: int
    if isinstance(v, bool):
        out = int(v)
    else:
        try:
            out = int(v)
        except (TypeError, ValueError):
            out = int(default)
    if minimum is not None and out < int(minimum):
        out = int(minimum)
    if maximum is not None and out > int(maximum):
        out = int(maximum)
    return int(out)


def _coerce_float(value: Any, *, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    v = _unwrap_json_value(value)
    out: float
    if isinstance(v, bool):
        out = float(default)
    else:
        try:
            out = float(v)
        except (TypeError, ValueError):
            out = float(default)
    if minimum is not None and out < float(minimum):
        out = float(minimum)
    if maximum is not None and out > float(maximum):
        out = float(maximum)
    return float(out)


def _coerce_str(value: Any, *, default: str) -> str:
    v = _unwrap_json_value(value)
    if v is None:
        return str(default)
    out = str(v).strip()
    if out:
        return out
    return str(default)


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


class ButtplugBridgeRuntimeNode(OperatorNode):
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
            return _coerce_bool(value, default=False)

        if name == "wsUrl":
            out = _coerce_str(value, default="ws://127.0.0.1:12345")
            if not (out.startswith("ws://") or out.startswith("wss://")):
                raise ValueError("wsUrl must start with ws:// or wss://")
            return out

        if name == "selectedDevice":
            return _coerce_str(value, default="")

        if name in ("scanDurationMs",):
            return _coerce_int(value, default=5000, minimum=100, maximum=120000)

        if name in ("reconnectIntervalMs",):
            return _coerce_int(value, default=2000, minimum=100, maximum=120000)

        if name in (
            "defaultPositionDurationMs",
            "vibrateFeatureIndex",
            "rotateFeatureIndex",
            "oscillateFeatureIndex",
            "positionFeatureIndex",
        ):
            if name == "defaultPositionDurationMs":
                return _coerce_int(value, default=500, minimum=0, maximum=120000)
            return _coerce_int(value, default=-1, minimum=-1, maximum=4096)

        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "rescan":
            if _coerce_bool(value, default=False):
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
            enabled = _coerce_bool(value, default=True)
            if not enabled:
                await self._disconnect_client()
            return

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        del in_port
        if not self._active:
            await self._emit_status_ports()
            return []
        await self._tick_once()

        target = await self._resolve_target_device(update_selection=True)
        if target is None:
            await self._emit_status_ports()
            return []

        stop_raw = await self.pull("stop", ctx_id=exec_id)
        stop_flag = _coerce_bool(stop_raw, default=False)
        if stop_flag:
            try:
                await target.stop(inputs=False, outputs=True)
                await self._mark_command_sent()
            except Exception as exc:
                await self._set_last_error_once("device_stop_failed", exc)
            await self._emit_status_ports()
            return []
        await self._dispatch_single_output(
            device=target,
            port_name="vibrate",
            output_name=_OUTPUT_VIBRATE,
            feature_state_name="vibrateFeatureIndex",
            minimum=0.0,
            maximum=1.0,
            ctx_id=exec_id,
            duration_ms=None,
        )
        await self._dispatch_single_output(
            device=target,
            port_name="rotate",
            output_name=_OUTPUT_ROTATE,
            feature_state_name="rotateFeatureIndex",
            minimum=-1.0,
            maximum=1.0,
            ctx_id=exec_id,
            duration_ms=None,
        )
        await self._dispatch_single_output(
            device=target,
            port_name="oscillate",
            output_name=_OUTPUT_OSCILLATE,
            feature_state_name="oscillateFeatureIndex",
            minimum=0.0,
            maximum=1.0,
            ctx_id=exec_id,
            duration_ms=None,
        )

        position_duration = await self._read_position_duration_ms(ctx_id=exec_id)
        await self._dispatch_single_output(
            device=target,
            port_name="position",
            output_name=_OUTPUT_POSITION_WITH_DURATION,
            feature_state_name="positionFeatureIndex",
            minimum=_POSITION_CLAMP_MIN,
            maximum=_POSITION_CLAMP_MAX,
            ctx_id=exec_id,
            duration_ms=position_duration,
            fallback_output_name=_OUTPUT_POSITION,
        )

        await self._emit_status_ports()
        return []

    def _start_worker(self) -> None:
        task = self._worker_task
        if task is not None and not task.done():
            return
        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._worker_loop(), name=f"buttplug_bridge:{self.node_id}")

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
            cfg = await self._read_bridge_config()
            await self._reconcile_connection(cfg)
            if self._rescan_requested:
                self._rescan_requested = False
                await self._run_scan_cycle(cfg)
            await self._publish_runtime_status()
            await self._publish_device_snapshot()

    async def _read_bridge_config(self) -> _BridgeConfig:
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
        return _BridgeConfig(
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
        client_obj = symbols.buttplug_client_cls("Feel8 Buttplug Bridge")
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

    async def _reconcile_connection(self, cfg: _BridgeConfig) -> None:
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
        except Exception as exc:
            await self._set_last_error_once("connect_failed", exc)

    async def _disconnect_client(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            if client.connected:
                await client.disconnect()
        except Exception as exc:
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

    async def _run_scan_cycle(self, cfg: _BridgeConfig) -> None:
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
        except Exception as exc:
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

    async def _dispatch_single_output(
        self,
        *,
        device: _DeviceLike,
        port_name: str,
        output_name: str,
        feature_state_name: str,
        minimum: float,
        maximum: float,
        ctx_id: str | int,
        duration_ms: int | None,
        fallback_output_name: str | None = None,
    ) -> None:
        raw = await self.pull(port_name, ctx_id=ctx_id)
        raw_unwrapped = _unwrap_json_value(raw)
        if raw_unwrapped is None:
            return

        value = _coerce_float(raw_unwrapped, default=0.0, minimum=minimum, maximum=maximum)
        feature_index = await self._read_int_state(feature_state_name, default=-1, minimum=-1, maximum=4096)

        selected_output_name = output_name
        if not device.has_output(output_name) and fallback_output_name is not None and device.has_output(fallback_output_name):
            selected_output_name = fallback_output_name

        if selected_output_name == output_name and not device.has_output(output_name):
            return

        try:
            command = self._build_output_command(
                output_name=selected_output_name,
                value=float(value),
                duration_ms=duration_ms if selected_output_name == _OUTPUT_POSITION_WITH_DURATION else None,
            )
        except Exception as exc:
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
                await self._set_last_error_message(
                    f"feature {feature_index} does not support {selected_output_name}"
                )
                return

            await feature.run_output(command)
            await self._mark_command_sent()
        except Exception as exc:
            await self._set_last_error_once(f"send_{selected_output_name}_failed", exc)

    async def _read_position_duration_ms(self, *, ctx_id: str | int) -> int:
        raw = await self.pull("positionDurationMs", ctx_id=ctx_id)
        raw_unwrapped = _unwrap_json_value(raw)
        if raw_unwrapped is not None:
            return _coerce_int(raw_unwrapped, default=500, minimum=0, maximum=120000)
        return await self._read_int_state("defaultPositionDurationMs", default=500, minimum=0, maximum=120000)

    async def _stop_target_device_outputs(self) -> None:
        target = await self._resolve_target_device(update_selection=False)
        if target is None:
            return
        try:
            await target.stop(inputs=False, outputs=True)
            await self._mark_command_sent()
        except Exception as exc:
            await self._set_last_error_once("stop_on_deactivate_failed", exc)

    async def _mark_command_sent(self) -> None:
        self._sent_commands = int(self._sent_commands) + 1
        self._last_command_ts_ms = _now_ms()
        await self._publish_state_if_changed("sentCommands", int(self._sent_commands))
        await self._publish_state_if_changed("lastCommandTsMs", int(self._last_command_ts_ms))

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
                    duration_range: list[int] | None = None
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

    async def _emit_status_ports(self) -> None:
        client = self._client
        connected = bool(client.connected) if client is not None else False
        await self.emit("connected", connected)

        selected_info = self._published_state_cache.get("selectedDeviceInfo")
        await self.emit("selectedDeviceInfo", selected_info)

        error_s = str(self._last_error_message or "")
        await self.emit("error", error_s)

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
            logger.exception("[%s:buttplug_bridge] %s", self.node_id, message, exc_info=exc)

    async def _set_last_error_message(self, message: str) -> None:
        self._last_error_message = str(message or "")
        await self._publish_state_if_changed("lastError", str(self._last_error_message))

    async def _clear_last_error(self) -> None:
        if not self._last_error_message:
            return
        self._last_error_message = ""
        await self._publish_state_if_changed("lastError", "")

    async def _read_raw_state(self, name: str) -> Any:
        live: Any
        try:
            live = await self.get_state_value(name)
        except Exception as exc:
            await self._set_last_error_once(f"read_state_{name}_failed", exc)
            live = None
        if live is not None:
            return _unwrap_json_value(live)
        return _unwrap_json_value(self._initial_state.get(name))

    async def _read_bool_state(self, name: str, *, default: bool) -> bool:
        return _coerce_bool(await self._read_raw_state(name), default=default)

    async def _read_int_state(self, name: str, *, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
        return _coerce_int(await self._read_raw_state(name), default=default, minimum=minimum, maximum=maximum)

    async def _read_str_state(self, name: str, *, default: str) -> str:
        return _coerce_str(await self._read_raw_state(name), default=default)

ButtplugBridgeRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Buttplug Bridge",
    description="Connect to Intiface/Buttplug, publish device capabilities, and drive selected device outputs.",
    tags=["io", "buttplug", "intiface", "haptics", "device"],
    execInPorts=["exec"],
    dataInPorts=[
        F8DataPortSpec(name="vibrate", description="Vibrate intensity (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="rotate", description="Rotate speed (-1..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="oscillate", description="Oscillate intensity (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="position", description="Position target (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(
            name="positionDurationMs",
            description="Optional position duration in milliseconds.",
            valueSchema=number_schema(default=500, minimum=0),
            required=False,
        ),
        F8DataPortSpec(
            name="stop",
            description="When true on exec, stop output on selected device.",
            valueSchema=boolean_schema(default=False),
            required=False,
        ),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="connected", description="Current connection status.", valueSchema=boolean_schema(default=False)),
        F8DataPortSpec(name="selectedDeviceInfo", description="Selected device info object.", valueSchema=any_schema()),
        F8DataPortSpec(name="error", description="Last error string.", valueSchema=string_schema(default="")),
    ],
    stateFields=[
        F8StateSpec(name="enabled", label="Enabled", description="Enable bridge connection and output control.", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, showOnNode=True),
        F8StateSpec(name="wsUrl", label="WebSocket URL", description="Buttplug server websocket URL.", valueSchema=string_schema(default="ws://127.0.0.1:12345"), access=F8StateAccess.rw, showOnNode=True),
        F8StateSpec(name="autoConnect", label="Auto Connect", description="Automatically connect while enabled.", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, showOnNode=True),
        F8StateSpec(name="autoScanOnConnect", label="Auto Scan On Connect", description="Start and stop scan once after connect.", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, showOnNode=False),
        F8StateSpec(name="scanDurationMs", label="Scan Duration (ms)", description="Scan duration before stop when scan is triggered.", valueSchema=integer_schema(default=5000, minimum=100, maximum=120000), access=F8StateAccess.rw, showOnNode=False),
        F8StateSpec(name="reconnectIntervalMs", label="Reconnect Interval (ms)", description="Reconnect throttle interval.", valueSchema=integer_schema(default=2000, minimum=100, maximum=120000), access=F8StateAccess.rw, showOnNode=False),
        F8StateSpec(name="selectedDevice", label="Selected Device", description="Target token: \"index|name\".", valueSchema=string_schema(default=""), access=F8StateAccess.rw, uiControl="select:[availableDevices]", showOnNode=True),
        F8StateSpec(name="rescan", label="Rescan", description="Set true to trigger one scan cycle; runtime resets it to false.", valueSchema=boolean_schema(default=False), access=F8StateAccess.rw, showOnNode=True),
        F8StateSpec(name="vibrateFeatureIndex", label="Vibrate Feature Index", description="Feature index for vibrate (-1 = all).", valueSchema=integer_schema(default=-1, minimum=-1, maximum=4096), access=F8StateAccess.rw, showOnNode=False),
        F8StateSpec(name="rotateFeatureIndex", label="Rotate Feature Index", description="Feature index for rotate (-1 = all).", valueSchema=integer_schema(default=-1, minimum=-1, maximum=4096), access=F8StateAccess.rw, showOnNode=False),
        F8StateSpec(name="oscillateFeatureIndex", label="Oscillate Feature Index", description="Feature index for oscillate (-1 = all).", valueSchema=integer_schema(default=-1, minimum=-1, maximum=4096), access=F8StateAccess.rw, showOnNode=False),
        F8StateSpec(name="positionFeatureIndex", label="Position Feature Index", description="Feature index for position (-1 = all).", valueSchema=integer_schema(default=-1, minimum=-1, maximum=4096), access=F8StateAccess.rw, showOnNode=False),
        F8StateSpec(name="defaultPositionDurationMs", label="Default Position Duration (ms)", description="Default duration for position output.", valueSchema=integer_schema(default=500, minimum=0, maximum=120000), access=F8StateAccess.rw, showOnNode=False),
        F8StateSpec(name="stopOnDeactivate", label="Stop On Deactivate", description="Send stop command when service deactivates.", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, showOnNode=False),
        F8StateSpec(name="connected", label="Connected", description="True when websocket is connected.", valueSchema=boolean_schema(default=False), access=F8StateAccess.ro, showOnNode=True, required=False),
        F8StateSpec(name="scanning", label="Scanning", description="True while scanning is active.", valueSchema=boolean_schema(default=False), access=F8StateAccess.ro, showOnNode=False, required=False),
        F8StateSpec(name="availableDevices", label="Available Devices", description="Device tokens for selection UI.", valueSchema=array_schema(items=string_schema()), access=F8StateAccess.ro, showOnNode=False, required=False),
        F8StateSpec(name="deviceInfos", label="Device Infos", description="Full discovered device infos.", valueSchema=any_schema(), access=F8StateAccess.ro, showOnNode=False, required=False),
        F8StateSpec(name="selectedDeviceInfo", label="Selected Device Info", description="Current selected device info object.", valueSchema=any_schema(), access=F8StateAccess.ro, showOnNode=True, required=False),
        F8StateSpec(name="lastError", label="Last Error", description="Last runtime error.", valueSchema=string_schema(default=""), access=F8StateAccess.ro, showOnNode=True, required=False),
        F8StateSpec(name="sentCommands", label="Sent Commands", description="Total sent output/stop commands.", valueSchema=integer_schema(default=0, minimum=0), access=F8StateAccess.ro, showOnNode=False, required=False),
        F8StateSpec(name="lastCommandTsMs", label="Last Command Ts (ms)", description="Timestamp of last sent command.", valueSchema=integer_schema(default=0, minimum=0), access=F8StateAccess.ro, showOnNode=False, required=False),
    ],
    editableStateFields=False,
    editableDataInPorts=False,
    editableDataOutPorts=False,
    editableExecInPorts=False,
    editableExecOutPorts=False,
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return ButtplugBridgeRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(ButtplugBridgeRuntimeNode.SPEC, overwrite=True)
    return reg
