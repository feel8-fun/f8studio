from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    integer_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.serial_out"


@dataclass(frozen=True)
class _SerialConfig:
    port: str
    baudrate: int
    enabled: bool


class SerialOutRuntimeNode(OperatorNode):
    """
    Serial output sink.

    On exec:
    - pulls `value`
    - converts it to bytes
    - writes to serial port (pyserial)
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._lock = asyncio.Lock()
        self._cfg: _SerialConfig | None = None
        self._serial: Any = None
        self._last_error: str | None = None

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        await self._ensure_serial()
        value = await self.pull("value", ctx_id=exec_id)
        if value is None:
            await self._emit_status(written_bytes=0)
            return []
        # print("SerialOutRuntimeNode.on_exec: value =", value)
        data = self._to_bytes(value)
        if not data:
            await self._emit_status(written_bytes=0)
            return []
        written = await self._write(data)
        await self._emit_status(written_bytes=written)
        return []

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        if str(field) in ("port", "baudrate", "enabled"):
            await self._ensure_serial(force_restart=True)

    async def close(self) -> None:
        await self._close_serial()

    async def _read_cfg_from_state(self) -> _SerialConfig:
        port = await self.get_state_value("port")
        if port is None:
            port = self._initial_state.get("port", "COM3")
        baudrate = await self.get_state_value("baudrate")
        if baudrate is None:
            baudrate = self._initial_state.get("baudrate", 115200)
        enabled = await self.get_state_value("enabled")
        if enabled is None:
            enabled = self._initial_state.get("enabled", True)

        port_s = str(port).strip()
        if not port_s:
            port_s = "COM3"
        try:
            baud_i = int(baudrate)
        except Exception:
            baud_i = 115200
        baud_i = max(300, min(4000000, baud_i))

        enabled_b = False
        if isinstance(enabled, bool):
            enabled_b = enabled
        elif isinstance(enabled, (int, float)):
            enabled_b = bool(enabled)
        else:
            enabled_b = str(enabled).strip().lower() in ("1", "true", "yes", "on")

        return _SerialConfig(
            port=port_s,
            baudrate=baud_i,
            enabled=enabled_b,
        )

    async def _ensure_serial(self, *, force_restart: bool = False) -> None:
        cfg = await self._read_cfg_from_state()
        async with self._lock:
            if not cfg.enabled:
                await self._close_serial_locked()
                self._cfg = cfg
                await self._emit_status(written_bytes=0)
                return
            if not force_restart and self._cfg == cfg and self._serial is not None:
                return
            await self._close_serial_locked()
            self._cfg = cfg
            await self._open_serial_locked(cfg)
            await self._emit_status(written_bytes=0)

    async def _open_serial_locked(self, cfg: _SerialConfig) -> None:
        try:
            import serial  # type: ignore[import-not-found]
        except Exception as exc:
            self._last_error = f"pyserial not available: {exc}"
            self._serial = None
            return
        try:
            self._serial = await asyncio.to_thread(
                serial.Serial,
                port=cfg.port,
                baudrate=int(cfg.baudrate),
                timeout=0,
                # Use a small write timeout (blocking) to avoid non-blocking partial writes on some platforms/drivers.
                write_timeout=0.1,
            )
            self._last_error = None
        except Exception as exc:
            self._serial = None
            self._last_error = f"{type(exc).__name__}: {exc}"

    async def _close_serial(self) -> None:
        async with self._lock:
            await self._close_serial_locked()

    async def _close_serial_locked(self) -> None:
        s = self._serial
        self._serial = None
        if s is None:
            return
        try:
            await asyncio.to_thread(s.close)
        except Exception:
            pass

    async def _write(self, data: bytes) -> int:
        async with self._lock:
            s = self._serial
            if s is None:
                return 0
            try:
                n = await asyncio.to_thread(s.write, data)
                try:
                    await asyncio.to_thread(s.flush)
                except Exception:
                    pass
                return int(n or 0)
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                return 0

    async def _emit_status(self, *, written_bytes: int) -> None:
        await self.emit("isOpen", bool(self._serial is not None))
        await self.emit("writtenBytes", int(written_bytes))
        await self.emit("error", str(self._last_error or ""))

    def _to_bytes(self, value: Any) -> bytes:
        # TCode devices expect ASCII. Keep encoding fixed to remove extra UI/edge cases.
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)

        # Common interop case: upstream may send a dict envelope; unwrap if possible.
        if isinstance(value, dict):
            if "tcode" in value:
                value = value.get("tcode")
            elif "value" in value:
                value = value.get("value")
        if isinstance(value, str):
            s = value
            # If we received a JSON-encoded string (e.g. "\"L09999I020\""),
            # unwrap it so devices don't see surrounding quotes/escapes.
            ss = s.strip()
            if len(ss) >= 2 and ss[0] == '"' and ss[-1] == '"':
                try:
                    decoded = json.loads(ss)
                    if isinstance(decoded, str):
                        s = decoded
                except Exception:
                    pass
        else:
            try:
                s = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                s = str(value)
        try:
            return s.encode("ascii", errors="replace")
        except Exception:
            return s.encode("ascii", errors="replace")


SerialOutRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Serial Out",
    description="Writes incoming values to a serial port (pyserial).",
    tags=["io", "serial", "uart", "com"],
    execInPorts=["exec"],
    dataInPorts=[F8DataPortSpec(name="value", description="Value to write.", valueSchema=any_schema())],
    dataOutPorts=[
        F8DataPortSpec(name="isOpen", description="Whether serial port is open.", valueSchema=boolean_schema(default=False)),
        F8DataPortSpec(name="writtenBytes", description="Bytes written by last exec.", valueSchema=integer_schema(default=0, minimum=0)),
        F8DataPortSpec(name="error", description="Last error (if any).", valueSchema=string_schema(default="")),
    ],
    stateFields=[
        F8StateSpec(
            name="enabled",
            label="Enabled",
            description="Enable/disable serial output.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="port",
            label="Port",
            description="Serial port name (e.g., COM3).",
            valueSchema=string_schema(default="COM4"),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="baudrate",
            label="Baudrate",
            description="Serial baud rate.",
            valueSchema=integer_schema(default=115200, minimum=300, maximum=4000000),
            access=F8StateAccess.wo,
            showOnNode=False,
        ),
        # Back-compat for saved sessions: these fields used to exist and may appear
        # in serialized node properties. They are intentionally ignored by runtime.
        F8StateSpec(
            name="encoding",
            label="Encoding (deprecated)",
            description="Deprecated. Serial Out always encodes as ASCII for TCode devices.",
            valueSchema=string_schema(default="ascii"),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="newline",
            label="Newline (deprecated)",
            description="Deprecated. Provide newline in the input payload if needed.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return SerialOutRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(SerialOutRuntimeNode.SPEC, overwrite=True)
    return reg

