from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import asyncio
import json
import logging
import socket
from dataclasses import dataclass
from typing import Any

from f8pysdk.codec import coerce_flag
from f8pysdk.specs import (
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
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.udp_out"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _UdpOutConfig:
    host: str
    port: int
    enabled: bool
    append_newline: bool
    force_text: bool


class UdpOutRuntimeNode(OperatorNode):
    """
    UDP output sink.

    On exec:
    - pulls `value`
    - converts it to bytes
    - sends one UDP datagram to (host, port)
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
        self._cfg: _UdpOutConfig | None = None
        self._cfg_dirty = True
        self._socket: socket.socket | None = None
        self._last_error: str | None = None

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        await self._ensure_socket()
        value = await self.pull("value", ctx_id=exec_id)
        if value is None:
            await self._emit_status(sent_bytes=0)
            return []

        data = self._to_bytes(value)
        if not data:
            await self._emit_status(sent_bytes=0)
            return []

        sent = await self._send(data)
        await self._emit_status(sent_bytes=sent)
        return []

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del value, ts_ms
        if str(field) in ("host", "port", "enabled", "appendNewline", "forceText"):
            self._cfg_dirty = True
            await self._ensure_socket(force_restart=True)

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        field_name = str(field or "").strip()
        if field_name == "host":
            host = str(value or "").strip()
            if not host:
                raise ValueError("host must not be empty")
            return host
        if field_name == "port":
            port = self._coerce_int(value, default=9000)
            if port <= 0 or port >= 65536:
                raise ValueError("port must be in range 1..65535")
            return port
        if field_name in ("enabled", "appendNewline", "forceText"):
            return coerce_flag(value, default=False if field_name == "appendNewline" else True)
        return value

    async def close(self) -> None:
        await self._close_socket()

    async def _read_cfg_from_state(self) -> _UdpOutConfig | None:
        host = await self.get_state_value("host")
        if host is None:
            host = self._initial_state.get("host", "127.0.0.1")
        port = await self.get_state_value("port")
        if port is None:
            port = self._initial_state.get("port", 9000)
        enabled = await self.get_state_value("enabled")
        if enabled is None:
            enabled = self._initial_state.get("enabled", True)
        append_newline = await self.get_state_value("appendNewline")
        if append_newline is None:
            append_newline = self._initial_state.get("appendNewline", False)
        force_text = await self.get_state_value("forceText")
        if force_text is None:
            force_text = self._initial_state.get("forceText", True)

        host_s = str(host or "").strip()
        if not host_s:
            self._last_error = "host must not be empty"
            return None
        port_i = self._coerce_int(port, default=9000)
        if port_i <= 0 or port_i >= 65536:
            self._last_error = f"Invalid port: {port_i}"
            return None

        return _UdpOutConfig(
            host=host_s,
            port=port_i,
            enabled=coerce_flag(enabled, default=True),
            append_newline=coerce_flag(append_newline, default=False),
            force_text=coerce_flag(force_text, default=True),
        )

    async def _ensure_socket(self, *, force_restart: bool = False) -> None:
        async with self._lock:
            prev_cfg = self._cfg
            if force_restart or self._cfg_dirty or self._cfg is None:
                cfg = await self._read_cfg_from_state()
                self._cfg_dirty = False
            else:
                cfg = self._cfg
            if cfg is None:
                await self._close_socket_locked()
                return
            if not cfg.enabled:
                await self._close_socket_locked()
                self._cfg = cfg
                await self._emit_status(sent_bytes=0)
                return
            if not force_restart and prev_cfg == cfg and self._socket is not None:
                return
            await self._close_socket_locked()
            self._cfg = cfg
            await self._open_socket_locked()
            await self._emit_status(sent_bytes=0)

    async def _open_socket_locked(self) -> None:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(True)
            self._last_error = None
        except OSError as exc:
            self._socket = None
            self._last_error = f"{type(exc).__name__}: {exc}"

    async def _close_socket(self) -> None:
        async with self._lock:
            await self._close_socket_locked()

    async def _close_socket_locked(self) -> None:
        sock = self._socket
        self._socket = None
        if sock is None:
            return
        try:
            await asyncio.to_thread(sock.close)
        except OSError as exc:
            logger.exception("[%s:udp_out] close socket failed: %s", self.node_id, exc)

    async def _send(self, data: bytes) -> int:
        async with self._lock:
            sock = self._socket
            cfg = self._cfg
            if sock is None or cfg is None or not cfg.enabled:
                return 0
            try:
                sent = await asyncio.to_thread(sock.sendto, data, (cfg.host, cfg.port))
                self._last_error = None
                return int(sent or 0)
            except OSError as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("[%s:udp_out] send failed host=%s port=%s", self.node_id, cfg.host, cfg.port)
                return 0

    async def _emit_status(self, *, sent_bytes: int) -> None:
        await self.emit("isOpen", bool(self._socket is not None))
        await self.emit("sentBytes", int(sent_bytes))
        await self.emit("error", str(self._last_error or ""))

    def _to_bytes(self, value: Any) -> bytes:
        cfg = self._cfg
        if cfg is None:
            return self._normalize_text_value(value).encode("utf-8", errors="replace")
        if cfg.force_text:
            normalized = self._normalize_text_value(value)
            if cfg.append_newline and not normalized.endswith("\n"):
                normalized = f"{normalized}\n"
            return normalized.encode("utf-8", errors="replace")

        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            if cfg.append_newline and not raw.endswith(b"\n"):
                raw = raw + b"\n"
            return raw

        if isinstance(value, str):
            normalized = value
            if cfg.append_newline and not normalized.endswith("\n"):
                normalized = f"{normalized}\n"
            return normalized.encode("utf-8", errors="replace")

        self._last_error = f"TypeError: forceText is disabled and value type {type(value).__name__} is not bytes or str"
        return b""

    @staticmethod
    def _normalize_text_value(value: Any) -> str:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value).decode("utf-8", errors="replace")

        if isinstance(value, dict):
            if "tcode" in value:
                value = value["tcode"]
            elif "value" in value:
                value = value["value"]

        if isinstance(value, str):
            stripped = value.strip()
            if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
                try:
                    decoded = json.loads(stripped)
                except json.JSONDecodeError:
                    return value
                if isinstance(decoded, str):
                    return decoded
            return value

        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        if value is None or isinstance(value, bool):
            return int(default)
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return int(default)


UdpOutRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.output",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="UDP Out",
    description="Sends incoming values to a UDP host/port.",
    tags=["io", "udp", "network", "socket", "tcode"],
    execInPorts=exec_port_specs(["exec"]),
    dataInPorts=[F8DataPortSpec(name="value", description="Value to send.", valueSchema=any_schema())],
    dataOutPorts=[
        F8DataPortSpec(name="isOpen", description="Whether the UDP socket is open.", valueSchema=boolean_schema(default=False)),
        F8DataPortSpec(name="sentBytes", description="Bytes sent by last exec.", valueSchema=integer_schema(default=0, minimum=0)),
        F8DataPortSpec(name="error", description="Last error (if any).", valueSchema=string_schema(default="")),
    ],
    stateFields=[
        F8StateSpec(
            name="enabled",
            label="Enabled",
            description="Enable/disable UDP output.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="host",
            label="Host",
            description="Target UDP host name or IP.",
            valueSchema=string_schema(default="127.0.0.1"),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="port",
            label="Port",
            description="Target UDP port.",
            valueSchema=integer_schema(default=9000, minimum=1, maximum=65535),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="appendNewline",
            label="Append Newline",
            description="Append a trailing newline to stringified values before sending.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="forceText",
            label="Force Text",
            description="When true, convert incoming values to text before sending. When false, only bytes and str are accepted.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(UdpOutRuntimeNode.SPEC, UdpOutRuntimeNode, overwrite=True)
    return registry
