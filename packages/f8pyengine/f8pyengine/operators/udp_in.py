from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import asyncio
import ipaddress
import json
import logging
import socket
from dataclasses import dataclass
from collections import deque
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
    complex_object_schema,
    integer_schema,
    string_schema,
)
from f8pysdk.capabilities import EntrypointNode, NodeBus
from f8pysdk.executors.exec_flow import EntrypointContext
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.time_utils import now_ms

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.udp_in"
logger = logging.getLogger(__name__)
_UDP_DRAIN_LOOP_ERRORS = (Exception,)
_UDP_EXEC_EMIT_ERRORS = (Exception,)
_UDP_EMIT_TASK_STOP_ERRORS = (Exception,)

_EXEC_PACKET_CACHE_MIN = 256


@dataclass(frozen=True)
class _UdpConfig:
    bind_address: str
    port: int
    max_queue: int
    reuse_address: bool


@dataclass(frozen=True)
class _PacketRecord:
    rx_ts_ms: int
    remote_address: str
    remote_port: int
    raw: bytes
    text: str
    json_value: Any | None
    json_valid: bool
    json_error: str


def _packet_payload_schema():
    return complex_object_schema(
        properties={
            "timestampMs": integer_schema(),
            "remoteAddress": string_schema(),
            "remotePort": integer_schema(),
            "byteLength": integer_schema(),
            "raw": any_schema(),
            "text": string_schema(),
            "json": any_schema(),
            "jsonValid": boolean_schema(),
        }
    )


class _UdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue[tuple[int, bytes, tuple[str, int]]], dropped_ref: list[int]) -> None:
        self._queue = queue
        self._dropped_ref = dropped_ref

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        item = (now_ms(), bytes(data), (str(addr[0]), int(addr[1])))
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._dropped_ref[0] += 1
            try:
                _ = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                pass


def _is_loopback_bind_address(value: str) -> bool:
    address = str(value or "").strip().lower()
    if not address:
        return False
    if address == "localhost":
        return True
    try:
        return bool(ipaddress.ip_address(address).is_loopback)
    except ValueError:
        return False


class UdpInRuntimeNode(OperatorNode, EntrypointNode):
    """
    Generic UDP input node.

    - listens on a local UDP port
    - keeps the latest packet
    - always exposes decoded text + raw bytearray
    - always exposes parsed JSON when the payload is valid UTF-8 JSON
    - emits exec on `packet` for incoming datagrams
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
            exec_in_ports=[str(p) for p in (node.execInPorts or [])],
            exec_out_ports=[str(p) for p in (node.execOutPorts or [])],
        )
        self._initial_state = dict(initial_state or {})

        self._lock = asyncio.Lock()
        self._packet_lock = asyncio.Lock()
        self._cfg: _UdpConfig | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._queue: asyncio.Queue[tuple[int, bytes, tuple[str, int]]] | None = None
        self._dropped_ref: list[int] = [0]
        self._drain_task: asyncio.Task[object] | None = None

        self._allow_non_loopback_bind = coerce_flag(
            self._initial_state.get("allowNonLoopbackBind"),
            default=False,
        )
        self._packet_count = 0
        self._last_error = ""
        self._published_listening: bool | None = None
        self._published_last_error: str | None = None
        self._latest_packet: _PacketRecord | None = None
        self._packet_by_ctx_id: dict[str, _PacketRecord] = {}
        self._packet_ctx_order: deque[str] = deque()
        self._packet_snapshot_limit = _EXEC_PACKET_CACHE_MIN
        self._entrypoint_ctx: EntrypointContext | None = None
        self._pending_exec_ids: deque[str | int] = deque()
        self._emit_wakeup = asyncio.Event()
        self._emit_task: asyncio.Task[None] | None = None
        self._emit_seq = 0

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        bus_like = bus if isinstance(bus, NodeBus) else None
        if bus_like is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.exception("udp_in attach failed: no running loop nodeId=%s", self.node_id)
            return
        if not bool(bus_like.active):
            loop.create_task(self._stop_receiver(), name=f"udp_in:deactivate:{self.node_id}")
            return
        loop.create_task(self._ensure_receiver(), name=f"udp_in:attach_start:{self.node_id}")

    async def on_lifecycle(self, active: bool, _meta: dict[str, Any]) -> None:
        if bool(active):
            await self._ensure_receiver()
        else:
            await self._stop_receiver()

    async def start_entrypoint(self, ctx: EntrypointContext) -> None:
        self._entrypoint_ctx = ctx

    async def stop_entrypoint(self) -> None:
        self._entrypoint_ctx = None
        await self._cancel_emit_task()

    def _bus_active(self) -> bool:
        bus = self._bus
        if bus is None:
            return True
        return bool(bus.active)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if not self._bus_active():
            await self._stop_receiver()
            return None

        port_name = str(port or "").strip()
        if port_name not in ("text", "raw", "json", "packet"):
            return None

        await self._ensure_receiver()
        async with self._packet_lock:
            packet = self._packet_for_ctx_locked(ctx_id)

        if packet is None:
            return None
        if port_name == "text":
            return packet.text
        if port_name == "raw":
            return bytearray(packet.raw)
        if port_name == "json":
            if not packet.json_valid:
                return None
            return packet.json_value
        return self._build_packet_payload(packet)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        field_name = str(field or "").strip()
        if field_name == "allowNonLoopbackBind":
            self._allow_non_loopback_bind = coerce_flag(value, default=False)
            if self._bus_active():
                await self._ensure_receiver(force_restart=True)
            return
        if field_name in ("bindAddress", "port", "maxQueue", "reuseAddress"):
            if self._bus_active():
                await self._ensure_receiver(force_restart=True)
            else:
                await self._stop_receiver()
            return

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        field_name = str(field or "").strip()
        if field_name == "allowNonLoopbackBind":
            return coerce_flag(value, default=False)
        if field_name == "bindAddress":
            bind_address = str(value or "").strip() or "127.0.0.1"
            if (not self._allow_non_loopback_bind) and (not _is_loopback_bind_address(bind_address)):
                raise ValueError("bindAddress must be loopback unless allowNonLoopbackBind is true")
            return bind_address
        if field_name == "port":
            port = self._coerce_int_or_default(value, default=39541)
            if port <= 0 or port >= 65536:
                raise ValueError("port must be in range 1..65535")
            return port
        if field_name == "maxQueue":
            max_queue = self._coerce_int_or_default(value, default=512)
            if max_queue <= 0 or max_queue > 4096:
                raise ValueError("maxQueue must be in range 1..4096")
            return max_queue
        if field_name == "reuseAddress":
            return coerce_flag(value, default=False)
        return value

    async def close(self) -> None:
        await self.stop_entrypoint()
        await self._stop_receiver()

    @staticmethod
    def _coerce_int_or_default(value: Any, *, default: int) -> int:
        if value is None or isinstance(value, bool):
            return int(default)
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return int(default)

    async def _read_cfg_from_state(self) -> _UdpConfig | None:
        bind_address = await self.get_state_value("bindAddress")
        if bind_address is None:
            bind_address = self._initial_state.get("bindAddress", "127.0.0.1")
        port = await self.get_state_value("port")
        if port is None:
            port = self._initial_state.get("port", 39541)
        max_queue = await self.get_state_value("maxQueue")
        if max_queue is None:
            max_queue = self._initial_state.get("maxQueue", 512)
        reuse_address = await self.get_state_value("reuseAddress")
        if reuse_address is None:
            reuse_address = self._initial_state.get("reuseAddress", False)

        bind_address_s = str(bind_address or "").strip() or "127.0.0.1"
        port_i = self._coerce_int_or_default(port, default=39541)
        max_queue_i = self._coerce_int_or_default(max_queue, default=512)
        if port_i <= 0 or port_i >= 65536:
            self._last_error = f"Invalid port: {port_i}"
            return None
        max_queue_i = max(1, min(4096, max_queue_i))
        if (not self._allow_non_loopback_bind) and (not _is_loopback_bind_address(bind_address_s)):
            self._last_error = "bindAddress must be loopback unless allowNonLoopbackBind is true"
            return None

        return _UdpConfig(
            bind_address=bind_address_s,
            port=port_i,
            max_queue=max_queue_i,
            reuse_address=coerce_flag(reuse_address, default=False),
        )

    async def _ensure_receiver(self, *, force_restart: bool = False) -> None:
        if not self._bus_active():
            await self._stop_receiver()
            return
        cfg = await self._read_cfg_from_state()
        async with self._lock:
            if cfg is None:
                await self._stop_receiver()
                await self._publish_runtime_state()
                return
            if not force_restart and self._cfg == cfg and self._transport is not None:
                return
            await self._stop_receiver()
            await self._start_receiver(cfg)

    async def _start_receiver(self, cfg: _UdpConfig) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[int, bytes, tuple[str, int]]] = asyncio.Queue(maxsize=int(cfg.max_queue))
        self._cfg = cfg
        self._queue = queue
        self._dropped_ref[0] = 0
        self._packet_snapshot_limit = max(_EXEC_PACKET_CACHE_MIN, int(cfg.max_queue) * 2)

        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if cfg.reuse_address:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except OSError:
                    pass
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except (AttributeError, OSError):
                    pass
            sock.bind((cfg.bind_address, cfg.port))
            sock.setblocking(False)

            transport, _protocol = await loop.create_datagram_endpoint(
                lambda: _UdpProtocol(queue, self._dropped_ref),
                sock=sock,
            )
            self._transport = transport  # type: ignore[assignment]
            self._last_error = ""
        except (OSError, RuntimeError) as exc:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            self._transport = None
            self._queue = None
            self._cfg = None
            self._last_error = f"{type(exc).__name__}: {exc}"
            await self._publish_runtime_state()
            return

        self._drain_task = asyncio.create_task(self._drain_loop(), name=f"udp_in:{self.node_id}")
        await self._publish_runtime_state()

    async def _stop_receiver(self) -> None:
        task = self._drain_task
        self._drain_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        transport = self._transport
        self._transport = None
        if transport is not None:
            transport.close()

        self._queue = None
        self._cfg = None
        async with self._packet_lock:
            self._packet_by_ctx_id.clear()
            self._packet_ctx_order.clear()
        await self._publish_runtime_state()

    async def _drain_loop(self) -> None:
        queue = self._queue
        if queue is None:
            return
        while True:
            try:
                rx_ts_ms, raw, addr = await queue.get()
                if not self._bus_active():
                    continue
                packet = self._decode_packet(rx_ts_ms=rx_ts_ms, raw=raw, addr=addr)
                self._packet_count += 1
                self._emit_seq += 1
                exec_id = int(self._emit_seq)
                async with self._packet_lock:
                    self._latest_packet = packet
                    self._store_packet_snapshot_locked(exec_id=exec_id, packet=packet)
                self._request_exec_emit(exec_id=exec_id)
            except asyncio.CancelledError:
                raise
            except _UDP_DRAIN_LOOP_ERRORS as exc:
                logger.exception("[%s:udp_in] drain loop failed", self.node_id, exc_info=exc)

    def _decode_packet(self, *, rx_ts_ms: int, raw: bytes, addr: tuple[str, int]) -> _PacketRecord:
        text = raw.decode("utf-8", errors="replace")
        json_value: Any | None = None
        json_valid = False
        json_error = ""
        stripped = text.strip()
        if stripped:
            try:
                json_value = json.loads(text)
                json_valid = True
            except json.JSONDecodeError as exc:
                json_error = f"{type(exc).__name__}: {exc.msg}"
        return _PacketRecord(
            rx_ts_ms=int(rx_ts_ms),
            remote_address=str(addr[0]),
            remote_port=int(addr[1]),
            raw=bytes(raw),
            text=text,
            json_value=json_value,
            json_valid=json_valid,
            json_error=json_error,
        )

    def _build_packet_payload(self, packet: _PacketRecord) -> dict[str, Any]:
        return {
            "timestampMs": int(packet.rx_ts_ms),
            "remoteAddress": packet.remote_address,
            "remotePort": int(packet.remote_port),
            "byteLength": len(packet.raw),
            "raw": bytearray(packet.raw),
            "text": packet.text,
            "json": packet.json_value if packet.json_valid else None,
            "jsonValid": bool(packet.json_valid),
        }

    @staticmethod
    def _ctx_key(ctx_id: str | int | None) -> str:
        return str(ctx_id) if ctx_id is not None else ""

    def _packet_for_ctx_locked(self, ctx_id: str | int | None) -> _PacketRecord | None:
        if ctx_id is not None:
            packet = self._packet_by_ctx_id.get(self._ctx_key(ctx_id))
            if packet is not None:
                return packet
        return self._latest_packet

    def _store_packet_snapshot_locked(self, *, exec_id: str | int, packet: _PacketRecord) -> None:
        ctx_key = self._ctx_key(exec_id)
        if not ctx_key:
            return
        if ctx_key not in self._packet_by_ctx_id:
            self._packet_ctx_order.append(ctx_key)
        self._packet_by_ctx_id[ctx_key] = packet
        while len(self._packet_ctx_order) > self._packet_snapshot_limit:
            oldest_key = self._packet_ctx_order.popleft()
            self._packet_by_ctx_id.pop(oldest_key, None)

    async def _publish_runtime_state(self) -> None:
        listening = bool(self._transport is not None)
        if self._published_listening != listening:
            await self.set_state("listening", listening)
            self._published_listening = listening

        last_error = str(self._last_error or "")
        if self._published_last_error != last_error:
            if last_error:
                await self.report_error(
                    "UDP_IN_ERROR",
                    last_error,
                    severity="error",
                    fingerprint=f"udp-in:{last_error}",
                )
            else:
                await self.clear_error()
            self._published_last_error = last_error

    def _request_exec_emit(self, *, exec_id: str | int) -> None:
        if self._entrypoint_ctx is None:
            return
        self._pending_exec_ids.append(exec_id)
        while len(self._pending_exec_ids) > self._packet_snapshot_limit:
            self._pending_exec_ids.popleft()
        self._emit_wakeup.set()
        task = self._emit_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._emit_task = loop.create_task(
            self._emit_exec_loop(),
            name=f"udp_in:emit_exec:{self.node_id}",
        )

    async def _emit_exec_loop(self) -> None:
        try:
            while True:
                await self._emit_wakeup.wait()
                self._emit_wakeup.clear()
                while self._pending_exec_ids:
                    exec_id = self._pending_exec_ids.popleft()
                    ctx = self._entrypoint_ctx
                    if ctx is None:
                        continue
                    try:
                        await ctx.emit_exec("packet", exec_id=exec_id)
                    except _UDP_EXEC_EMIT_ERRORS as exc:
                        logger.exception("[%s:udp_in] emit exec failed", self.node_id, exc_info=exc)
        except asyncio.CancelledError:
            raise
        finally:
            self._emit_task = None

    async def _cancel_emit_task(self) -> None:
        task = self._emit_task
        self._emit_task = None
        self._pending_exec_ids.clear()
        self._emit_wakeup.clear()
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        except _UDP_EMIT_TASK_STOP_ERRORS as exc:
            logger.exception("[%s:udp_in] stop emit task failed", self.node_id, exc_info=exc)

UdpInRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.input",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="UDP In",
    description="Receives UDP packets and exposes explicit raw/text/json views plus packet metadata.",
    tags=["io", "udp", "network", "input", "json", "bytes", "bytearray"],
    execInPorts=exec_port_specs([]),
    execOutPorts=exec_port_specs(["packet"]),
    dataOutPorts=[
        F8DataPortSpec(
            name="text",
            description="Latest packet decoded as UTF-8 text with replacement for invalid bytes.",
            valueSchema=string_schema(default=""),
        ),
        F8DataPortSpec(
            name="raw",
            description="Latest packet as bytearray, preserving non-ASCII bytes.",
            valueSchema=any_schema(),
        ),
        F8DataPortSpec(
            name="json",
            description="Latest packet parsed as JSON when valid; otherwise None.",
            valueSchema=any_schema(),
        ),
        F8DataPortSpec(
            name="packet",
            description="Latest packet metadata plus raw/text/json views.",
            valueSchema=_packet_payload_schema(),
        ),
    ],
    stateFields=[
        F8StateSpec(
            name="bindAddress",
            label="Bind Address",
            description="Local address to bind (loopback by default).",
            valueSchema=string_schema(default="127.0.0.1"),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="allowNonLoopbackBind",
            label="Allow Non-loopback Bind",
            description="When true, allow bindAddress values other than loopback.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="port",
            label="Port",
            description="UDP listen port.",
            valueSchema=integer_schema(default=39541, minimum=1, maximum=65535),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="maxQueue",
            label="Max Queue",
            description="Max queued packets before dropping (1..4096).",
            valueSchema=integer_schema(default=512, minimum=1, maximum=4096),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="reuseAddress",
            label="Reuse Address",
            description="Best-effort: allow multiple listeners on the same bind tuple if the OS supports it.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="listening",
            label="Listening",
            description="Readonly flag telling whether the UDP socket is active.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(UdpInRuntimeNode.SPEC, UdpInRuntimeNode, overwrite=True)
    return registry
