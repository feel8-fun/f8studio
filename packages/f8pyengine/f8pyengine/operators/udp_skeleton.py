from __future__ import annotations

import asyncio
import base64
import json
import socket
import struct
from collections import deque
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
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.time_utils import now_ms

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.udp_skeleton"


@dataclass(frozen=True)
class _UdpConfig:
    bind_address: str
    port: int
    max_queue: int
    buffer_size: int
    emit_buffer_on_rx: bool
    reuse_address: bool


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


class UdpSkeletonRuntimeNode(RuntimeNode):
    """
    UDP skeleton receiver (poll-driven).

    - Starts a UDP listener (bindAddress/port)
    - Decodes incoming packets (skeleton binary -> dict, or JSON/utf-8, otherwise base64)
    - Stores latest payload and emits it onto data outputs
    - Passes exec through so you can place it in an exec chain (tick -> udp -> ...)
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._exec_out_ports = list(getattr(node, "execOutPorts", None) or []) or ["exec"]

        self._lock = asyncio.Lock()
        self._buf_lock = asyncio.Lock()
        self._cfg: _UdpConfig | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._queue: asyncio.Queue[tuple[int, bytes, tuple[str, int]]] | None = None
        self._dropped_ref: list[int] = [0]
        self._drain_task: asyncio.Task[object] | None = None

        self._packet_count = 0
        self._buffer: deque[dict[str, Any]] = deque(maxlen=512)
        self._last_payload: Any = None
        self._last_filtered_payload: Any = None
        self._last_source: dict[str, Any] | None = None
        self._last_rx_ts_ms: int | None = None
        self._last_error: str | None = None
        self._filter_names: set[str] = self._parse_filter_names(self._initial_state.get("filterModelNames", ""))

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        await self._ensure_receiver()
        await self._emit_snapshot()
        return list(self._exec_out_ports)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        if str(field) in ("bindAddress", "port", "maxQueue", "bufferSize", "emitBufferOnRx", "reuseAddress"):
            await self._ensure_receiver(force_restart=True)
            return
        if str(field) in ("filterModelNames",):
            self._filter_names = self._parse_filter_names(value)
            await self._emit_snapshot()

    async def close(self) -> None:
        await self._stop_receiver()

    def _desired_cfg(self) -> _UdpConfig | None:
        bind_address = self._initial_state.get("bindAddress", "0.0.0.0")
        port = self._initial_state.get("port", 39540)
        max_queue = self._initial_state.get("maxQueue", 512)

        # Best-effort pull current state (may be None if bus isn't attached yet).
        # This method is only called from async sites that can await; values are resolved there.
        return _UdpConfig(bind_address=str(bind_address), port=int(port), max_queue=int(max_queue))

    async def _read_cfg_from_state(self) -> _UdpConfig | None:
        bind_address = await self.get_state("bindAddress")
        if bind_address is None:
            bind_address = self._initial_state.get("bindAddress", "0.0.0.0")
        port = await self.get_state("port")
        if port is None:
            port = self._initial_state.get("port", 39540)
        max_queue = await self.get_state("maxQueue")
        if max_queue is None:
            max_queue = self._initial_state.get("maxQueue", 512)

        buffer_size = await self.get_state("bufferSize")
        if buffer_size is None:
            buffer_size = self._initial_state.get("bufferSize", 512)

        emit_buffer_on_rx = await self.get_state("emitBufferOnRx")
        if emit_buffer_on_rx is None:
            emit_buffer_on_rx = self._initial_state.get("emitBufferOnRx", False)

        reuse_address = await self.get_state("reuseAddress")
        if reuse_address is None:
            reuse_address = self._initial_state.get("reuseAddress", False)

        try:
            bind_address_s = str(bind_address).strip() or "0.0.0.0"
        except Exception:
            bind_address_s = "0.0.0.0"
        try:
            port_i = int(port)
        except Exception:
            port_i = 39540
        try:
            max_q = int(max_queue)
        except Exception:
            max_q = 512
        if port_i <= 0 or port_i >= 65536:
            self._last_error = f"Invalid port: {port_i}"
            return None
        max_q = max(1, min(4096, max_q))
        try:
            buf_n = int(buffer_size)
        except Exception:
            buf_n = 512
        buf_n = max(1, min(4096, buf_n))

        emit_buf = False
        if isinstance(emit_buffer_on_rx, bool):
            emit_buf = emit_buffer_on_rx
        elif isinstance(emit_buffer_on_rx, (int, float)):
            emit_buf = bool(emit_buffer_on_rx)
        else:
            emit_buf = str(emit_buffer_on_rx).strip().lower() in ("1", "true", "yes", "on")

        reuse_addr = False
        if isinstance(reuse_address, bool):
            reuse_addr = reuse_address
        elif isinstance(reuse_address, (int, float)):
            reuse_addr = bool(reuse_address)
        else:
            reuse_addr = str(reuse_address).strip().lower() in ("1", "true", "yes", "on")

        return _UdpConfig(
            bind_address=bind_address_s,
            port=port_i,
            max_queue=max_q,
            buffer_size=buf_n,
            emit_buffer_on_rx=emit_buf,
            reuse_address=reuse_addr,
        )

    async def _ensure_receiver(self, *, force_restart: bool = False) -> None:
        cfg = await self._read_cfg_from_state()
        async with self._lock:
            if cfg is None:
                await self._stop_receiver()
                await self._emit_status()
                return
            if not force_restart and self._cfg == cfg and self._transport is not None:
                return
            await self._stop_receiver()
            await self._start_receiver(cfg)

    async def _start_receiver(self, cfg: _UdpConfig) -> None:
        loop = asyncio.get_running_loop()
        self._cfg = cfg
        self._dropped_ref[0] = 0
        self._queue = asyncio.Queue(maxsize=int(cfg.max_queue))
        async with self._buf_lock:
            self._buffer = deque(maxlen=int(cfg.buffer_size))
            self._last_filtered_payload = None

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if cfg.reuse_address:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except OSError:
                    pass
                if hasattr(socket, "SO_REUSEPORT"):
                    try:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    except OSError:
                        pass
            sock.bind((cfg.bind_address, cfg.port))
            sock.setblocking(False)

            transport, _protocol = await loop.create_datagram_endpoint(
                lambda: _UdpProtocol(self._queue, self._dropped_ref),
                sock=sock,
            )
            self._transport = transport  # type: ignore[assignment]
            self._last_error = None
        except Exception as exc:
            try:
                sock.close()
            except Exception:
                pass
            self._transport = None
            self._queue = None
            self._last_error = f"{type(exc).__name__}: {exc}"
            await self._emit_status()
            return

        self._drain_task = asyncio.create_task(self._drain_loop(), name=f"udp_skeleton:{self.node_id}")
        await self._emit_status()

    async def _stop_receiver(self) -> None:
        t = self._drain_task
        self._drain_task = None
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass
            await asyncio.gather(t, return_exceptions=True)
        tr = self._transport
        self._transport = None
        if tr is not None:
            try:
                tr.close()
            except Exception:
                pass
        self._queue = None
        self._cfg = None

    async def _drain_loop(self) -> None:
        assert self._queue is not None
        q = self._queue
        while True:
            rx_ts_ms, raw, addr = await q.get()
            self._packet_count += 1
            self._last_rx_ts_ms = int(rx_ts_ms)
            self._last_source = {"host": addr[0], "port": addr[1]}
            payload = self._decode_payload(raw)
            self._last_payload = payload

            raw_b64 = base64.b64encode(raw).decode("ascii") if raw else ""
            model_name = self._extract_model_name(payload)
            entry = {
                "rxTsMs": int(rx_ts_ms),
                "source": {"host": addr[0], "port": addr[1]},
                "modelName": model_name,
                "payload": payload,
                "rawBase64": raw_b64,
            }
            async with self._buf_lock:
                self._buffer.append(entry)
            await self._emit_payload(raw, payload=payload, model_name=model_name)
            if not self._filter_names or (model_name is not None and model_name.strip().lower() in self._filter_names):
                self._last_filtered_payload = payload
                await self.emit("filteredPayload", payload)
            if bool(getattr(self._cfg, "emit_buffer_on_rx", False)):
                await self._emit_snapshot()

    async def _emit_payload(self, raw: bytes, *, payload: Any, model_name: str | None) -> None:
        raw_b64 = base64.b64encode(raw).decode("ascii") if raw else ""
        await self.emit("payload", payload)
        await self.emit("modelName", str(model_name or ""))
        await self.emit("source", self._last_source)
        await self.emit("rxTsMs", int(self._last_rx_ts_ms or now_ms()))
        await self.emit("rawBase64", raw_b64)
        await self._emit_status()

    async def _emit_snapshot(self) -> None:
        cfg = self._cfg
        async with self._buf_lock:
            items = list(self._buffer)
        names = set(self._filter_names or set())
        filtered = items
        if names:
            filtered = [x for x in items if str(x.get("modelName") or "").strip().lower() in names]
        latest_filtered_payload: Any = filtered[-1]["payload"] if filtered else None
        self._last_filtered_payload = latest_filtered_payload
        available = sorted(
            {str(x.get("modelName") or "").strip() for x in items if str(x.get("modelName") or "").strip()}
        )

        await self.emit("buffer", items)
        await self.emit("filteredBuffer", filtered)
        await self.emit("filteredPayload", latest_filtered_payload)
        await self.emit("availableModels", available)

        if cfg is not None:
            await self.emit("bindAddress", str(cfg.bind_address))
            await self.emit("port", int(cfg.port))

    async def _emit_status(self) -> None:
        queue_len = self._queue.qsize() if self._queue is not None else 0
        await self.emit("packetCount", int(self._packet_count))
        await self.emit("droppedCount", int(self._dropped_ref[0]))
        await self.emit("queueLen", int(queue_len))
        await self.emit("error", str(self._last_error or ""))

    @staticmethod
    def _parse_filter_names(raw: Any) -> set[str]:
        try:
            s = str(raw or "")
        except Exception:
            s = ""
        parts = [p.strip() for p in s.replace(";", ",").split(",")]
        return {p.lower() for p in parts if p}

    @staticmethod
    def _extract_model_name(payload: Any) -> str | None:
        if isinstance(payload, dict):
            for k in ("modelName", "name", "character", "actor"):
                v = payload.get(k)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    return s
        return None

    @staticmethod
    def _decode_payload(raw: bytes) -> Any:
        if not isinstance(raw, (bytes, bytearray)):
            return raw

        decoded_skeleton = UdpSkeletonRuntimeNode._decode_skeleton_packet(bytes(raw))
        if decoded_skeleton is not None:
            return decoded_skeleton

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"rawBase64": base64.b64encode(raw).decode("ascii")}
        if not text:
            return ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    @staticmethod
    def _read_aligned_string(buf: bytes, offset: int) -> tuple[str, int]:
        end = buf.find(b"\x00", offset)
        if end == -1:
            raise ValueError("Missing string terminator")
        value = buf[offset:end].decode("utf-8")
        end += 1
        pad = (4 - (end & 0x03)) & 0x03
        return value, end + pad

    @staticmethod
    def _decode_skeleton_packet(raw: bytes) -> dict[str, Any] | None:
        data = bytes(raw)
        offset = 0
        try:
            model_name, offset = UdpSkeletonRuntimeNode._read_aligned_string(data, offset)
            if offset + 8 > len(data):
                return None
            (timestamp_ms,) = struct.unpack_from("<Q", data, offset)
            offset += 8

            schema, offset = UdpSkeletonRuntimeNode._read_aligned_string(data, offset)
            if offset + 4 > len(data):
                return None
            (bone_count,) = struct.unpack_from("<i", data, offset)
            offset += 4
            if bone_count < 0 or bone_count > 4096:
                return None

            bones: list[dict[str, Any]] = []
            for _ in range(int(bone_count)):
                name, offset = UdpSkeletonRuntimeNode._read_aligned_string(data, offset)
                if offset + 7 * 4 > len(data):
                    return None
                x, y, z, qw, qx, qy, qz = struct.unpack_from("<fffffff", data, offset)
                offset += 7 * 4
                bones.append({"name": name, "pos": (x, y, z), "rot": (qw, qx, qy, qz)})

            return {"modelName": model_name, "timestampMs": int(timestamp_ms), "schema": schema, "bones": bones}
        except (struct.error, UnicodeDecodeError, ValueError):
            return None


UdpSkeletonRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="UDP Skeleton",
    description="Receives UDP packets and decodes Feel8 skeleton payloads (or JSON/utf-8).",
    tags=["io", "udp", "network", "skeleton", "mocap"],
    execInPorts=["exec"],
    execOutPorts=["exec"],
    dataOutPorts=[
        F8DataPortSpec(name="payload", description="Decoded payload (skeleton dict / JSON / text).", valueSchema=any_schema()),
        F8DataPortSpec(name="modelName", description="Extracted model/person name (if available).", valueSchema=string_schema(default="")),
        F8DataPortSpec(name="source", description="Sender address info.", valueSchema=any_schema()),
        F8DataPortSpec(name="rxTsMs", description="Receive timestamp (ms).", valueSchema=integer_schema(default=0, minimum=0)),
        F8DataPortSpec(name="rawBase64", description="Raw UDP payload (base64).", valueSchema=string_schema(default="")),
        F8DataPortSpec(name="buffer", description="Recent packet buffer (list of dicts).", valueSchema=any_schema()),
        F8DataPortSpec(name="filteredBuffer", description="Filtered buffer by model name.", valueSchema=any_schema()),
        F8DataPortSpec(name="filteredPayload", description="Latest payload matching filter.", valueSchema=any_schema()),
        F8DataPortSpec(name="availableModels", description="Unique model/person names in buffer.", valueSchema=any_schema()),
        F8DataPortSpec(name="bindAddress", description="Current bind address (echo).", valueSchema=string_schema(default="0.0.0.0")),
        F8DataPortSpec(name="port", description="Current bound port (echo).", valueSchema=integer_schema(default=0, minimum=0)),
        F8DataPortSpec(name="packetCount", description="Total received packets.", valueSchema=integer_schema(default=0, minimum=0)),
        F8DataPortSpec(name="droppedCount", description="Dropped packets due to full queue.", valueSchema=integer_schema(default=0, minimum=0)),
        F8DataPortSpec(name="queueLen", description="Current receive queue length.", valueSchema=integer_schema(default=0, minimum=0)),
        F8DataPortSpec(name="error", description="Last bind/receive error (if any).", valueSchema=string_schema(default="")),
    ],
    stateFields=[
        F8StateSpec(
            name="bindAddress",
            label="Bind Address",
            description="Local address to bind (use 0.0.0.0 for all).",
            valueSchema=string_schema(default="0.0.0.0"),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="port",
            label="Port",
            description="UDP listen port.",
            valueSchema=integer_schema(default=39540, minimum=1, maximum=65535),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="maxQueue",
            label="Max Queue",
            description="Max queued packets before dropping (1..4096).",
            valueSchema=integer_schema(default=512, minimum=1, maximum=4096),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="bufferSize",
            label="Buffer Size",
            description="How many recent packets to keep for `buffer` (1..4096).",
            valueSchema=integer_schema(default=512, minimum=1, maximum=4096),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="filterModelNames",
            label="Filter Model Names",
            description="Comma-separated model/person names to filter (empty means all).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="emitBufferOnRx",
            label="Emit Buffer On Receive",
            description="If enabled, also emits buffer snapshots on every received packet (can be heavy).",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="reuseAddress",
            label="Reuse Address",
            description="Best-effort: allow multiple listeners on same (address, port) if OS supports.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return UdpSkeletonRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(UdpSkeletonRuntimeNode.SPEC, overwrite=True)
    return reg
