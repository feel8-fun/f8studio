from __future__ import annotations

import asyncio
import json
import socket
import struct
from dataclasses import dataclass
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    any_schema,
    boolean_schema,
    integer_schema,
    string_schema,
)
from f8pysdk.capabilities import NodeBus
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.time_utils import now_ms

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

OPERATOR_CLASS = "f8.udp_skeleton"


@dataclass(frozen=True)
class _UdpConfig:
    bind_address: str
    port: int
    max_queue: int
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
    UDP skeleton receiver.

    - Starts a UDP listener (bindAddress/port)
    - Decodes incoming packets (skeleton binary -> dict, or JSON/utf-8, otherwise base64)
    - Maintains a per-model table (key -> latest skeleton + rxTsMs)
    - Removes stale models older than `cleanupAfterMs`
    - Emits `skeletons` (all latest), `selectedSkeleton`, and `availableKeys`
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
        self._exec_out_ports = exec_out_ports(node, default=["exec"])

        self._lock = asyncio.Lock()
        self._models_lock = asyncio.Lock()
        self._cfg: _UdpConfig | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._queue: asyncio.Queue[tuple[int, bytes, tuple[str, int]]] | None = None
        self._dropped_ref: list[int] = [0]
        self._drain_task: asyncio.Task[object] | None = None

        self._packet_count = 0
        self._last_error: str | None = None

        self._cleanup_after_ms = self._parse_int(self._initial_state.get("cleanupAfterMs", 10000), default=10000)
        self._selected_key = str(self._initial_state.get("selectedKey", "") or "")

        # key -> {"rxTsMs": int, "payload": Any}
        self._skeletons_by_key: dict[str, dict[str, Any]] = {}
        self._skeletons_version = 0
        self._last_emitted_version = -1
        self._last_emitted_keys: list[str] = []
        self._last_emitted_selected: tuple[str, int] | None = None  # (key, rxTsMs)

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        # Apply current active state immediately (best-effort).
        bus_like = bus if isinstance(bus, NodeBus) else None
        if bus_like is not None:
            try:
                if not bool(bus_like.active):
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._stop_receiver(), name=f"udp_skeleton:deactivate:{self.node_id}")
            except Exception:
                pass

    async def on_lifecycle(self, active: bool, _meta: dict[str, Any]) -> None:
        if bool(active):
            await self._ensure_receiver()
        else:
            await self._stop_receiver()

    def _bus_active(self) -> bool:
        bus = self._bus
        if bus is None:
            return True
        try:
            return bool(bus.active)
        except Exception:
            return True

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        if not self._bus_active():
            await self._stop_receiver()
            return list(self._exec_out_ports)
        await self._ensure_receiver()
        await self._cleanup_stale()
        await self._emit_updates(force=True)
        return list(self._exec_out_ports)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if not self._bus_active():
            await self._stop_receiver()
            return None
        p = str(port or "").strip()
        if p not in ("skeletons", "selectedSkeleton"):
            return None

        await self._ensure_receiver()
        await self._cleanup_stale()

        async with self._models_lock:
            keys = sorted(self._skeletons_by_key.keys())
            if p == "skeletons":
                return [self._skeletons_by_key[k].get("payload") for k in keys]
            selected_key = str(self._selected_key or "").strip()
            selected = self._skeletons_by_key.get(selected_key) if selected_key else None
            return None if selected is None else selected.get("payload")

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        f = str(field)
        if f in ("bindAddress", "port", "maxQueue", "reuseAddress"):
            if self._bus_active():
                await self._ensure_receiver(force_restart=True)
            else:
                await self._stop_receiver()
            return
        if f == "cleanupAfterMs":
            self._cleanup_after_ms = self._parse_int(value, default=self._cleanup_after_ms)
            await self._cleanup_stale()
            await self._emit_updates(force=True)
            return
        if f == "selectedKey":
            self._selected_key = str(value or "")
            await self._emit_updates(force=True)
            return

    async def close(self) -> None:
        await self._stop_receiver()

    @staticmethod
    def _parse_int(value: Any, *, default: int) -> int:
        if value is None:
            return int(default)
        if isinstance(value, bool):
            return int(default)
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return int(default)

    async def _read_cfg_from_state(self) -> _UdpConfig | None:
        bind_address = await self.get_state_value("bindAddress")
        if bind_address is None:
            bind_address = self._initial_state.get("bindAddress", "0.0.0.0")
        port = await self.get_state_value("port")
        if port is None:
            port = self._initial_state.get("port", 39540)
        max_queue = await self.get_state_value("maxQueue")
        if max_queue is None:
            max_queue = self._initial_state.get("maxQueue", 512)

        reuse_address = await self.get_state_value("reuseAddress")
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
            reuse_address=reuse_addr,
        )

    async def _ensure_receiver(self, *, force_restart: bool = False) -> None:
        if not self._bus_active():
            await self._stop_receiver()
            return
        cfg = await self._read_cfg_from_state()
        async with self._lock:
            if cfg is None:
                await self._stop_receiver()
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

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if cfg.reuse_address:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                except OSError:
                    pass
                try:
                    reuseport = socket.SO_REUSEPORT
                except Exception:
                    reuseport = None
                if reuseport is not None:
                    try:
                        sock.setsockopt(socket.SOL_SOCKET, reuseport, 1)
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
            return

        self._drain_task = asyncio.create_task(self._drain_loop(), name=f"udp_skeleton:{self.node_id}")

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
            if not self._bus_active():
                continue
            self._packet_count += 1
            payload = self._decode_payload(raw)
            model_name = self._extract_model_name(payload)
            key = str(model_name or "").strip()
            if not key:
                key = f"{addr[0]}:{addr[1]}"

            entry = {
                "rxTsMs": int(rx_ts_ms),
                "payload": payload,
            }

            async with self._models_lock:
                self._skeletons_by_key[key] = entry
                self._skeletons_version += 1

            # Do not emit on packet arrival: keep this node tick/pull-driven.
            # Cleanup/emission happens in `on_exec` / `compute_output`.

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
            return {"rawBytesLen": len(raw)}
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

    async def _cleanup_stale(self, *, now_ts_ms: int | None = None) -> None:
        ttl_ms = int(self._cleanup_after_ms)
        if ttl_ms <= 0:
            return
        if now_ts_ms is None:
            now_ts_ms = int(now_ms())

        cutoff = int(now_ts_ms) - ttl_ms
        removed = False
        async with self._models_lock:
            for k, v in list(self._skeletons_by_key.items()):
                try:
                    rx = int(v.get("rxTsMs") or 0)
                except Exception:
                    rx = 0
                if rx and rx < cutoff:
                    self._skeletons_by_key.pop(k, None)
                    removed = True
            if removed:
                self._skeletons_version += 1

        if removed:
            await self._emit_updates(force=True)

    async def _emit_updates(self, *, force: bool = False) -> None:
        async with self._models_lock:
            version = int(self._skeletons_version)
            keys = sorted(self._skeletons_by_key.keys())
            skeletons = [self._skeletons_by_key[k].get("payload") for k in keys]

            selected_key = str(self._selected_key or "").strip()
            selected = self._skeletons_by_key.get(selected_key) if selected_key else None

        if keys != self._last_emitted_keys:
            self._last_emitted_keys = list(keys)
            await self.set_state("availableKeys", list(keys))

        if force or version != self._last_emitted_version:
            self._last_emitted_version = int(version)
            await self.emit("skeletons", skeletons)

        if not selected_key:
            if force or self._last_emitted_selected is not None:
                self._last_emitted_selected = None
                await self.emit("selectedSkeleton", None)
            return

        if selected is None:
            if force or self._last_emitted_selected is not None:
                self._last_emitted_selected = None
                await self.emit("selectedSkeleton", None)
            return

        try:
            rx_ts = int(selected.get("rxTsMs") or 0)
        except Exception:
            rx_ts = 0
        marker = (selected_key, rx_ts)
        if force or marker != self._last_emitted_selected:
            self._last_emitted_selected = marker
            await self.emit("selectedSkeleton", selected.get("payload"))


UdpSkeletonRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="UDP Skeleton",
    description="Receives UDP packets and keeps latest skeleton per model key, with TTL cleanup and selection.",
    tags=["io", "udp", "network", "skeleton", "mocap"],
    execInPorts=["exec"],
    execOutPorts=["exec"],
    dataOutPorts=[
        F8DataPortSpec(
            name="skeletons", description="List of latest payloads (ordered by key).", valueSchema=any_schema()
        ),
        F8DataPortSpec(
            name="selectedSkeleton",
            description="Latest payload matching `selectedKey` (or None).",
            valueSchema=any_schema(),
        ),
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
            name="reuseAddress",
            label="Reuse Address",
            description="Best-effort: allow multiple listeners on same (address, port) if OS supports.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="cleanupAfterMs",
            label="Cleanup After (ms)",
            description="Remove models that haven't updated for this many ms (<=0 disables cleanup).",
            valueSchema=integer_schema(default=10000, minimum=0, maximum=60_000_000),
            access=F8StateAccess.wo,
            showOnNode=True,
        ),
        F8StateSpec(
            name="selectedKey",
            label="Selected Key",
            description="If set and matches an available key, outputs `selectedSkeleton`; otherwise None.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.wo,
            showOnNode=True,
        ),
        F8StateSpec(
            name="availableKeys",
            label="Available Keys",
            description="Read-only list of current keys (updated only on changes).",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
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
