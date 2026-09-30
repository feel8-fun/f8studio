from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec

from ...capabilities import ComputableNode, DataReceivableNode
from ...data import CrossPublishPolicy, DataDeliveryMode
from ...generated import F8Edge, F8EdgeStrategyEnum
from ...f8_naming import data_key
from ...time_utils import now_ms
from ...codec import decode_obj, dump_json, encode_obj
from ..internal.cache import CappedOrderedDict
from ..internal.logging import log_error_once
from .emit import CrossPublishPlan, DataEmitOptions

if TYPE_CHECKING:
    from ..runtime import ServiceBus


log = logging.getLogger(__name__)
_DATA_NODE_CALLBACK_ERRORS = (Exception,)
_DATA_ROUTER_LIFECYCLE_ERRORS = (OSError, RuntimeError, TypeError, ValueError)
_DATA_ROUTER_SCHEDULE_ERRORS = (RuntimeError, TypeError, ValueError)
_DATA_TRANSPORT_SUBSCRIPTION_ERRORS = (OSError, RuntimeError, TypeError, ValueError)

DataRouteTarget = tuple[str, str, F8Edge]
DataOutRoutes = dict[tuple[str, str], tuple[DataRouteTarget, ...]]
DataCrossInRoutes = dict[str, tuple[DataRouteTarget, ...]]
DataCrossOutRoutes = dict[tuple[str, str], str]
DataInputStreamRoutes = dict[tuple[str, str], str]
DEFAULT_DATA_EMIT_OPTIONS = DataEmitOptions()
LOCAL_COMPUTE_DATA_EMIT_OPTIONS = DataEmitOptions.local_compute_only()


def _payload_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "json_object"
    if isinstance(value, list):
        return "json_array"
    if isinstance(value, (str, int, float, bool)):
        return "json_scalar"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes"
    return type(value).__name__


def _optional_runtime_text(value: Any) -> str:
    if value is None or isinstance(value, msgspec.UnsetType):
        return ""
    return str(value)


@dataclass
class InputBuffer:
    to_node: str
    to_port: str
    edge: F8Edge | None
    queue: deque[tuple[Any, int]] = field(default_factory=deque)
    last_seen_value: Any = None
    last_seen_ts: int | None = None
    last_seen_ctx_id: str | int | None = None
    last_pulled_value: Any = None
    last_pulled_ts: int | None = None
    last_pulled_ctx_id: str | int | None = None


@dataclass
class OutputSnapshot:
    from_node: str
    from_port: str
    value: Any
    ts_ms: int
    ctx_id: str | int | None
    delivered_local: bool
    cross_publish_decision: str
    cross_publish_key: str


@dataclass
class OutputBuffer:
    from_node: str
    from_port: str
    snapshots: deque[OutputSnapshot]


class DataRouter:
    def __init__(
        self,
        bus: "ServiceBus",
        *,
        cross_publish_policy: CrossPublishPolicy,
        data_delivery: DataDeliveryMode,
        input_max_buffers: int,
        default_queue_size: int,
        output_debug_max_ports: int,
        output_debug_history_size: int,
    ) -> None:
        self._bus = bus
        self._cross_publish_policy: CrossPublishPolicy = cross_publish_policy
        self._data_delivery: DataDeliveryMode = data_delivery
        self._default_queue_size = max(1, int(default_queue_size))
        self._output_history_size = max(1, min(int(output_debug_history_size), 128))
        self._intra_data_out: DataOutRoutes = {}
        self._intra_data_in: DataOutRoutes = {}
        self._cross_in_by_key: DataCrossInRoutes = {}
        self._cross_out_keys: DataCrossOutRoutes = {}
        self._input_stream_keys: DataInputStreamRoutes = {}
        self._inputs: CappedOrderedDict[tuple[str, str], InputBuffer] = CappedOrderedDict(
            max_entries=max(0, int(input_max_buffers))
        )
        self._outputs: CappedOrderedDict[tuple[str, str], OutputBuffer] = CappedOrderedDict(
            max_entries=max(0, int(output_debug_max_ports))
        )
        self._route_subscriptions: dict[str, Any] = {}
        self._custom_subscriptions: list[Any] = []
        self._on_data_push_queue: deque[tuple[str, str, Any, int]] = deque()
        self._on_data_flush_task: asyncio.Task[None] | None = None

    @property
    def cross_publish_policy(self) -> CrossPublishPolicy:
        return self._cross_publish_policy

    @property
    def data_delivery(self) -> DataDeliveryMode:
        return self._data_delivery

    @property
    def input_buffers(self) -> CappedOrderedDict[tuple[str, str], InputBuffer]:
        return self._inputs

    @property
    def intra_data_out(self) -> DataOutRoutes:
        return self._intra_data_out

    @property
    def intra_data_in(self) -> DataOutRoutes:
        return self._intra_data_in

    @property
    def cross_in_by_key(self) -> DataCrossInRoutes:
        return self._cross_in_by_key

    @property
    def cross_out_keys(self) -> DataCrossOutRoutes:
        return self._cross_out_keys

    def input_stream_key(self, *, node_id: str, port: str) -> str | None:
        key = self._input_stream_keys.get((str(node_id), str(port)))
        if not key:
            return None
        return str(key)

    def debug_input_buffers(
        self,
        *,
        node_id: str = "",
        port: str = "",
        limit: int = 100,
        include_value: bool = True,
        max_value_bytes: int = 65536,
    ) -> dict[str, Any]:
        node_filter = str(node_id or "").strip()
        port_filter = str(port or "").strip()
        item_limit = max(1, min(int(limit), 1000))
        entries: list[dict[str, Any]] = []
        matched = 0
        for (buffer_node_id, buffer_port), buffer in list(self._inputs.items()):
            if node_filter and buffer_node_id != node_filter:
                continue
            if port_filter and buffer_port != port_filter:
                continue
            matched += 1
            if item_limit > 0 and len(entries) >= item_limit:
                continue
            entries.append(
                self._debug_input_buffer_entry(
                    buffer,
                    include_value=bool(include_value),
                    max_value_bytes=max(0, int(max_value_bytes)),
                )
            )
        return {
            "serviceId": str(self._bus.service_id),
            "active": bool(self._bus.active),
            "nodeId": node_filter,
            "port": port_filter,
            "limit": item_limit,
            "matched": matched,
            "truncated": item_limit > 0 and matched > item_limit,
            "buffers": entries,
            "outputMatched": self._count_debug_outputs(node_id=node_filter, port=port_filter),
            "outputs": self.debug_output_snapshots(
                node_id=node_filter,
                port=port_filter,
                limit=item_limit,
                include_value=bool(include_value),
                max_value_bytes=max(0, int(max_value_bytes)),
            ),
        }

    def _debug_input_buffer_entry(
        self,
        buffer: InputBuffer,
        *,
        include_value: bool,
        max_value_bytes: int,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "nodeId": str(buffer.to_node),
            "port": str(buffer.to_port),
            "queueLength": len(buffer.queue),
            "lastSeenTs": buffer.last_seen_ts,
            "lastPulledTs": buffer.last_pulled_ts,
            "lastSeenCtxId": buffer.last_seen_ctx_id,
            "lastPulledCtxId": buffer.last_pulled_ctx_id,
            "hasLastSeenValue": buffer.last_seen_value is not None,
            "hasLastPulledValue": buffer.last_pulled_value is not None,
        }
        if buffer.edge is not None:
            entry["edgeId"] = str(buffer.edge.edgeId)
            entry["fromServiceId"] = str(buffer.edge.fromServiceId)
            entry["fromOperatorId"] = _optional_runtime_text(buffer.edge.fromOperatorId)
            entry["fromPort"] = str(buffer.edge.fromPort)
        self._attach_debug_value(
            entry,
            key_prefix="lastSeen",
            value=buffer.last_seen_value,
            include_value=include_value,
            max_value_bytes=max_value_bytes,
        )
        self._attach_debug_value(
            entry,
            key_prefix="lastPulled",
            value=buffer.last_pulled_value,
            include_value=include_value,
            max_value_bytes=max_value_bytes,
        )
        return entry

    def debug_output_snapshots(
        self,
        *,
        node_id: str = "",
        port: str = "",
        limit: int = 100,
        include_value: bool = True,
        max_value_bytes: int = 65536,
    ) -> list[dict[str, Any]]:
        node_filter = str(node_id or "").strip()
        port_filter = str(port or "").strip()
        item_limit = max(1, min(int(limit), 1000))
        entries: list[dict[str, Any]] = []
        indexed_entries: list[tuple[int, int, dict[str, Any]]] = []
        sequence_index = 0
        for (buffer_node_id, buffer_port), buffer in list(self._outputs.items()):
            if node_filter and buffer_node_id != node_filter:
                continue
            if port_filter and buffer_port != port_filter:
                continue
            for snapshot in list(buffer.snapshots):
                indexed_entries.append(
                    (
                        int(snapshot.ts_ms),
                        sequence_index,
                        self._debug_output_snapshot_entry(
                            snapshot,
                            include_value=bool(include_value),
                            max_value_bytes=max(0, int(max_value_bytes)),
                        ),
                    )
                )
                sequence_index += 1
        indexed_entries.sort(key=lambda item: (item[0], item[1]))
        for _, _, entry in indexed_entries[-item_limit:]:
            entries.append(entry)
        return entries

    def _count_debug_outputs(self, *, node_id: str, port: str) -> int:
        count = 0
        for buffer_node_id, buffer_port in list(self._outputs.keys()):
            if node_id and buffer_node_id != node_id:
                continue
            if port and buffer_port != port:
                continue
            buffer = self._outputs.get((buffer_node_id, buffer_port))
            if buffer is not None:
                count += len(buffer.snapshots)
        return count

    def _debug_output_snapshot_entry(
        self,
        snapshot: OutputSnapshot,
        *,
        include_value: bool,
        max_value_bytes: int,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "nodeId": str(snapshot.from_node),
            "port": str(snapshot.from_port),
            "tsMs": int(snapshot.ts_ms),
            "ctxId": snapshot.ctx_id,
            "deliveredLocal": bool(snapshot.delivered_local),
            "crossPublishDecision": str(snapshot.cross_publish_decision),
            "crossPublishKey": str(snapshot.cross_publish_key),
        }
        self._attach_debug_value(
            entry,
            key_prefix="lastEmitted",
            value=snapshot.value,
            include_value=include_value,
            max_value_bytes=max_value_bytes,
        )
        return entry

    @staticmethod
    def _attach_debug_value(
        entry: dict[str, Any],
        *,
        key_prefix: str,
        value: Any,
        include_value: bool,
        max_value_bytes: int,
    ) -> None:
        if value is None:
            entry[f"{key_prefix}PayloadKind"] = "null"
            return
        json_value = dump_json(value, mode="json")
        try:
            encoded = json.dumps(json_value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            entry[f"{key_prefix}PayloadKind"] = type(value).__name__
            entry[f"{key_prefix}ValueSummary"] = str(type(value).__name__)
            return
        entry[f"{key_prefix}PayloadKind"] = _payload_kind(json_value)
        entry[f"{key_prefix}ValueJsonBytes"] = len(encoded)
        if not include_value:
            entry[f"{key_prefix}ValueOmitted"] = True
            return
        if len(encoded) > max_value_bytes:
            entry[f"{key_prefix}ValueOmitted"] = True
            entry[f"{key_prefix}OmitReason"] = "value_too_large"
            return
        entry[f"{key_prefix}Value"] = json_value

    def set_cross_publish_policy(self, policy: CrossPublishPolicy) -> None:
        self._cross_publish_policy = policy

    def set_data_delivery(self, mode: DataDeliveryMode) -> None:
        self._data_delivery = mode

    def remove_node_inputs(self, node_id: str) -> None:
        node_id_s = str(node_id)
        for key in [item for item in self._inputs.keys() if item[0] == node_id_s]:
            self._inputs.pop(key, None)

    def queue_depth(self) -> int:
        depth = 0
        for buf in list(self._inputs.values()):
            try:
                depth += int(len(buf.queue))
            except (AttributeError, RuntimeError, TypeError):
                continue
        if depth < 0:
            return 0
        return int(depth)

    async def replace_routes(
        self,
        *,
        intra_data_out: DataOutRoutes,
        intra_data_in: DataOutRoutes,
        cross_in_by_key: DataCrossInRoutes,
        cross_out_keys: DataCrossOutRoutes,
        input_stream_keys: DataInputStreamRoutes | None = None,
    ) -> None:
        self._inputs.clear()
        self._outputs.clear()
        self._intra_data_out = dict(intra_data_out)
        self._intra_data_in = dict(intra_data_in)
        self._cross_in_by_key = dict(cross_in_by_key)
        self._cross_out_keys = dict(cross_out_keys)
        self._input_stream_keys = dict(input_stream_keys or {})
        self.precreate_input_buffers_for_cross_in(self._cross_in_by_key)
        await self.sync_subscriptions(set(self._cross_in_by_key.keys()))

    async def stop(self) -> None:
        for sub in list(self._custom_subscriptions):
            await sub.unsubscribe()
        self._custom_subscriptions.clear()

        for sub in list(self._route_subscriptions.values()):
            await sub.unsubscribe()
        self._route_subscriptions.clear()

        self._cross_in_by_key.clear()
        self._intra_data_out.clear()
        self._intra_data_in.clear()
        self._cross_out_keys.clear()
        self._input_stream_keys.clear()
        self._inputs.clear()
        self._on_data_push_queue.clear()
        await self._stop_flush_task()

    async def emit_data(
        self,
        node_id: str,
        port: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        ctx_id: str | int | None = None,
        options: DataEmitOptions | None = None,
    ) -> None:
        bus = self._bus
        if not bus._active:
            return
        emit_options = options or DEFAULT_DATA_EMIT_OPTIONS
        ts = int(ts_ms or now_ms())
        from_node = str(node_id)
        from_port = str(port)
        bus._monitor_record_emit(from_node, from_port, ts)
        await self._route_emitted_value(
            from_node=from_node,
            from_port=from_port,
            value=value,
            ts_ms=ts,
            ctx_id=ctx_id,
            options=emit_options,
        )

    async def pull_data(self, node_id: str, port: str, *, ctx_id: str | int | None = None) -> Any:
        bus = self._bus
        if not bus._active:
            return None
        buf = self._ensure_input_buffer(to_node=node_id, to_port=port, edge=None)
        edge = buf.edge
        now_ts = int(now_ms())

        last_seen_ts = int(buf.last_seen_ts or now_ts)
        if self.is_stale(edge, last_seen_ts):
            return None

        strategy = edge.strategy if edge is not None else F8EdgeStrategyEnum.latest
        if not isinstance(strategy, F8EdgeStrategyEnum):
            strategy = F8EdgeStrategyEnum.latest

        if strategy == F8EdgeStrategyEnum.queue:
            if not buf.queue:
                if ctx_id is None or buf.last_seen_ctx_id != ctx_id:
                    await self.ensure_input_available(node_id=node_id, port=port, ctx_id=ctx_id)
                if not buf.queue:
                    return None
            value, ts = buf.queue.popleft()
            if ts is not None:
                wait_ms = float(max(0, now_ts - int(ts)))
                bus._monitor_record_wait(wait_ms)
            buf.last_pulled_value = value
            buf.last_pulled_ts = int(ts) if ts is not None else now_ts
            buf.last_pulled_ctx_id = ctx_id
            bus._monitor_record_buffer_pull_delivery()
            return value

        if not buf.queue and (ctx_id is None or buf.last_seen_ctx_id != ctx_id):
            await self.ensure_input_available(node_id=node_id, port=port, ctx_id=ctx_id)
        value = buf.last_seen_value
        if buf.last_seen_ts is not None:
            wait_ms = float(max(0, now_ts - int(buf.last_seen_ts)))
            bus._monitor_record_wait(wait_ms)
        buf.queue.clear()
        if value is not None:
            buf.last_pulled_value = value
            buf.last_pulled_ts = now_ts
            buf.last_pulled_ctx_id = ctx_id
            bus._monitor_record_buffer_pull_delivery()
        return value

    async def ensure_input_available(self, *, node_id: str, port: str, ctx_id: str | int | None = None) -> None:
        bus = self._bus
        if not bus._graph:
            return

        upstream = self._intra_data_in.get((node_id, port)) or ()
        if not upstream:
            return

        stack: set[tuple[str, str]] = set()
        await self.compute_and_buffer_for_input(node_id=node_id, port=port, ctx_id=ctx_id, stack=stack)

    async def compute_and_buffer_for_input(
        self,
        *,
        node_id: str,
        port: str,
        ctx_id: str | int | None,
        stack: set[tuple[str, str]],
    ) -> None:
        bus = self._bus
        key = (str(node_id), str(port))
        if key in stack:
            return
        stack.add(key)
        try:
            for from_node, from_port, edge in self._intra_data_in.get(key) or ():
                src = bus._nodes.get(from_node)
                if src is None:
                    continue
                try:
                    if isinstance(src, ComputableNode):
                        value = await src.compute_output(from_port, ctx_id=ctx_id)
                    else:
                        value = None
                except _DATA_NODE_CALLBACK_ERRORS as exc:
                    log_error_once(
                        bus,
                        key=f"compute_output_failed:{from_node}:{from_port}",
                        message=f"compute_output failed for {from_node}.{from_port}",
                        exc=exc,
                    )
                    continue
                if value is None:
                    continue

                ts_now = int(now_ms())
                delivered = await self._route_emitted_value(
                    from_node=str(from_node),
                    from_port=str(from_port),
                    value=value,
                    ts_ms=ts_now,
                    ctx_id=ctx_id,
                    options=LOCAL_COMPUTE_DATA_EMIT_OPTIONS,
                    force_buffer_target=key,
                )
                if delivered:
                    continue
                self._deliver_local_input(
                    to_node=node_id,
                    to_port=port,
                    value=value,
                    ts_ms=ts_now,
                    edge=edge,
                    ctx_id=ctx_id,
                    force_buffer=True,
                )
        finally:
            stack.discard(key)

    async def on_cross_data_msg(self, key: str, payload: bytes) -> None:
        bus = self._bus
        if not bus._active:
            return
        targets = self._cross_in_by_key.get(str(key).strip("/")) or []
        if not targets:
            return
        value: Any = None
        ts: int | None = None
        try:
            msg = decode_obj(payload)
            value = msg.get("value")
            ts = msg.get("ts")
        except ValueError:
            value = payload

        ts_i = int(ts) if ts is not None else int(now_ms())
        for to_node, to_port, edge in targets:
            try:
                if self.is_stale(edge, ts_i):
                    continue
                self.push_input(to_node, to_port, value, ts_ms=ts_i, edge=edge)
            except _DATA_ROUTER_LIFECYCLE_ERRORS as exc:
                log_error_once(
                    bus,
                    key=f"cross_data_push_failed:{to_node}:{to_port}",
                    message=f"cross-data delivery failed for {to_node}.{to_port}",
                    exc=exc,
                )

    @staticmethod
    def is_stale(edge: F8Edge | None, ts_ms: int) -> bool:
        if edge is None:
            return False
        try:
            timeout = edge.timeoutMs
            if timeout is None or isinstance(timeout, msgspec.UnsetType):
                return False
            timeout_i = int(timeout)
            if timeout_i <= 0:
                return False
            return (now_ms() - int(ts_ms)) > timeout_i
        except (AttributeError, TypeError, ValueError):
            return False

    def push_input(self, to_node: str, to_port: str, value: Any, *, ts_ms: int, edge: F8Edge | None = None) -> None:
        self._deliver_local_input(
            to_node=to_node,
            to_port=to_port,
            value=value,
            ts_ms=int(ts_ms),
            edge=edge,
            ctx_id=None,
        )

    def buffer_input(
        self,
        to_node: str,
        to_port: str,
        value: Any,
        *,
        ts_ms: int,
        edge: F8Edge | None,
        ctx_id: str | int | None,
    ) -> None:
        bus = self._bus
        buf = self._ensure_input_buffer(to_node=to_node, to_port=to_port, edge=edge)

        buf.last_seen_value = value
        buf.last_seen_ts = int(ts_ms)
        buf.last_seen_ctx_id = ctx_id
        bus._monitor_record_input(str(to_node), str(to_port), int(ts_ms))

        buf.queue.append((value, int(ts_ms)))
        max_n = self._default_queue_size
        if buf.edge is not None:
            try:
                queue_size = buf.edge.queueSize
                max_n = max(1, queue_size) if isinstance(queue_size, int) else self._default_queue_size
            except (AttributeError, TypeError, ValueError):
                max_n = self._default_queue_size
        dropped_count = 0
        if len(buf.queue) > max_n:
            while len(buf.queue) > max_n:
                buf.queue.popleft()
                dropped_count += 1
        if dropped_count > 0:
            bus._monitor_record_drop(int(dropped_count))

    async def sync_subscriptions(self, want_keys: set[str]) -> None:
        bus = self._bus
        for key in list(self._route_subscriptions.keys()):
            if key in want_keys:
                continue
            sub = self._route_subscriptions.pop(key, None)
            if sub is None:
                continue
            try:
                await sub.unsubscribe()
            except _DATA_TRANSPORT_SUBSCRIPTION_ERRORS as exc:
                log.error("failed to unsubscribe routed key=%s", key, exc_info=exc)

        for key in want_keys:
            if key in self._route_subscriptions:
                continue

            async def _cb(s: str, p: bytes) -> None:
                await self.on_cross_data_msg(s, p)

            handle = await bus._transport.subscribe(key, cb=_cb)
            self._route_subscriptions[key] = handle

    async def subscribe_key(
        self,
        key_expr: str,
        *,
        queue: str | None = None,
        cb: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> Any:
        key_s = str(key_expr or "").strip("/")
        if not key_s:
            raise ValueError("key_expr must be non-empty")
        handle = await self._bus._transport.subscribe(key_s, queue=str(queue) if queue else None, cb=cb)
        self._custom_subscriptions.append(handle)
        return handle

    async def unsubscribe_key(self, handle: Any) -> None:
        if handle is None:
            return
        await handle.unsubscribe()
        if handle in self._custom_subscriptions:
            self._custom_subscriptions.remove(handle)

    def precreate_input_buffers_for_cross_in(self, cross_in: DataCrossInRoutes) -> None:
        for targets in cross_in.values():
            for to_node, to_port, edge in targets:
                self._ensure_input_buffer(
                    to_node=str(to_node),
                    to_port=str(to_port),
                    edge=edge,
                )

    def _ensure_input_buffer(self, *, to_node: str, to_port: str, edge: F8Edge | None) -> InputBuffer:
        key = (to_node, to_port)
        buf = self._inputs.get(key)
        if buf is None:
            buf = InputBuffer(to_node=to_node, to_port=to_port, edge=edge)
            self._inputs[key] = buf
        if edge is not None:
            buf.edge = edge
        return buf

    def _buffers_data_locally(self) -> bool:
        return True

    def _callbacks_data_locally(self) -> bool:
        return self._data_delivery == "callback"

    def _enqueue_on_data_callback(self, *, to_node: str, to_port: str, value: Any, ts_ms: int) -> None:
        bus = self._bus
        bus._monitor_record_callback_delivery()
        self._on_data_push_queue.append((to_node, to_port, value, int(ts_ms)))
        task = self._on_data_flush_task
        if task is None or task.done():
            try:
                loop = asyncio.get_running_loop()
                self._on_data_flush_task = loop.create_task(
                    self._flush_on_data_push_queue(),
                    name=f"service_bus:on_data_flush:{bus.service_id}",
                )
            except _DATA_ROUTER_SCHEDULE_ERRORS as exc:
                log_error_once(
                    bus,
                    key=f"push_on_data_schedule_failed:{to_node}:{to_port}",
                    message=f"failed to schedule on_data for {to_node}.{to_port}",
                    exc=exc,
                )

    def _deliver_local_input(
        self,
        *,
        to_node: str,
        to_port: str,
        value: Any,
        ts_ms: int,
        edge: F8Edge | None,
        ctx_id: str | int | None,
        force_buffer: bool = False,
    ) -> None:
        bus = self._bus
        if not bus._active:
            return
        if force_buffer or self._buffers_data_locally():
            self.buffer_input(
                to_node=to_node,
                to_port=to_port,
                value=value,
                ts_ms=ts_ms,
                edge=edge,
                ctx_id=ctx_id,
            )
        if self._callbacks_data_locally():
            self._enqueue_on_data_callback(to_node=to_node, to_port=to_port, value=value, ts_ms=int(ts_ms))

    async def _route_emitted_value(
        self,
        *,
        from_node: str,
        from_port: str,
        value: Any,
        ts_ms: int,
        ctx_id: str | int | None,
        options: DataEmitOptions,
        force_buffer_target: tuple[str, str] | None = None,
    ) -> bool:
        delivered = self._fanout_local_routes(
            from_node=from_node,
            from_port=from_port,
            value=value,
            ts_ms=int(ts_ms),
            ctx_id=ctx_id,
            options=options,
            force_buffer_target=force_buffer_target,
        )
        plan = self._cross_publish_plan(node_id=from_node, port=from_port)
        self._record_output_snapshot(
            from_node=from_node,
            from_port=from_port,
            value=value,
            ts_ms=int(ts_ms),
            ctx_id=ctx_id,
            delivered_local=delivered,
            cross_publish_plan=plan,
        )
        if options.publish_cross_service and plan.will_publish:
            payload = encode_obj({"value": value, "ts": int(ts_ms)})
            await self._bus._transport.publish(plan.key, payload)
            self._bus._monitor_record_routed_cross_emit()
            return delivered
        self._record_skipped_cross_publish(plan=plan, publish_enabled=options.publish_cross_service)
        return delivered

    def _record_output_snapshot(
        self,
        *,
        from_node: str,
        from_port: str,
        value: Any,
        ts_ms: int,
        ctx_id: str | int | None,
        delivered_local: bool,
        cross_publish_plan: CrossPublishPlan,
    ) -> None:
        key = (str(from_node), str(from_port))
        buffer = self._outputs.get(key)
        if buffer is None:
            buffer = OutputBuffer(
                from_node=key[0],
                from_port=key[1],
                snapshots=deque(maxlen=self._output_history_size),
            )
            self._outputs[key] = buffer
        buffer.snapshots.append(
            OutputSnapshot(
            from_node=str(from_node),
            from_port=str(from_port),
            value=value,
            ts_ms=int(ts_ms),
            ctx_id=ctx_id,
            delivered_local=bool(delivered_local),
            cross_publish_decision=str(cross_publish_plan.decision),
            cross_publish_key=str(cross_publish_plan.key),
            )
        )

    def _fanout_local_routes(
        self,
        *,
        from_node: str,
        from_port: str,
        value: Any,
        ts_ms: int,
        ctx_id: str | int | None,
        options: DataEmitOptions,
        force_buffer_target: tuple[str, str] | None,
    ) -> bool:
        if not options.deliver_local:
            return False
        delivered = False
        for to_node, to_port, edge in self._intra_data_out.get((from_node, from_port), ()):
            to_node_s = str(to_node)
            to_port_s = str(to_port)
            self._deliver_local_input(
                to_node=to_node_s,
                to_port=to_port_s,
                value=value,
                ts_ms=ts_ms,
                edge=edge,
                ctx_id=ctx_id,
                force_buffer=force_buffer_target == (to_node_s, to_port_s),
            )
            delivered = True
        return delivered

    def _record_skipped_cross_publish(self, *, plan: CrossPublishPlan, publish_enabled: bool) -> None:
        bus = self._bus
        if publish_enabled:
            if plan.decision == "suppressed":
                bus._monitor_record_suppressed_cross_publish()
            elif plan.decision == "local_only":
                bus._monitor_record_local_only_emit()
            return
        if plan.decision in ("publish", "suppressed"):
            bus._monitor_record_suppressed_cross_publish()

    def _cross_publish_plan(self, *, node_id: str, port: str) -> CrossPublishPlan:
        if self._cross_publish_policy == "none":
            if (node_id, port) in self._cross_out_keys:
                return CrossPublishPlan(key="", decision="suppressed")
            return CrossPublishPlan(key="", decision="local_only")
        if self._cross_publish_policy == "all":
            return CrossPublishPlan(
                key=data_key(self._bus.service_id, from_node_id=node_id, port_id=port),
                decision="publish",
            )
        key = self._cross_out_keys.get((node_id, port)) or ""
        if key:
            return CrossPublishPlan(key=key, decision="publish")
        return CrossPublishPlan(key="", decision="local_only")

    async def _flush_on_data_push_queue(self) -> None:
        try:
            while self._on_data_push_queue:
                if not self._bus._active:
                    self._on_data_push_queue.clear()
                    return
                batch: list[tuple[str, str, Any, int]] = []
                while self._on_data_push_queue:
                    batch.append(self._on_data_push_queue.popleft())
                coalesced: dict[tuple[str, str], tuple[Any, int]] = {}
                for node_id, port, value, ts_ms in batch:
                    coalesced[(node_id, port)] = (value, int(ts_ms))
                for (node_id, port), (value, ts_ms) in coalesced.items():
                    if not self._bus._active:
                        return
                    node = self._bus._nodes.get(node_id)
                    if node is None or not isinstance(node, DataReceivableNode):
                        continue
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            node.on_data(port, value, ts_ms=ts_ms),  # type: ignore[misc]
                            name=f"service_bus:on_data:{node_id}:{port}",
                        )
                    except _DATA_ROUTER_SCHEDULE_ERRORS as exc:
                        log_error_once(
                            self._bus,
                            key=f"push_on_data_schedule_failed:{node_id}:{port}",
                            message=f"failed to schedule on_data for {node_id}.{port}",
                            exc=exc,
                        )
                await asyncio.sleep(0)
        finally:
            self._on_data_flush_task = None

    async def _stop_flush_task(self) -> None:
        flush_task = self._on_data_flush_task
        self._on_data_flush_task = None
        if flush_task is not None:
            flush_task.cancel()
            try:
                await flush_task
            except asyncio.CancelledError:
                pass
            except _DATA_ROUTER_LIFECYCLE_ERRORS as exc:
                log.error("on_data flush task stop failed", exc_info=exc)
