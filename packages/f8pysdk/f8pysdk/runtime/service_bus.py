from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from nats.js.api import StorageType  # type: ignore[import-not-found]

from ..generated import F8Edge, F8EdgeKindEnum, F8EdgeStrategyEnum, F8RuntimeGraph
from .nats_naming import data_subject, ensure_token, kv_bucket_for_service, kv_key_node_state, kv_key_rungraph
from .nats_transport import NatsTransport, NatsTransportConfig
from .service_runtime_node import RuntimeNode


def _now_ms() -> int:
    return int(time.time() * 1000)


def _debug_state_enabled() -> bool:
    return str(os.getenv("F8_STATE_DEBUG", "")).lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ServiceBusConfig:
    service_id: str
    nats_url: str = "nats://127.0.0.1:4222"
    publish_all_data: bool = True
    kv_storage: StorageType = StorageType.MEMORY
    delete_bucket_on_start: bool = False
    delete_bucket_on_stop: bool = False


@dataclass
class _Sub:
    subject: str
    handle: Any


@dataclass
class _InputBuffer:
    to_node: str
    to_port: str
    edge: F8Edge | None
    queue: list[tuple[Any, int]] = None  # type: ignore[assignment]
    last_seen_value: Any = None
    last_seen_ts: int | None = None
    prev_seen_value: Any = None
    prev_seen_ts: int | None = None
    last_pulled_value: Any = None
    last_pulled_ts: int | None = None

    def __post_init__(self) -> None:
        if self.queue is None:
            self.queue = []


class ServiceBus:
    """
    Service bus (clean, protocol-first).

    - Shared NATS connection (pub/sub + JetStream KV).
    - Watches `rungraph` (KV) which must be a `F8RuntimeGraph`.
    - Builds intra/cross routing tables for data edges.
    - Provides a shared state KV API for nodes.
    - Data edges are pull-based: consumers pull buffered inputs and may trigger
      intra-service computation via `compute_output(...)`.
    """

    def __init__(self, config: ServiceBusConfig) -> None:
        self.service_id = ensure_token(config.service_id, label="service_id")
        self._publish_all_data = bool(getattr(config, "publish_all_data", True))
        self._debug_state = _debug_state_enabled()

        bucket = kv_bucket_for_service(self.service_id)
        self._transport = NatsTransport(
            NatsTransportConfig(
                url=str(config.nats_url),
                kv_bucket=str(bucket),
                kv_storage=getattr(config, "kv_storage", None),
                delete_bucket_on_connect=bool(getattr(config, "delete_bucket_on_start", False)),
                delete_bucket_on_close=bool(getattr(config, "delete_bucket_on_stop", False)),
            )
        )

        self._nodes: dict[str, RuntimeNode] = {}
        self._graph: F8RuntimeGraph | None = None

        self._rungraph_key = kv_key_rungraph()
        self._rungraph_watch: Any | None = None
        self._local_state_watch: Any | None = None

        # Routing tables (data only).
        self._intra_data_out: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self._intra_data_in: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
        self._cross_in_by_subject: dict[str, list[tuple[str, str, F8Edge]]] = {}
        self._cross_out_subjects: dict[tuple[str, str], str] = {}
        self._data_inputs: dict[tuple[str, str], _InputBuffer] = {}

        # Cross-state binding (remote KV -> local node.on_state + local KV mirror).
        self._cross_state_in_by_key: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
        self._remote_state_watches: dict[tuple[str, str], Any] = {}

        self._state_cache: dict[tuple[str, str], tuple[Any, int]] = {}
        self._subs: dict[str, _Sub] = {}

        self._state_listeners: list[Callable[[str, str, Any, int, dict[str, Any]], Awaitable[None] | None]] = []
        self._rungraph_listeners: list[Callable[[F8RuntimeGraph], Awaitable[None] | None]] = []

    def add_state_listener(self, cb: Callable[[str, str, Any, int, dict[str, Any]], Awaitable[None] | None]) -> None:
        """
        Listen to local KV state updates for this service.

        Callback signature: (node_id, field, value, ts_ms, meta_dict)
        """
        self._state_listeners.append(cb)

    def add_rungraph_listener(self, cb: Callable[[F8RuntimeGraph], Awaitable[None] | None]) -> None:
        """
        Listen to rungraph updates (after `F8RuntimeGraph` validation).
        """
        self._rungraph_listeners.append(cb)

    # ---- lifecycle ------------------------------------------------------
    def register_node(self, node: RuntimeNode) -> None:
        node_id = ensure_token(node.node_id, label="node_id")
        self._nodes[node_id] = node
        node.attach(self)

    def unregister_node(self, node_id: str) -> None:
        node_id = ensure_token(node_id, label="node_id")
        self._nodes.pop(node_id, None)
        for key in [k for k in self._data_inputs.keys() if k[0] == node_id]:
            self._data_inputs.pop(key, None)

    def get_node(self, node_id: str) -> RuntimeNode | None:
        """
        Return the local runtime node instance if registered.
        """
        try:
            node_id = ensure_token(node_id, label="node_id")
        except Exception:
            return None
        return self._nodes.get(node_id)

    async def start(self) -> None:
        await self._transport.connect()
        if self._rungraph_watch is None:
            self._rungraph_watch = await self._transport.kv_watch(self._rungraph_key, cb=self._on_rungraph_kv)
        if self._local_state_watch is None:
            pattern = "nodes.>"
            self._local_state_watch = await self._transport.kv_watch(pattern, cb=self._on_local_state_kv)
        await self._reload_rungraph()

    async def stop(self) -> None:
        for sub in list(self._subs.values()):
            try:
                await sub.handle.unsubscribe()
            except Exception:
                pass
        self._subs.clear()

        self._cross_in_by_subject.clear()
        self._intra_data_out.clear()
        self._intra_data_in.clear()
        self._cross_out_subjects.clear()
        self._data_inputs.clear()

        self._cross_state_in_by_key.clear()
        for (_sid, _key), watch in list(self._remote_state_watches.items()):
            try:
                watcher, task = watch
                try:
                    task.cancel()
                except Exception:
                    pass
                try:
                    await watcher.stop()
                except Exception:
                    pass
            except Exception:
                pass
        self._remote_state_watches.clear()

        self._state_cache.clear()

        try:
            if self._rungraph_watch is not None:
                watcher, task = self._rungraph_watch
                try:
                    task.cancel()
                except Exception:
                    pass
                try:
                    await watcher.stop()
                except Exception:
                    pass
        finally:
            self._rungraph_watch = None

        try:
            if self._local_state_watch is not None:
                watcher, task = self._local_state_watch
                try:
                    task.cancel()
                except Exception:
                    pass
                try:
                    await watcher.stop()
                except Exception:
                    pass
        finally:
            self._local_state_watch = None

        await self._transport.close()

    # ---- KV state -------------------------------------------------------
    async def set_state(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        node_id = ensure_token(node_id, label="node_id")
        key = kv_key_node_state(node_id=node_id, field=str(field))
        payload = {"value": value, "actor": self.service_id, "ts": int(ts_ms or _now_ms())}
        if self._debug_state:
            print(
                "state_debug[%s] set_state node=%s field=%s ts=%s"
                % (self.service_id, node_id, str(field), str(payload.get("ts")))
            )
        await self._transport.kv_put(key, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        self._state_cache[(node_id, str(field))] = (value, int(payload["ts"]))

    async def set_state_with_meta(
        self,
        node_id: str,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        source: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        node_id = ensure_token(node_id, label="node_id")
        key = kv_key_node_state(node_id=node_id, field=str(field))
        payload: dict[str, Any] = {"value": value, "actor": self.service_id, "ts": int(ts_ms or _now_ms())}
        if source:
            payload["source"] = str(source)
        if meta:
            payload.update(meta)
        if self._debug_state:
            print(
                "state_debug[%s] set_state_with_meta node=%s field=%s ts=%s source=%s meta=%s"
                % (self.service_id, node_id, str(field), str(payload.get("ts")), str(source or ""), str(meta or ""))
            )
        await self._transport.kv_put(key, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        self._state_cache[(node_id, str(field))] = (value, int(payload["ts"]))

    async def get_state(self, node_id: str, field: str) -> Any:
        node_id = ensure_token(node_id, label="node_id")
        field = str(field)
        cached = self._state_cache.get((node_id, field))
        if cached is not None:
            return cached[0]
        key = kv_key_node_state(node_id=node_id, field=field)
        raw = await self._transport.kv_get(key)
        if not raw:
            if self._debug_state:
                print("state_debug[%s] get_state miss node=%s field=%s" % (self.service_id, node_id, field))
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict) and "value" in payload:
            v = payload.get("value")
            ts_raw = payload.get("ts")
            try:
                ts = int(ts_raw) if ts_raw is not None else 0
            except Exception:
                ts = 0
            self._state_cache[(node_id, field)] = (v, ts)
            if self._debug_state:
                print(
                    "state_debug[%s] get_state kv node=%s field=%s ts=%s" % (self.service_id, node_id, field, str(ts))
                )
            return v
        self._state_cache[(node_id, field)] = (payload, 0)
        return payload

    async def _on_local_state_kv(self, key: str, value: bytes) -> None:
        parsed = self._parse_state_key(key)
        if not parsed:
            return
        node_id, field = parsed
        try:
            payload = json.loads(value.decode("utf-8")) if value else {}
        except Exception:
            payload = {}
        meta_dict: dict[str, Any] = {}
        if isinstance(payload, dict):
            meta_dict = dict(payload)
            v = payload.get("value")
            ts_raw = payload.get("ts")
            try:
                ts = int(ts_raw) if ts_raw is not None else 0
            except Exception:
                ts = 0
        else:
            v = payload
            ts = 0
        cached = self._state_cache.get((node_id, field))
        if cached is not None and ts <= int(cached[1]):
            if self._debug_state:
                actor = ""
                if isinstance(payload, dict):
                    actor = str(payload.get("actor") or "")
                print(
                    "state_debug[%s] kv_update drop node=%s field=%s ts=%s cached_ts=%s actor=%s"
                    % (self.service_id, node_id, field, str(ts), str(cached[1]), actor)
                )
            return
        self._state_cache[(node_id, field)] = (v, ts)
        if isinstance(payload, dict) and str(payload.get("actor") or "") == self.service_id:
            if self._debug_state:
                print(
                    "state_debug[%s] kv_update self node=%s field=%s ts=%s"
                    % (self.service_id, node_id, field, str(ts))
                )
            return
        if self._debug_state:
            actor = ""
            if isinstance(payload, dict):
                actor = str(payload.get("actor") or "")
            print(
                "state_debug[%s] kv_update apply node=%s field=%s ts=%s actor=%s"
                % (self.service_id, node_id, field, str(ts), actor)
            )

        for cb in list(self._state_listeners):
            try:
                r = cb(node_id, field, v, ts, meta_dict)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                continue

        node = self._nodes.get(node_id)
        if node is None:
            return
        try:
            await node.on_state(field, v, ts_ms=ts)
        except Exception:
            return

    # ---- rungraph -------------------------------------------------------
    async def set_rungraph(self, graph: F8RuntimeGraph) -> None:
        """
        Publish a full rungraph snapshot for this service.
        """
        payload = graph.model_dump(mode="json", by_alias=True)
        await self._transport.kv_put(
            self._rungraph_key, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        )

    async def _on_rungraph_kv(self, key: str, value: bytes) -> None:
        if str(key) != self._rungraph_key:
            return
        await self._apply_rungraph_bytes(value)

    async def _reload_rungraph(self) -> None:
        raw = await self._transport.kv_get(self._rungraph_key)
        if raw:
            await self._apply_rungraph_bytes(raw)

    async def _apply_rungraph_bytes(self, raw: bytes) -> None:
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            return
        try:
            graph = F8RuntimeGraph.model_validate(payload)
        except Exception:
            return

        self._graph = graph
        if self._debug_state:
            try:
                node_count = len(list(graph.nodes or []))
                edge_count = len(list(graph.edges or []))
                graph_id = str(getattr(graph, "graphId", "") or "")
            except Exception:
                node_count = 0
                edge_count = 0
                graph_id = ""
            print(
                "state_debug[%s] rungraph_applied graph=%s nodes=%s edges=%s"
                % (self.service_id, graph_id, str(node_count), str(edge_count))
            )
        await self._rebuild_routes()

        for cb in list(self._rungraph_listeners):
            try:
                r = cb(graph)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                continue

    async def _rebuild_routes(self) -> None:
        graph = self._graph
        if graph is None:
            return

        self._data_inputs.clear()

        # Intra (in-process) routing: local service -> local service.
        intra: dict[tuple[str, str], list[tuple[str, str]]] = {}
        intra_in: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
        for edge in graph.edges:
            if edge.kind != F8EdgeKindEnum.data:
                continue
            if str(edge.fromServiceId) != self.service_id or str(edge.toServiceId) != self.service_id:
                continue
            if not edge.fromOperatorId or not edge.toOperatorId:
                continue
            intra.setdefault((str(edge.fromOperatorId), str(edge.fromPort)), []).append(
                (str(edge.toOperatorId), str(edge.toPort))
            )
            intra_in.setdefault((str(edge.toOperatorId), str(edge.toPort)), []).append(
                (str(edge.fromOperatorId), str(edge.fromPort), edge)
            )
        self._intra_data_out = intra
        self._intra_data_in = intra_in

        # Cross routing.
        cross_in: dict[str, list[tuple[str, str, F8Edge]]] = {}
        cross_out: dict[tuple[str, str], str] = {}
        for edge in graph.edges:
            if edge.kind != F8EdgeKindEnum.data:
                continue
            if str(edge.fromServiceId) == str(edge.toServiceId):
                continue
            if not edge.fromOperatorId:
                continue

            subject = data_subject(
                str(edge.fromServiceId), from_node_id=str(edge.fromOperatorId), port_id=str(edge.fromPort)
            )

            # Incoming: local service is the target.
            if str(edge.toServiceId) == self.service_id:
                if not edge.toOperatorId:
                    continue
                to_node = str(edge.toOperatorId)
                cross_in.setdefault(subject, []).append((to_node, str(edge.toPort), edge))
                continue

            # Outgoing: local service is the source.
            if str(edge.fromServiceId) == self.service_id:
                from_node = str(edge.fromOperatorId)
                cross_out[(from_node, str(edge.fromPort))] = subject

        self._cross_in_by_subject = cross_in
        self._cross_out_subjects = cross_out

        # Pre-create input buffers for known local inputs.
        for subject, targets in cross_in.items():
            for to_node, to_port, edge in targets:
                self._data_inputs[(str(to_node), str(to_port))] = _InputBuffer(
                    to_node=str(to_node), to_port=str(to_port), edge=edge
                )

        await self._sync_subscriptions(set(cross_in.keys()))
        await self._sync_cross_state_watches(graph)

    # ---- data routing ---------------------------------------------------
    async def emit_data(self, node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        node_id = ensure_token(node_id, label="node_id")
        port = ensure_token(port, label="port_id")
        ts = int(ts_ms or _now_ms())

        # Intra edges.
        for to_node, to_port in self._intra_data_out.get((node_id, port), []):
            self._push_input(to_node, to_port, value, ts_ms=ts)

        # Cross edges (fan-out) - publish once per (node, out_port).
        if self._publish_all_data:
            subject = data_subject(self.service_id, from_node_id=node_id, port_id=port)
        else:
            subject = self._cross_out_subjects.get((node_id, port)) or ""
        if not subject:
            return
        payload = json.dumps({"value": value, "ts": ts}, ensure_ascii=False, default=str).encode("utf-8")
        await self._transport.publish(subject, payload)

    async def publish(self, subject: str, payload: bytes) -> None:
        await self._transport.publish(str(subject), bytes(payload))

    async def subscribe(
        self,
        subject: str,
        *,
        queue: str | None = None,
        cb: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> Any:
        return await self._transport.subscribe(str(subject), queue=queue, cb=cb)

    async def pull_data(self, node_id: str, port: str, *, ctx_id: str | int | None = None) -> Any:
        """
        Pull-based access to buffered inputs.

        Strategy semantics:
        - `latest`: return newest and clear the buffer.
        - `queue`: pop the oldest buffered item (FIFO).
        - `timeoutMs`: if newest sample is stale, return None.
        """
        node_id = ensure_token(node_id, label="node_id")
        port = ensure_token(port, label="port_id")
        buf = self._data_inputs.get((node_id, port))
        if buf is None:
            buf = _InputBuffer(to_node=node_id, to_port=port, edge=None)
            self._data_inputs[(node_id, port)] = buf
        edge = buf.edge
        now_ms = _now_ms()

        last_seen_ts = int(buf.last_seen_ts or now_ms)
        if self._is_stale(edge, last_seen_ts):
            return None

        strategy = getattr(edge, "strategy", None) if edge is not None else None
        if not isinstance(strategy, F8EdgeStrategyEnum):
            strategy = F8EdgeStrategyEnum.latest

        if strategy == F8EdgeStrategyEnum.queue:
            if not buf.queue:
                await self._ensure_input_available(node_id=node_id, port=port, ctx_id=ctx_id)
                if not buf.queue:
                    return None
            v, ts = buf.queue.pop(0)
            buf.last_pulled_value = v
            buf.last_pulled_ts = int(ts) if ts is not None else now_ms
            return v

        # latest
        if not buf.queue and buf.last_seen_value is None:
            await self._ensure_input_available(node_id=node_id, port=port, ctx_id=ctx_id)
        v = buf.queue[-1][0] if buf.queue else buf.last_seen_value
        buf.queue.clear()
        if v is not None:
            buf.last_pulled_value = v
            buf.last_pulled_ts = now_ms
        return v

    async def _ensure_input_available(self, *, node_id: str, port: str, ctx_id: str | int | None = None) -> None:
        """
        Best-effort intra-service pull-triggered computation.

        If (node_id, port) has no buffered samples and the rungraph defines intra data
        edges feeding it, attempt to compute upstream outputs and buffer the results.
        """
        if not self._graph:
            return

        upstream = self._intra_data_in.get((str(node_id), str(port))) or []
        if not upstream:
            return

        stack: set[tuple[str, str]] = set()
        await self._compute_and_buffer_for_input(node_id=str(node_id), port=str(port), ctx_id=ctx_id, stack=stack)

    async def _compute_and_buffer_for_input(
        self,
        *,
        node_id: str,
        port: str,
        ctx_id: str | int | None,
        stack: set[tuple[str, str]],
    ) -> None:
        key = (str(node_id), str(port))
        if key in stack:
            return
        stack.add(key)
        try:
            for from_node, from_port, edge in list(self._intra_data_in.get(key) or []):
                src = self._nodes.get(str(from_node))
                if src is None:
                    continue
                if not hasattr(src, "compute_output"):
                    continue
                try:
                    v = await src.compute_output(str(from_port), ctx_id=ctx_id)  # type: ignore[misc]
                except Exception:
                    continue
                if v is None:
                    continue
                self._buffer_input(str(node_id), str(port), v, ts_ms=_now_ms(), edge=edge, notify=False)
        finally:
            stack.discard(key)

    async def _on_cross_data_msg(self, subject: str, payload: bytes) -> None:
        targets = self._cross_in_by_subject.get(str(subject)) or []
        if not targets:
            return
        value: Any = None
        ts: int | None = None
        try:
            msg = json.loads(payload.decode("utf-8")) if payload else {}
            if isinstance(msg, dict):
                value = msg.get("value")
                ts = msg.get("ts")
            else:
                value = msg
        except Exception:
            value = payload

        ts_i = int(ts) if ts is not None else _now_ms()
        for to_node, to_port, edge in targets:
            try:
                if self._is_stale(edge, ts_i):
                    continue
                self._push_input(to_node, to_port, value, ts_ms=ts_i, edge=edge)
            except Exception:
                continue

    @staticmethod
    def _is_stale(edge: F8Edge | None, ts_ms: int) -> bool:
        if edge is None:
            return False
        try:
            timeout = getattr(edge, "timeoutMs", None)
            if timeout is None:
                return False
            t = int(timeout)
            if t <= 0:
                return False
            return (_now_ms() - int(ts_ms)) > t
        except Exception:
            return False

    def _push_input(self, to_node: str, to_port: str, value: Any, *, ts_ms: int, edge: F8Edge | None = None) -> None:
        # Data inputs are buffered. We intentionally do not invoke `node.on_data`:
        # the system is moving to exec-driven scheduling + pull-based data reads.
        self._buffer_input(
            to_node=str(to_node),
            to_port=str(to_port),
            value=value,
            ts_ms=int(ts_ms),
            edge=edge,
            notify=False,
        )

    def _buffer_input(
        self,
        to_node: str,
        to_port: str,
        value: Any,
        *,
        ts_ms: int,
        edge: F8Edge | None,
        notify: bool,
    ) -> None:
        to_node = str(to_node)
        to_port = str(to_port)
        buf = self._data_inputs.get((to_node, to_port))
        if buf is None:
            buf = _InputBuffer(to_node=to_node, to_port=to_port, edge=edge)
            self._data_inputs[(to_node, to_port)] = buf
        if edge is not None:
            buf.edge = edge

        buf.prev_seen_value = buf.last_seen_value
        buf.prev_seen_ts = buf.last_seen_ts
        buf.last_seen_value = value
        buf.last_seen_ts = int(ts_ms)

        buf.queue.append((value, int(ts_ms)))
        max_n = 256
        if buf.edge is not None:
            try:
                qs = getattr(buf.edge, "queueSize", None)
                if qs is not None:
                    max_n = max(1, int(qs))
            except Exception:
                max_n = 256
        if len(buf.queue) > max_n:
            del buf.queue[0 : len(buf.queue) - max_n]

        if not notify:
            return

    # ---- cross-state ----------------------------------------------------
    async def _sync_cross_state_watches(self, graph: F8RuntimeGraph) -> None:
        """
        Cross-state binding via remote KV watch (read remote, apply to local).
        """
        want: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
        for edge in graph.edges:
            if edge.kind != F8EdgeKindEnum.state:
                continue
            if str(edge.fromServiceId) == str(edge.toServiceId):
                continue
            if str(edge.toServiceId) != self.service_id:
                continue
            peer = str(edge.fromServiceId or "").strip()
            try:
                peer = ensure_token(peer, label="fromServiceId")
            except Exception:
                continue

            if not edge.toOperatorId or not edge.fromOperatorId:
                continue

            local_node = str(edge.toOperatorId)
            local_field = str(edge.toPort)
            remote_node = str(edge.fromOperatorId)
            remote_field = str(edge.fromPort)
            remote_key = kv_key_node_state(node_id=remote_node, field=remote_field)
            want.setdefault((peer, remote_key), []).append((local_node, local_field, edge))

        # Stop watches no longer needed.
        for k, watch in list(self._remote_state_watches.items()):
            if k in want:
                continue
            try:
                watcher, task = watch
                try:
                    task.cancel()
                except Exception:
                    pass
                try:
                    await watcher.stop()
                except Exception:
                    pass
            except Exception:
                pass
            self._remote_state_watches.pop(k, None)

        self._cross_state_in_by_key = want

        # Start new watches.
        for k in want.keys():
            if k in self._remote_state_watches:
                continue
            peer, remote_key = k
            bucket = kv_bucket_for_service(peer)

            async def _cb(key: str, val: bytes, *, _peer: str = peer) -> None:
                await self._on_remote_state_kv(_peer, key, val)

            self._remote_state_watches[k] = await self._transport.kv_watch_in_bucket(bucket, remote_key, cb=_cb)

    async def _on_remote_state_kv(self, peer_service_id: str, key: str, value: bytes) -> None:
        parsed = self._parse_state_key(key)
        if not parsed:
            return
        remote_node, remote_field = parsed
        remote_key = kv_key_node_state(node_id=remote_node, field=remote_field)
        targets = self._cross_state_in_by_key.get((peer_service_id, remote_key)) or []
        if not targets:
            return
        try:
            payload = json.loads(value.decode("utf-8")) if value else {}
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            v = payload.get("value")
            ts = int(payload.get("ts") or _now_ms())
        else:
            v = payload
            ts = _now_ms()

        for local_node_id, local_field, _edge in targets:
            node = self._nodes.get(local_node_id)
            if node is None:
                continue
            try:
                await node.on_state(local_field, v, ts_ms=ts)
            except Exception:
                pass
            try:
                await self.set_state(local_node_id, local_field, v, ts_ms=ts)
            except Exception:
                pass

    @staticmethod
    def _parse_state_key(key: str) -> tuple[str, str] | None:
        parts = str(key).strip(".").split(".")
        if len(parts) < 4:
            return None
        if parts[0] != "nodes" or parts[2] != "state":
            return None
        node_id = parts[1]
        field = ".".join(parts[3:])
        if not node_id or not field:
            return None
        return node_id, field

    # ---- subscriptions --------------------------------------------------
    async def _sync_subscriptions(self, want_subjects: set[str]) -> None:
        for subject in list(self._subs.keys()):
            if subject in want_subjects:
                continue
            sub = self._subs.pop(subject, None)
            if sub is None:
                continue
            try:
                await sub.handle.unsubscribe()
            except Exception:
                pass

        for subject in want_subjects:
            if subject in self._subs:
                continue

            async def _cb(s: str, p: bytes) -> None:
                await self._on_cross_data_msg(s, p)

            handle = await self._transport.subscribe(subject, cb=_cb)
            self._subs[subject] = _Sub(subject=subject, handle=handle)
