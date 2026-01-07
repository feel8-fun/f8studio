from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
from collections.abc import Awaitable, Callable

from ..generated import F8EdgeScopeEnum, F8EdgeStrategyEnum

from ..graph.operator_graph import OperatorGraph
from .nats_naming import (
    data_subject,
    ensure_token,
    kv_bucket_for_service,
    kv_key_node_state,
    kv_key_topology,
)
from .nats_transport import NatsTransport, NatsTransportConfig
from .service_runtime_node import ServiceRuntimeNode


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    service_id: str
    nats_url: str = "nats://127.0.0.1:4222"
    kv_bucket: str | None = None
    actor_id: str | None = None
    publish_all_data: bool = True


@dataclass
class _Sub:
    subject: str
    handle: Any


@dataclass
class _InputBuffer:
    """
    Per-input-port buffer for rate mismatch handling (pull-based).
    """

    to_node: str
    to_port: str
    edge: Any
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


class ServiceRuntime:
    """
    ServiceSDK runtime (v1 + push v2).

    - Shared NATS connection (pub/sub + KV).
    - Watches `svc.<serviceId>.topology` inside the per-service KV bucket.
    - Builds intra/cross routing tables for data edges.
    - Provides a shared state KV API for nodes.
    - Push-based: triggers `ServiceRuntimeNode.on_data()` on new data events.
    """

    def __init__(self, config: ServiceRuntimeConfig) -> None:
        self.service_id = ensure_token(config.service_id, label="service_id")
        self.actor_id = ensure_token(config.actor_id or uuid.uuid4().hex, label="actor_id")
        self._publish_all_data = bool(getattr(config, "publish_all_data", True))

        bucket = (config.kv_bucket or "").strip() or os.environ.get("F8_NATS_BUCKET") or kv_bucket_for_service(self.service_id)
        self._transport = NatsTransport(NatsTransportConfig(url=str(config.nats_url), kv_bucket=str(bucket)))

        self._nodes: dict[str, ServiceRuntimeNode] = {}
        self._graph: OperatorGraph | None = None

        self._topology_key = kv_key_topology(self.service_id)
        self._topology_watch: Any | None = None
        self._local_state_watch: Any | None = None

        # Routing tables (data only for v1).
        self._intra_data_out: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self._cross_in_by_subject: dict[str, list[tuple[str, str, Any]]] = {}  # subject -> (to_node,to_port,edge)
        self._cross_out_subjects: dict[tuple[str, str], str] = {}  # (from_node, out_port) -> subject
        self._data_inputs: dict[tuple[str, str], _InputBuffer] = {}
        self._cross_state_in_by_key: dict[tuple[str, str], list[tuple[str, str, Any]]] = {}
        self._remote_state_watches: dict[tuple[str, str], Any] = {}
        self._state_cache: dict[tuple[str, str], tuple[Any, int]] = {}

        self._subs: dict[str, _Sub] = {}

        self._state_listeners: list[Callable[[str, str, Any, int, dict[str, Any]], Awaitable[None] | None]] = []
        self._topology_listeners: list[Callable[[OperatorGraph], Awaitable[None] | None]] = []

    def add_state_listener(
        self, cb: Callable[[str, str, Any, int, dict[str, Any]], Awaitable[None] | None]
    ) -> None:
        """
        Listen to local KV state updates for this service.

        Callback signature: (node_id, field, value, ts_ms, meta_dict)
        """
        self._state_listeners.append(cb)

    def add_topology_listener(self, cb: Callable[[OperatorGraph], Awaitable[None] | None]) -> None:
        """
        Listen to topology updates (after `OperatorGraph.from_dict` validation).
        """
        self._topology_listeners.append(cb)

    # ---- lifecycle ------------------------------------------------------
    def register_node(self, node: ServiceRuntimeNode) -> None:
        node_id = ensure_token(node.node_id, label="node_id")
        self._nodes[node_id] = node
        node.attach(self)

    def unregister_node(self, node_id: str) -> None:
        node_id = ensure_token(node_id, label="node_id")
        self._nodes.pop(node_id, None)
        for key in [k for k in self._data_inputs.keys() if k[0] == node_id]:
            self._data_inputs.pop(key, None)

    async def start(self) -> None:
        await self._transport.connect()
        if self._topology_watch is None:
            self._topology_watch = await self._transport.kv_watch(self._topology_key, cb=self._on_topology_kv)
        if self._local_state_watch is None:
            pattern = f"svc.{self.service_id}.nodes.>"
            self._local_state_watch = await self._transport.kv_watch(pattern, cb=self._on_local_state_kv)
        # Load once (if present).
        await self._reload_topology()

    async def stop(self) -> None:
        for sub in list(self._subs.values()):
            try:
                await sub.handle.unsubscribe()
            except Exception:
                pass
        self._subs.clear()
        self._cross_in_by_subject.clear()
        self._intra_data_out.clear()
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
            if self._topology_watch is not None:
                watcher, task = self._topology_watch
                try:
                    task.cancel()
                except Exception:
                    pass
                try:
                    await watcher.stop()
                except Exception:
                    pass
        finally:
            self._topology_watch = None
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
        key = kv_key_node_state(self.service_id, node_id=node_id, field=str(field))
        payload = {"value": value, "actor": self.actor_id, "ts": int(ts_ms or _now_ms())}
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
        """
        Like `set_state`, but allows additional metadata (eg. `source="editor"`).
        """
        node_id = ensure_token(node_id, label="node_id")
        key = kv_key_node_state(self.service_id, node_id=node_id, field=str(field))
        payload: dict[str, Any] = {"value": value, "actor": self.actor_id, "ts": int(ts_ms or _now_ms())}
        if source:
            payload["source"] = str(source)
        if meta:
            payload.update(meta)
        await self._transport.kv_put(key, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        self._state_cache[(node_id, str(field))] = (value, int(payload["ts"]))

    async def get_state(self, node_id: str, field: str) -> Any:
        node_id = ensure_token(node_id, label="node_id")
        cached = self._state_cache.get((node_id, str(field)))
        if cached is not None:
            return cached[0]
        key = kv_key_node_state(self.service_id, node_id=node_id, field=str(field))
        raw = await self._transport.kv_get(key)
        if not raw:
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict) and "value" in payload:
            return payload.get("value")
        return payload

    async def _on_local_state_kv(self, key: str, value: bytes) -> None:
        parsed = self._parse_state_key(key, service_id=self.service_id)
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
            if str(payload.get("actor") or "") == self.actor_id:
                return
            v = payload.get("value")
            ts = int(payload.get("ts") or _now_ms())
        else:
            v = payload
            ts = _now_ms()
        self._state_cache[(node_id, field)] = (v, ts)

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

    async def set_topology(self, payload: dict[str, Any]) -> None:
        """
        Publish a full topology snapshot for this service.
        """
        await self._transport.kv_put(
            self._topology_key, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        )

    # ---- data routing ---------------------------------------------------
    async def emit_data(self, node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        node_id = ensure_token(node_id, label="node_id")
        port = ensure_token(port, label="port_id")
        ts = int(ts_ms or _now_ms())

        # Intra edges (in-process).
        for to_node, to_port in self._intra_data_out.get((node_id, port), []):
            self._push_input(to_node, to_port, value, ts_ms=ts)

        # Cross edges (fanout) - publish once per (node, out_port).
        if self._publish_all_data:
            subject = data_subject(self.service_id, from_node_id=node_id, port_id=port)
        else:
            subject = self._cross_out_subjects.get((node_id, port)) or ""
        if not subject:
            return
        payload = json.dumps({"value": value, "ts": ts}, ensure_ascii=False, default=str).encode("utf-8")
        await self._transport.publish(subject, payload)

    async def publish(self, subject: str, payload: bytes) -> None:
        """
        Publish raw bytes to a NATS core subject (service-to-service commands, etc).
        """
        await self._transport.publish(str(subject), bytes(payload))

    async def subscribe(
        self,
        subject: str,
        *,
        queue: str | None = None,
        cb: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> Any:
        """
        Subscribe to a NATS core subject.
        """
        return await self._transport.subscribe(str(subject), queue=queue, cb=cb)

    async def pull_data(self, node_id: str, port: str) -> Any:
        """
        Pull-based access to buffered inputs.

        Strategy semantics (v1):
        - `latest`: return newest and clear the buffer.
        - `hold` / `repeat`: return newest if available else return last pulled value.
        - `average`: average buffered numeric values and clear buffer.
        - `interpolate`: compute linear interpolation between prev/newest at pull time.
        - `timeoutMs`: if newest sample is stale, return None.
        """
        node_id = ensure_token(node_id, label="node_id")
        port = ensure_token(port, label="port_id")
        buf = self._data_inputs.get((node_id, port))
        if buf is None:
            return None
        edge = getattr(buf, "edge", None) or {}
        now_ms = _now_ms()

        if self._is_stale(edge, int(buf.last_seen_ts or now_ms)):
            return None

        strat = getattr(edge, "strategy", None)
        strategy = strat if isinstance(strat, F8EdgeStrategyEnum) else F8EdgeStrategyEnum.latest

        if strategy == F8EdgeStrategyEnum.average:
            v = self._avg_buffer(buf)
            buf.queue.clear()
        elif strategy == F8EdgeStrategyEnum.interpolate:
            v = self._interp_buffer(buf, now_ms)
        else:
            v = self._latest_buffer(buf)
            if strategy == F8EdgeStrategyEnum.latest:
                buf.queue.clear()

        if strategy in (F8EdgeStrategyEnum.hold, F8EdgeStrategyEnum.repeat):
            if v is None:
                v = buf.last_pulled_value
            else:
                buf.last_pulled_value = v
                buf.last_pulled_ts = now_ms
        return v

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
    def _edge_direction(edge: Any) -> str:
        """
        Normalize `edge.direction` to "in"/"out".

        Generated schemas may represent direction as an Enum (`Direction.in_`),
        where `str()` is "Direction.in_" and `.name` is "in_".
        """
        d = getattr(edge, "direction", None)
        if d is None:
            return ""
        if isinstance(d, str):
            return d.strip().lower()
        try:
            v = getattr(d, "value", None)
            if isinstance(v, str) and v:
                return v.strip().lower()
        except Exception:
            pass
        try:
            n = getattr(d, "name", None)
            if isinstance(n, str) and n:
                return n.strip().lower().rstrip("_")
        except Exception:
            pass
        return str(d).strip().lower().replace("direction.", "").rstrip("_")

    @staticmethod
    def _is_stale(edge: Any, ts_ms: int) -> bool:
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

    @staticmethod
    def _interp_buffer(buf: _InputBuffer, now_ms: int) -> Any:
        if buf.prev_seen_ts is None or buf.last_seen_ts is None:
            return buf.last_seen_value
        if buf.prev_seen_value is None:
            return buf.last_seen_value
        try:
            a = float(buf.prev_seen_value)
            b = float(buf.last_seen_value)
        except Exception:
            return buf.last_seen_value
        t0 = int(buf.prev_seen_ts)
        t1 = int(buf.last_seen_ts)
        if t1 <= t0:
            return buf.last_seen_value
        if now_ms <= t0:
            return buf.prev_seen_value
        if now_ms >= t1:
            return buf.last_seen_value
        f = float(now_ms - t0) / float(t1 - t0)
        return a + (b - a) * f

    @staticmethod
    def _latest_buffer(buf: _InputBuffer) -> Any:
        if buf.queue:
            return buf.queue[-1][0]
        return buf.last_seen_value

    @staticmethod
    def _avg_buffer(buf: _InputBuffer) -> Any:
        vals = []
        for v, _ts in buf.queue:
            try:
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            return buf.last_seen_value
        return sum(vals) / float(len(vals))

    def _push_input(self, to_node: str, to_port: str, value: Any, *, ts_ms: int, edge: Any | None = None) -> None:
        to_node = str(to_node)
        to_port = str(to_port)
        buf = self._data_inputs.get((to_node, to_port))
        if buf is None:
            # Fallback buffer without edge metadata.
            buf = _InputBuffer(to_node=to_node, to_port=to_port, edge=edge or {})
            self._data_inputs[(to_node, to_port)] = buf
        if edge is not None:
            buf.edge = edge

        buf.prev_seen_value = buf.last_seen_value
        buf.prev_seen_ts = buf.last_seen_ts
        buf.last_seen_value = value
        buf.last_seen_ts = int(ts_ms)

        buf.queue.append((value, int(ts_ms)))
        # Cap queue size using `queueSize` if present.
        try:
            max_q = getattr(buf.edge, "queueSize", None)
            max_n = int(max_q) if max_q is not None else 0
        except Exception:
            max_n = 0
        if max_n <= 0:
            max_n = 256
        if len(buf.queue) > max_n:
            del buf.queue[0 : len(buf.queue) - max_n]

        # Push-based delivery (v2): trigger the target node immediately.
        node = self._nodes.get(to_node)
        if node is None:
            return
        try:
            r = node.on_data(to_port, value, ts_ms=int(ts_ms))
            if asyncio.iscoroutine(r):
                asyncio.create_task(r, name=f"on_data:{self.service_id}:{to_node}:{to_port}")
        except Exception:
            return

    # ---- topology -------------------------------------------------------
    async def _on_topology_kv(self, key: str, value: bytes) -> None:
        if str(key) != self._topology_key:
            return
        await self._apply_topology_bytes(value)

    async def _reload_topology(self) -> None:
        raw = await self._transport.kv_get(self._topology_key)
        if raw:
            await self._apply_topology_bytes(raw)

    async def _apply_topology_bytes(self, raw: bytes) -> None:
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            return
        try:
            graph = OperatorGraph.from_dict(payload)
        except Exception:
            return
        self._graph = graph
        await self._rebuild_routes()
        for cb in list(self._topology_listeners):
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
        # Topology can change; reset buffers to avoid keeping stale edges.
        self._data_inputs.clear()

        # Build intra routes for local nodes.
        intra: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for edge in graph.data_edges:
            if edge.scope != F8EdgeScopeEnum.intra:
                continue
            intra.setdefault((str(edge.from_), str(edge.fromPort)), []).append((str(edge.to), str(edge.toPort)))
        self._intra_data_out = intra

        # Build cross routes.
        cross_in: dict[str, list[tuple[str, str, Any]]] = {}
        cross_out: dict[tuple[str, str], str] = {}
        for edge in graph.data_edges:
            if edge.scope != F8EdgeScopeEnum.cross:
                continue
            direction = self._edge_direction(edge)
            subject = str(getattr(edge, "subject", "") or "")
            if direction == "in":
                if not subject:
                    continue
                to_node = str(edge.to)
                if to_node not in self._nodes:
                    continue
                cross_in.setdefault(subject, []).append((to_node, str(edge.toPort), edge))
            elif direction == "out":
                if not subject:
                    continue
                from_node = str(edge.from_)
                if from_node not in self._nodes:
                    continue
                cross_out[(from_node, str(edge.fromPort))] = subject
        self._cross_in_by_subject = cross_in
        self._cross_out_subjects = cross_out

        # Pre-create input buffers for known local inputs.
        for subject, targets in cross_in.items():
            for to_node, to_port, edge in targets:
                self._data_inputs[(str(to_node), str(to_port))] = _InputBuffer(
                    to_node=str(to_node), to_port=str(to_port), edge=edge
                )

        # Update subscriptions (only for subjects we currently need).
        await self._sync_subscriptions(set(cross_in.keys()))

        await self._sync_cross_state_watches(graph)

    async def _sync_cross_state_watches(self, graph: OperatorGraph) -> None:
        """
        Cross-state binding via remote KV watch (read remote, apply to local).
        """
        want: dict[tuple[str, str], list[tuple[str, str, Any]]] = {}
        for edge in graph.state_edges:
            if edge.scope != F8EdgeScopeEnum.cross:
                continue
            direction = self._edge_direction(edge)
            if direction != "in":
                continue
            peer = str(getattr(edge, "peerServiceId", "") or "").strip()
            if not peer:
                continue
            try:
                peer = ensure_token(peer, label="peerServiceId")
            except Exception:
                continue

            # local endpoint is `to`, remote endpoint is `from`.
            local_node = str(edge.to)
            local_field = str(edge.toPort)
            if local_node not in self._nodes:
                continue
            remote_node = str(edge.from_)
            remote_field = str(edge.fromPort)
            remote_key = kv_key_node_state(peer, node_id=remote_node, field=remote_field)
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
        parsed = self._parse_state_key(key, service_id=peer_service_id)
        if not parsed:
            return
        remote_node, remote_field = parsed
        remote_key = kv_key_node_state(peer_service_id, node_id=remote_node, field=remote_field)
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
            # Mirror into local KV/cache as the resolved state value.
            try:
                await self.set_state(local_node_id, local_field, v, ts_ms=ts)
            except Exception:
                pass

    @staticmethod
    def _parse_state_key(key: str, *, service_id: str) -> tuple[str, str] | None:
        """
        Parse `svc.<serviceId>.nodes.<nodeId>.state.<field...>`
        """
        parts = str(key).strip(".").split(".")
        if len(parts) < 6:
            return None
        if parts[0] != "svc" or parts[2] != "nodes" or parts[4] != "state":
            return None
        if parts[1] != str(service_id):
            return None
        node_id = parts[3]
        field = ".".join(parts[5:])
        if not node_id or not field:
            return None
        return node_id, field

    async def _sync_subscriptions(self, want_subjects: set[str]) -> None:
        # Remove.
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

        # Add.
        for subject in want_subjects:
            if subject in self._subs:
                continue

            async def _cb(s: str, p: bytes) -> None:
                await self._on_cross_data_msg(s, p)

            handle = await self._transport.subscribe(subject, cb=_cb)
            self._subs[subject] = _Sub(subject=subject, handle=handle)
