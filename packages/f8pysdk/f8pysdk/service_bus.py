from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from nats.js.api import StorageType  # type: ignore[import-not-found]
from nats.micro import ServiceConfig, add_service  # type: ignore[import-not-found]
from nats.micro.service import EndpointConfig  # type: ignore[import-not-found]

from .capabilities import BusAttachableNode, ClosableNode, CommandableNode, ComputableNode, StatefulNode
from .generated import F8Edge, F8EdgeKindEnum, F8EdgeStrategyEnum, F8RuntimeGraph
from .nats_naming import (
    cmd_channel_subject,
    data_subject,
    ensure_token,
    kv_bucket_for_service,
    kv_key_node_state,
    kv_key_rungraph,
    new_id,
    svc_endpoint_subject,
    svc_micro_name,
)
from .nats_transport import NatsTransport, NatsTransportConfig
from .time_utils import now_ms


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
    last_seen_ctx_id: str | int | None = None
    last_pulled_value: Any = None
    last_pulled_ts: int | None = None
    last_pulled_ctx_id: str | int | None = None

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
        self._active = True

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

        self._nodes: dict[str, BusAttachableNode] = {}
        self._graph: F8RuntimeGraph | None = None

        self._rungraph_key = kv_key_rungraph()
        self._rungraph_watch: Any | None = None
        self._local_state_watch: Any | None = None
        self._micro: Any | None = None

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
        self._raw_subs: list[Any] = []

        self._state_listeners: list[Callable[[str, str, Any, int, dict[str, Any]], Awaitable[None] | None]] = []
        self._rungraph_listeners: list[Callable[[F8RuntimeGraph], Awaitable[None] | None]] = []
        self._lifecycle_listeners: list[Callable[[bool, dict[str, Any]], Awaitable[None] | None]] = []

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

    def add_lifecycle_listener(self, cb: Callable[[bool, dict[str, Any]], Awaitable[None] | None]) -> None:
        """
        Listen to service lifecycle changes (activate/deactivate).

        Callback signature: (active, meta_dict)
        """
        self._lifecycle_listeners.append(cb)

    # ---- lifecycle ------------------------------------------------------
    @property
    def active(self) -> bool:
        return bool(self._active)

    async def set_active(self, active: bool, *, source: str | None = None, meta: dict[str, Any] | None = None) -> None:
        """
        Set service active state.

        - Persists `active` into KV under `nodes.<service_id>.state.active`
        - Notifies lifecycle listeners (engine/executor can pause/resume)
        """
        await self._apply_active(active, persist=True, source=source, meta=meta)

    def register_node(self, node: BusAttachableNode) -> None:
        node_id = ensure_token(node.node_id, label="node_id")
        self._nodes[node_id] = node
        node.attach(self)

    def unregister_node(self, node_id: str) -> None:
        node_id = ensure_token(node_id, label="node_id")
        node = self._nodes.pop(node_id, None)
        for key in [k for k in self._data_inputs.keys() if k[0] == node_id]:
            self._data_inputs.pop(key, None)
        if node is not None and isinstance(node, ClosableNode):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(node.close(), name=f"service_bus:close:{node_id}")
            except Exception:
                pass

    def get_node(self, node_id: str) -> BusAttachableNode | None:
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
        if self._micro is None:
            await self._start_micro_endpoints()
        if self._rungraph_watch is None:
            self._rungraph_watch = await self._transport.kv_watch(self._rungraph_key, cb=self._on_rungraph_kv)
        if self._local_state_watch is None:
            pattern = "nodes.>"
            self._local_state_watch = await self._transport.kv_watch(pattern, cb=self._on_local_state_kv)
        await self._load_active_from_kv()
        await self._reload_rungraph()

    async def stop(self) -> None:
        try:
            if self._micro is not None:
                await self._micro.stop()
        except Exception:
            pass
        self._micro = None

        for sub in list(self._raw_subs):
            try:
                await sub.unsubscribe()
            except Exception:
                pass
        self._raw_subs.clear()

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

    async def _load_active_from_kv(self) -> None:
        """
        Initialize `active` from KV (best-effort).
        """
        try:
            v = await self.get_state(self.service_id, "active")
        except Exception:
            v = None
        if v is None:
            return
        try:
            active = bool(v)
        except Exception:
            return
        await self._apply_active(active, persist=False, source="kv", meta={"init": True})

    async def _apply_active(
        self, active: bool, *, persist: bool, source: str | None, meta: dict[str, Any] | None
    ) -> None:
        try:
            active = bool(active)
        except Exception:
            active = True

        changed = active != self._active
        self._active = active

        if persist:
            try:
                await self.set_state_with_meta(
                    self.service_id,
                    "active",
                    bool(active),
                    source=source or "runtime",
                    meta={"lifecycle": True, **(dict(meta or {}))},
                )
            except Exception:
                pass

        if not changed:
            return

        payload = {"source": str(source or "runtime"), **(dict(meta or {}))}
        for cb in list(self._lifecycle_listeners):
            try:
                r = cb(bool(active), payload)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                continue

    async def _start_micro_endpoints(self) -> None:
        """
        Start NATS micro endpoints for built-in lifecycle control.

        Built-ins:
        - `svc.<serviceId>.activate`
        - `svc.<serviceId>.deactivate`
        - `svc.<serviceId>.set_active`
        - `svc.<serviceId>.status`

        Reserved custom command channel:
        - `svc.<serviceId>.cmd` (JSON envelope: reqId/call/args/meta)
        """
        nc = await self._transport.require_client()
        self._micro = await add_service(
            nc,
            ServiceConfig(
                name=svc_micro_name(self.service_id),
                version="0.0.1",
                description="F8 service runtime control plane (lifecycle + cmd).",
                metadata={"serviceId": self.service_id},
            ),
        )

        async def _respond(req: Any, *, req_id: str, ok: bool, result: Any = None, error: dict[str, Any] | None = None) -> None:
            payload = {"reqId": req_id, "ok": bool(ok), "result": result if ok else None, "error": error if not ok else None}
            await req.respond(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))

        def _parse_envelope(data: bytes) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
            req: dict[str, Any] = {}
            if data:
                try:
                    req = json.loads(data.decode("utf-8"))
                except Exception:
                    req = {}
            if not isinstance(req, dict):
                req = {}
            req_id = str(req.get("reqId") or "") or new_id()
            args = req.get("args") if isinstance(req.get("args"), dict) else {}
            meta = req.get("meta") if isinstance(req.get("meta"), dict) else {}
            return req_id, req, dict(args), dict(meta)

        async def _set_active_req(req: Any, active: bool, *, cmd: str) -> None:
            req_id, _raw, args, meta = _parse_envelope(req.data)
            want_active = bool(active)

            service_node = self.get_node(self.service_id)
            if service_node is not None and isinstance(service_node, CommandableNode):
                try:
                    await service_node.on_command("activate", {"active": want_active}, meta={"cmd": cmd, **meta})  # type: ignore[misc]
                except Exception as exc:
                    await _respond(req, req_id=req_id, ok=False, error={"code": "FORBIDDEN", "message": str(exc)})
                    return

            await self.set_active(want_active, source="cmd", meta={"cmd": cmd, **meta})
            await _respond(req, req_id=req_id, ok=True, result={"active": self.active})

        async def _activate(req: Any) -> None:
            await _set_active_req(req, True, cmd="activate")

        async def _deactivate(req: Any) -> None:
            await _set_active_req(req, False, cmd="deactivate")

        async def _set_active(req: Any) -> None:
            req_id, raw, args, meta = _parse_envelope(req.data)
            want_active = args.get("active")
            if want_active is None:
                want_active = raw.get("active")
            if want_active is None:
                await _respond(req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing active"})
                return
            await _set_active_req(req, bool(want_active), cmd="set_active")

        async def _status(req: Any) -> None:
            req_id, _raw, _args, _meta = _parse_envelope(req.data)
            await _respond(req, req_id=req_id, ok=True, result={"serviceId": self.service_id, "active": self.active})

        async def _cmd(req: Any) -> None:
            req_id, raw, args, meta = _parse_envelope(req.data)
            call = str(raw.get("call") or "").strip()
            if not call:
                await _respond(req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing call"})
                return
            service_node = self.get_node(self.service_id)
            if service_node is None or not isinstance(service_node, CommandableNode):
                await _respond(req, req_id=req_id, ok=False, error={"code": "UNKNOWN_CALL", "message": f"unknown call: {call}"})
                return
            try:
                out = await service_node.on_command(call, args, meta=meta)  # type: ignore[misc]
            except Exception as exc:
                await _respond(req, req_id=req_id, ok=False, error={"code": "INTERNAL", "message": str(exc)})
                return
            await _respond(req, req_id=req_id, ok=True, result=out)

        sid = self.service_id
        await self._micro.add_endpoint(
            EndpointConfig(name="activate", subject=svc_endpoint_subject(sid, "activate"), handler=_activate, metadata={"builtin": "true"})
        )
        await self._micro.add_endpoint(
            EndpointConfig(name="deactivate", subject=svc_endpoint_subject(sid, "deactivate"), handler=_deactivate, metadata={"builtin": "true"})
        )
        await self._micro.add_endpoint(
            EndpointConfig(name="set_active", subject=svc_endpoint_subject(sid, "set_active"), handler=_set_active, metadata={"builtin": "true"})
        )
        await self._micro.add_endpoint(
            EndpointConfig(name="status", subject=svc_endpoint_subject(sid, "status"), handler=_status, metadata={"builtin": "true"})
        )
        await self._micro.add_endpoint(
            EndpointConfig(name="cmd", subject=cmd_channel_subject(sid), handler=_cmd, metadata={"builtin": "false"})
        )

    # ---- raw subscription ---------------------------------------------
    async def subscribe_subject(
        self,
        subject: str,
        *,
        queue: str | None = None,
        cb: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> Any:
        """
        Subscribe to an arbitrary NATS subject (not tied to rungraph routing).

        Returns the subscription handle (must be unsubscribed by caller or via `unsubscribe_subject`).
        """
        subject = str(subject or "").strip()
        if not subject:
            raise ValueError("subject must be non-empty")
        handle = await self._transport.subscribe(subject, queue=str(queue) if queue else None, cb=cb)
        self._raw_subs.append(handle)
        return handle

    async def unsubscribe_subject(self, handle: Any) -> None:
        if handle is None:
            return
        try:
            await handle.unsubscribe()
        except Exception:
            return
        try:
            if handle in self._raw_subs:
                self._raw_subs.remove(handle)
        except Exception:
            pass

    # ---- KV state -------------------------------------------------------
    async def set_state(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        node_id = ensure_token(node_id, label="node_id")
        key = kv_key_node_state(node_id=node_id, field=str(field))
        payload = {"value": value, "actor": self.service_id, "ts": int(ts_ms or now_ms())}
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
        payload: dict[str, Any] = {"value": value, "actor": self.service_id, "ts": int(ts_ms or now_ms())}
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

        # Keep runtime lifecycle in sync with service node state (if changed externally).
        if str(node_id) == str(self.service_id) and str(field) == "active":
            try:
                await self._apply_active(bool(v), persist=False, source="kv", meta={"kvUpdate": True, **meta_dict})
            except Exception:
                pass

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
            if isinstance(node, StatefulNode):
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
        try:
            for n in list(getattr(graph, "nodes", None) or []):
                # Service/container nodes use `nodeId == serviceId`.
                if getattr(n, "operatorClass", None) is None and str(getattr(n, "nodeId", "")) != str(getattr(n, "serviceId", "")):
                    raise ValueError("invalid rungraph: service node requires nodeId == serviceId")
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
        ts = int(ts_ms or now_ms())

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
        _now_ms = now_ms()

        last_seen_ts = int(buf.last_seen_ts or _now_ms)
        if self._is_stale(edge, last_seen_ts):
            return None

        strategy = getattr(edge, "strategy", None) if edge is not None else None
        if not isinstance(strategy, F8EdgeStrategyEnum):
            strategy = F8EdgeStrategyEnum.latest

        if strategy == F8EdgeStrategyEnum.queue:
            if not buf.queue:
                if ctx_id is None or buf.last_seen_ctx_id != ctx_id:
                    await self._ensure_input_available(node_id=node_id, port=port, ctx_id=ctx_id)
                if not buf.queue:
                    return None
            v, ts = buf.queue.pop(0)
            buf.last_pulled_value = v
            buf.last_pulled_ts = int(ts) if ts is not None else _now_ms
            buf.last_pulled_ctx_id = ctx_id
            return v

        # latest
        if not buf.queue and (ctx_id is None or buf.last_seen_ctx_id != ctx_id):
            await self._ensure_input_available(node_id=node_id, port=port, ctx_id=ctx_id)
        v = buf.queue[-1][0] if buf.queue else buf.last_seen_value
        buf.queue.clear()
        if v is not None:
            buf.last_pulled_value = v
            buf.last_pulled_ts = _now_ms
            buf.last_pulled_ctx_id = ctx_id
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
                    if isinstance(src, ComputableNode):
                        v = await src.compute_output(str(from_port), ctx_id=ctx_id)
                    else:
                        v = None
                except Exception:
                    continue
                if v is None:
                    continue
                # Treat pull-triggered computation as producing a real output sample:
                # route it through `emit_data` so intra edges get buffered and any
                # cross-service subscribers can also receive the computed value.
                try:
                    await self.emit_data(str(from_node), str(from_port), v, ts_ms=now_ms())
                except Exception:
                    # Fallback: still satisfy the local pull.
                    self._buffer_input(
                        str(node_id),
                        str(port),
                        v,
                        ts_ms=now_ms(),
                        edge=edge,
                        ctx_id=ctx_id,
                        notify=False,
                    )
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

        ts_i = int(ts) if ts is not None else now_ms()
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
            return (now_ms() - int(ts_ms)) > t
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
            ctx_id=None,
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
        ctx_id: str | int | None,
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

        buf.last_seen_value = value
        buf.last_seen_ts = int(ts_ms)
        buf.last_seen_ctx_id = ctx_id

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
            ts = int(payload.get("ts") or now_ms())
        else:
            v = payload
            ts = now_ms()

        for local_node_id, local_field, _edge in targets:
            node = self._nodes.get(local_node_id)
            if node is None:
                continue
            try:
                if isinstance(node, StatefulNode):
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
