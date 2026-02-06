from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast

from nats.js.api import StorageType  # type: ignore[import-not-found]
from nats.micro import ServiceConfig, add_service  # type: ignore[import-not-found]
from nats.micro.service import EndpointConfig  # type: ignore[import-not-found]

from .capabilities import (
    BusAttachableNode,
    ClosableNode,
    CommandableNode,
    ComputableNode,
    DataReceivableNode,
    LifecycleNode,
    RungraphHook,
    ServiceHook,
    StatefulNode,
)
from .generated import F8Edge, F8EdgeKindEnum, F8EdgeStrategyEnum, F8RuntimeGraph, F8StateAccess
from .nats_naming import (
    cmd_channel_subject,
    data_subject,
    ensure_token,
    kv_key_ready,
    kv_bucket_for_service,
    kv_key_node_state,
    kv_key_rungraph,
    new_id,
    svc_endpoint_subject,
    svc_micro_name,
)
from .nats_transport import NatsTransport, NatsTransportConfig
from .state_write import StateWriteContext, StateWriteError, StateWriteOrigin
from .time_utils import now_ms


log = logging.getLogger(__name__)

DataDeliveryMode: TypeAlias = Literal["pull", "push", "both"]


def _debug_state_enabled() -> bool:
    return str(os.getenv("F8_STATE_DEBUG", "")).lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ServiceBusConfig:
    service_id: str
    service_name: str | None = None
    service_class: str | None = None
    nats_url: str = "nats://127.0.0.1:4222"
    publish_all_data: bool = True
    kv_storage: StorageType = StorageType.MEMORY
    delete_bucket_on_start: bool = False
    delete_bucket_on_stop: bool = False
    data_delivery: DataDeliveryMode = "pull"


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


@dataclass
class _StateUpdate:
    node_id: str
    field: str
    value: Any
    ts_ms: int
    origin: StateWriteOrigin
    source: str
    actor: str
    meta: dict[str, Any]


class _ServiceBusNode(StatefulNode, BusAttachableNode, Protocol):
    """
    Local-only node contract for ServiceBus registration.
    """


class _ServiceBusMicroEndpoints:
    def __init__(self, bus: "ServiceBus") -> None:
        self._bus = bus
        self._micro: Any | None = None

    async def start(self) -> Any:
        nc = await self._bus._transport.require_client()
        service_name = str(self._bus._service_name or "") or self._bus.service_id
        service_class = str(self._bus._service_class or "")
        description_parts = [f"serviceName={service_name}", f"serviceId={self._bus.service_id}"]
        if service_class:
            description_parts.insert(1, f"serviceClass={service_class}")
        description = "F8 service runtime control plane (%s)." % ", ".join(description_parts)
        metadata: dict[str, Any] = {"serviceId": self._bus.service_id, "serviceName": service_name}
        if service_class:
            metadata["serviceClass"] = service_class
        self._micro = await add_service(
            nc,
            ServiceConfig(
                name=svc_micro_name(self._bus.service_id),
                version="0.0.1",
                description=description,
                metadata=metadata,
            ),
        )
        await self._register_endpoints()
        return self._micro

    async def stop(self) -> None:
        if self._micro is None:
            return
        try:
            await self._micro.stop()
        except Exception:
            pass
        self._micro = None

    def _parse_envelope(self, data: bytes) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
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

    async def _respond(
        self, req: Any, *, req_id: str, ok: bool, result: Any = None, error: dict[str, Any] | None = None
    ) -> None:
        payload = {
            "reqId": req_id,
            "ok": bool(ok),
            "result": result if ok else None,
            "error": error if not ok else None,
        }
        await req.respond(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))

    async def _set_active_req(self, req: Any, active: bool, *, cmd: str) -> None:
        req_id, _raw, _args, meta = self._parse_envelope(req.data)
        want_active = bool(active)
        await self._bus.set_active(want_active, source="cmd", meta={"cmd": cmd, **meta})
        await self._respond(req, req_id=req_id, ok=True, result={"active": self._bus.active})

    async def _activate(self, req: Any) -> None:
        await self._set_active_req(req, True, cmd="activate")

    async def _deactivate(self, req: Any) -> None:
        await self._set_active_req(req, False, cmd="deactivate")

    async def _set_active(self, req: Any) -> None:
        req_id, raw, args, _meta = self._parse_envelope(req.data)
        want_active = args.get("active")
        if want_active is None:
            want_active = raw.get("active")
        if want_active is None:
            await self._respond(
                req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing active"}
            )
            return
        await self._set_active_req(req, bool(want_active), cmd="set_active")

    async def _status(self, req: Any) -> None:
        req_id, _raw, _args, _meta = self._parse_envelope(req.data)
        await self._respond(
            req, req_id=req_id, ok=True, result={"serviceId": self._bus.service_id, "active": self._bus.active}
        )

    async def _terminate(self, req: Any) -> None:
        req_id, _raw, _args, meta = self._parse_envelope(req.data)
        try:
            log.info("terminate requested serviceId=%s meta=%s", self._bus.service_id, dict(meta or {}))
        except Exception:
            pass
        try:
            self._bus._terminate_event.set()
        except Exception:
            pass
        await self._respond(req, req_id=req_id, ok=True, result={"terminating": True})

    async def _cmd(self, req: Any) -> None:
        req_id, raw, args, meta = self._parse_envelope(req.data)
        call = str(raw.get("call") or "").strip()
        if not call:
            await self._respond(
                req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing call"}
            )
            return
        service_node = self._bus.get_node(self._bus.service_id)
        if service_node is None or not isinstance(service_node, CommandableNode):
            await self._respond(
                req, req_id=req_id, ok=False, error={"code": "UNKNOWN_CALL", "message": f"unknown call: {call}"}
            )
            return
        try:
            out = await service_node.on_command(call, args, meta=meta)  # type: ignore[misc]
        except Exception as exc:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INTERNAL", "message": str(exc)})
            return
        await self._respond(req, req_id=req_id, ok=True, result=out)

    async def _set_state(self, req: Any) -> None:
        req_id, raw, args, meta = self._parse_envelope(req.data)
        node_id = args.get("nodeId") or raw.get("nodeId")
        field = args.get("field") or raw.get("field")
        value = args.get("value") if "value" in args else raw.get("value")
        if node_id is None or field is None:
            await self._respond(
                req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing nodeId/field"}
            )
            return
        node_id_s = str(node_id).strip()
        field_s = str(field).strip()
        if not node_id_s or not field_s:
            await self._respond(
                req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "empty nodeId/field"}
            )
            return

        try:
            node_id_s = ensure_token(node_id_s, label="node_id")
        except Exception:
            await self._respond(
                req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "invalid nodeId"}
            )
            return

        source = "endpoint"
        user_meta = dict(meta)
        user_meta.pop("source", None)
        user_meta.pop("origin", None)
        try:
            await self._bus._publish_state(
                node_id_s,
                field_s,
                value,
                origin=StateWriteOrigin.external,
                source=source,
                meta={"via": "endpoint", **user_meta},
            )
        except StateWriteError as exc:
            await self._respond(
                req,
                req_id=req_id,
                ok=False,
                error={"code": exc.code, "message": exc.message, "details": exc.details},
            )
            return
        except Exception as exc:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INTERNAL", "message": str(exc)})
            return
        await self._respond(req, req_id=req_id, ok=True, result={"nodeId": node_id_s, "field": field_s})

    async def _set_rungraph(self, req: Any) -> None:
        req_id, raw, args, _meta = self._parse_envelope(req.data)
        graph_obj = args.get("graph") if isinstance(args.get("graph"), dict) else raw.get("graph")
        if graph_obj is None and isinstance(raw, dict):
            graph_obj = raw if "nodes" in raw and "edges" in raw else None
        if not isinstance(graph_obj, dict):
            await self._respond(
                req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing graph"}
            )
            return

        try:
            graph = F8RuntimeGraph.model_validate(graph_obj)
        except Exception as exc:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INVALID_RUNGRAPH", "message": str(exc)})
            return
        try:
            await self._bus._validate_rungraph_or_raise(graph)
        except Exception as exc:
            await self._respond(
                req, req_id=req_id, ok=False, error={"code": "FORBIDDEN_RUNGRAPH", "message": str(exc)}
            )
            return

        try:
            payload = graph.model_dump(mode="json", by_alias=True)
            meta_payload = dict(payload.get("meta") or {})
            meta_payload["ts"] = int(now_ms())
            payload["meta"] = meta_payload
            raw_bytes = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            await self._bus._transport.kv_put(self._bus._rungraph_key, raw_bytes)
            await self._bus._apply_rungraph_bytes(raw_bytes)
        except Exception as exc:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INTERNAL", "message": str(exc)})
            return
        await self._respond(req, req_id=req_id, ok=True, result={"graphId": str(getattr(graph, "graphId", "") or "")})

    async def _register_endpoints(self) -> None:
        micro = self._micro
        if micro is None:
            return
        sid = self._bus.service_id
        await micro.add_endpoint(
            EndpointConfig(
                name="activate",
                subject=svc_endpoint_subject(sid, "activate"),
                handler=self._activate,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="deactivate",
                subject=svc_endpoint_subject(sid, "deactivate"),
                handler=self._deactivate,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="set_active",
                subject=svc_endpoint_subject(sid, "set_active"),
                handler=self._set_active,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="status",
                subject=svc_endpoint_subject(sid, "status"),
                handler=self._status,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="terminate",
                subject=svc_endpoint_subject(sid, "terminate"),
                handler=self._terminate,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="quit",
                subject=svc_endpoint_subject(sid, "quit"),
                handler=self._terminate,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="cmd", subject=cmd_channel_subject(sid), handler=self._cmd, metadata={"builtin": "false"}
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="set_state",
                subject=svc_endpoint_subject(sid, "set_state"),
                handler=self._set_state,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="set_rungraph",
                subject=svc_endpoint_subject(sid, "set_rungraph"),
                handler=self._set_rungraph,
                metadata={"builtin": "true"},
            )
        )


class ServiceBus:
    """
    Service bus (clean, protocol-first).

    - Shared NATS connection (pub/sub + JetStream KV).
    - Rungraph updates are applied via micro endpoints.
    - Builds intra/cross routing tables for data edges.
    - Provides a shared state KV API for nodes.
    - Data edges are pull-based: consumers pull buffered inputs and may trigger
      intra-service computation via `compute_output(...)`.
    """

    def __init__(self, config: ServiceBusConfig, *, transport: NatsTransport | None = None) -> None:
        self.service_id = ensure_token(config.service_id, label="service_id")
        self._service_name = str(config.service_name or "") or self.service_id
        self._service_class = str(config.service_class or "")
        self._publish_all_data = bool(config.publish_all_data)
        self._debug_state = _debug_state_enabled()
        self._active = True
        data_delivery = str(config.data_delivery or "pull").strip().lower()
        if data_delivery not in ("pull", "push", "both"):
            if self._debug_state or log.isEnabledFor(logging.WARNING):
                try:
                    log.warning("Invalid data_delivery=%r; defaulting to 'pull'", data_delivery)
                except Exception:
                    pass
            data_delivery = "pull"
        self._data_delivery = cast(DataDeliveryMode, data_delivery)

        bucket = kv_bucket_for_service(self.service_id)
        if transport is None:
            self._transport = NatsTransport(
                NatsTransportConfig(
                    url=str(config.nats_url),
                    kv_bucket=str(bucket),
                    kv_storage=getattr(config, "kv_storage", None),
                    delete_bucket_on_connect=bool(getattr(config, "delete_bucket_on_start", False)),
                    delete_bucket_on_close=bool(getattr(config, "delete_bucket_on_stop", False)),
                )
            )
        else:
            self._transport = transport

        self._nodes: dict[str, _ServiceBusNode] = {}
        self._graph: F8RuntimeGraph | None = None

        self._rungraph_key = kv_key_rungraph()
        self._ready_key = kv_key_ready()
        self._micro_endpoints: _ServiceBusMicroEndpoints | None = None

        # Routing tables (data only).
        self._intra_data_out: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self._intra_data_in: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
        self._cross_in_by_subject: dict[str, list[tuple[str, str, F8Edge]]] = {}
        self._cross_out_subjects: dict[tuple[str, str], str] = {}
        self._data_inputs: dict[tuple[str, str], _InputBuffer] = {}

        # Intra-service state fanout (state edges within the same service).
        self._intra_state_out: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}

        # Cross-state binding (remote KV -> local node.on_state + local KV mirror).
        self._cross_state_in_by_key: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
        self._remote_state_watches: dict[tuple[str, str], Any] = {}
        self._cross_state_targets: set[tuple[str, str]] = set()
        self._cross_state_last_ts: dict[tuple[str, str], int] = {}

        self._state_cache: dict[tuple[str, str], tuple[Any, int]] = {}
        self._state_access_by_node_field: dict[tuple[str, str], F8StateAccess] = {}
        self._data_route_subs: dict[str, Any] = {}
        self._custom_subs: list[Any] = []

        self._rungraph_hooks: list[RungraphHook] = []
        self._service_hooks: list[ServiceHook] = []

        # Process-level termination request (set via `svc.<serviceId>.terminate`).
        # Service entrypoints may `await bus.wait_terminate()` to exit gracefully.
        self._terminate_event = asyncio.Event()

    async def wait_terminate(self) -> None:
        await self._terminate_event.wait()

    @staticmethod
    def _coerce_data_delivery(value: Any) -> DataDeliveryMode | None:
        s = str(value or "").strip().lower()
        if s in ("pull", "push", "both"):
            return cast(DataDeliveryMode, s)
        return None

    @staticmethod
    def _origin_allows_access(origin: StateWriteOrigin, access: F8StateAccess) -> bool:
        if origin == StateWriteOrigin.system:
            return True
        if origin == StateWriteOrigin.runtime:
            return access in (F8StateAccess.rw, F8StateAccess.ro, F8StateAccess.wo)
        if origin == StateWriteOrigin.rungraph:
            return access in (F8StateAccess.rw, F8StateAccess.wo)
        if origin == StateWriteOrigin.external:
            return access in (F8StateAccess.rw, F8StateAccess.wo)
        return False
 
    def set_data_delivery(self, value: Any, *, source: str = "service") -> None:
        """
        Update data delivery behavior at runtime (service-controlled).
        """
        mode = self._coerce_data_delivery(value)
        if mode is None:
            return
        if mode == self._data_delivery:
            return
        self._data_delivery = mode
        if self._debug_state:
            print(f"state_debug[{self.service_id}] data_delivery={mode} source={source}")

    def register_rungraph_hook(self, hook: RungraphHook) -> None:
        """
        Register a rungraph hook (called after validation + routing rebuild).
        """
        self._rungraph_hooks.append(hook)

    def unregister_rungraph_hook(self, hook: RungraphHook) -> None:
        reg = self._rungraph_hooks
        reg.remove(hook)
        

    def register_service_hook(self, hook: ServiceHook) -> None:
        """
        Register a service bus hook (ready/stop/activate/deactivate).
        """
        self._service_hooks.append(hook)

    def unregister_service_hook(self, hook: ServiceHook) -> None:
        reg = self._service_hooks
        reg.remove(hook)
        

    # ---- lifecycle ------------------------------------------------------
    @property
    def active(self) -> bool:
        return bool(self._active)

    async def set_active(self, active: bool, *, source: str | None = None, meta: dict[str, Any] | None = None) -> None:
        """
        Set service active state.

        - Persists `active` into KV under `nodes.<service_id>.state.active`
        - Notifies lifecycle nodes + service hooks (engine/executor can pause/resume)
        """
        await self._apply_active(active, persist=True, source=source, meta=meta)

    def register_node(self, node: _ServiceBusNode) -> None:
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

    def get_node(self, node_id: str) -> _ServiceBusNode | None:
        """
        Return the local runtime node instance if registered.
        """
        node_id = ensure_token(node_id, label="node_id")
        return self._nodes.get(node_id)

    async def start(self) -> None:
        # Reset termination latch for a fresh run.
        self._terminate_event = asyncio.Event()
        await self._transport.connect()
        # Clear any stale ready flag from a previous run as early as possible.
        await self._announce_ready(False, reason="starting")
        if self._micro_endpoints is None:
            await self._start_micro_endpoints()
        await self._notify_before_ready()
        await self._announce_ready(True, reason="start")
        await self._notify_after_ready()

    async def stop(self) -> None:
        await self._notify_before_stop()
        await self._announce_ready(False, reason="stop")

        endpoints = self._micro_endpoints
        if endpoints is not None:
            await endpoints.stop()
        self._micro_endpoints = None

        for sub in list(self._custom_subs):
            await sub.unsubscribe()
        self._custom_subs.clear()

        for sub in list(self._data_route_subs.values()):
            await sub.unsubscribe()
        self._data_route_subs.clear()

        self._cross_in_by_subject.clear()
        self._intra_data_out.clear()
        self._intra_data_in.clear()
        self._cross_out_subjects.clear()
        self._data_inputs.clear()

        self._cross_state_in_by_key.clear()
        for (_sid, _key), watch in list(self._remote_state_watches.items()):
            watcher, task = watch
            task.cancel()
            await watcher.stop()

        self._remote_state_watches.clear()

        self._state_cache.clear()

        await self._transport.close()
        await self._notify_after_stop()
        self._rungraph_hooks.clear()
        self._service_hooks.clear()

    async def _announce_ready(self, ready: bool, *, reason: str) -> None:
        payload = {
            "serviceId": self.service_id,
            "ready": bool(ready),
            "reason": str(reason or ""),
            "ts": int(now_ms()),
        }
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        await self._transport.kv_put(self._ready_key, raw)

    async def _notify_before_ready(self) -> None:
        for hook in list(self._service_hooks):
            try:
                r = hook.on_before_ready(self)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                continue

    async def _notify_after_ready(self) -> None:
        for hook in list(self._service_hooks):
            try:
                r = hook.on_after_ready(self)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                continue

    async def _notify_before_stop(self) -> None:
        for hook in list(self._service_hooks):
            try:
                r = hook.on_before_stop(self)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                continue

    async def _notify_after_stop(self) -> None:
        for hook in list(self._service_hooks):
            try:
                r = hook.on_after_stop(self)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                continue

    async def _apply_active(
        self, active: bool, *, persist: bool, source: str | None, meta: dict[str, Any] | None
    ) -> None:
        active = bool(active)

        changed = active != self._active
        self._active = active

        if persist:
            await self._publish_state(
                self.service_id,
                "active",
                bool(active),
                origin=StateWriteOrigin.runtime,
                source=source or "runtime",
                meta={"lifecycle": True, **(dict(meta or {}))},
            )

        if not changed:
            return

        payload = {"source": str(source or "runtime"), **(dict(meta or {}))}

        for node in list(self._nodes.values()):
            if not isinstance(node, LifecycleNode):
                continue
            r = node.on_lifecycle(bool(active), dict(payload))
            if asyncio.iscoroutine(r):
                await r

        for hook in list(self._service_hooks):
            try:
                if bool(active):
                    r = hook.on_activate(self, dict(payload))
                else:
                    r = hook.on_deactivate(self, dict(payload))
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
        self._micro_endpoints = _ServiceBusMicroEndpoints(self)
        await self._micro_endpoints.start()

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
        self._custom_subs.append(handle)
        return handle

    async def unsubscribe_subject(self, handle: Any) -> None:
        if handle is None:
            return
        
        await handle.unsubscribe()

        if handle in self._custom_subs:
            self._custom_subs.remove(handle)
        
    # ---- KV state -------------------------------------------------------
    @staticmethod
    def _build_state_payload(update: _StateUpdate) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": update.value,
            "actor": update.actor,
            "ts": int(update.ts_ms),
            "source": update.source,
            "origin": update.origin.value,
        }
        for k, v in dict(update.meta or {}).items():
            if k in ("value", "actor", "ts", "source", "origin"):
                continue
            payload[k] = v
        return payload

    async def _publish_state(
        self,
        node_id: str,
        field: str,
        value: Any,
        *,
        origin: StateWriteOrigin,
        ts_ms: int | None = None,
        source: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        node_id = ensure_token(node_id, label="node_id")
        field = str(field)
        ts = int(ts_ms or now_ms())
        ctx = StateWriteContext(origin=origin, source=source)
        update = _StateUpdate(
            node_id=node_id,
            field=field,
            value=value,
            ts_ms=ts,
            origin=ctx.origin,
            source=ctx.resolved_source,
            actor=self.service_id,
            meta=dict(meta or {}),
        )
        payload = self._build_state_payload(update)
        update.value = await self._validate_state_update(
            node_id=node_id,
            field=field,
            value=payload.get("value"),
            ts_ms=int(payload.get("ts") or now_ms()),
            meta=dict(payload),
            ctx=ctx,
        )
        update.value = self._coerce_state_value(update.value)
        payload = self._build_state_payload(update)

        key = kv_key_node_state(node_id=node_id, field=field)
        if self._debug_state:
            print(
                "state_debug[%s] publish_state node=%s field=%s ts=%s origin=%s source=%s"
                % (self.service_id, node_id, field, str(payload.get("ts")), ctx.origin.value, update.source)
            )
        await self._transport.kv_put(key, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        self._state_cache[(node_id, field)] = (update.value, int(payload["ts"]))
        # Local writes (actor == self.service_id) do not round-trip through the KV watcher.
        # Apply to listeners and the node callback immediately.
        await self._deliver_state_local(node_id, field, update.value, int(payload["ts"]), dict(payload))

    async def publish_state_runtime(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        await self._publish_state(
            node_id,
            field,
            value,
            origin=StateWriteOrigin.runtime,
            source="runtime",
            ts_ms=ts_ms,
        )

    async def apply_state_local(
        self,
        node_id: str,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Apply a state update locally (cache + listeners + node callback) without writing to KV.

        This is useful for endpoint-only / fan-in paths where we want UI updates but do not
        want to persist synthetic state fields (eg. aggregated/fan-in keys) to the local bucket.
        """
        node_id = ensure_token(node_id, label="node_id")
        field = str(field)
        ts = int(ts_ms or now_ms())
        self._state_cache[(node_id, field)] = (value, ts)
        meta_dict = dict(meta or {})
        meta_dict.setdefault("actor", self.service_id)
        meta_dict.setdefault("ts", ts)
        meta_dict.setdefault("value", value)
        meta_dict.setdefault("source", str(meta_dict.get("source") or "local"))
        meta_dict.setdefault("origin", str(meta_dict.get("origin") or meta_dict.get("source") or "local"))
        await self._deliver_state_local(node_id, field, value, ts, meta_dict)

    async def _validate_state_update(
        self,
        *,
        node_id: str,
        field: str,
        value: Any,
        ts_ms: int,
        meta: dict[str, Any] | None,
        ctx: StateWriteContext,
    ) -> Any:
        """
        Centralized state validation hook.

        If a node implements `validate_state(field, value, ts_ms=..., meta=...)`, it may:
        - return a (possibly transformed) value to accept
        - raise StateWriteError/ValueError to reject
        """
        node = self._nodes.get(str(node_id))
        allow_unknown = bool(node.allow_unknown_state_fields) if node is not None else False

        access = self._state_access_by_node_field.get((str(node_id), str(field)))
        # If we have an applied graph, unknown fields are rejected by default.
        # Nodes may opt into dynamic fields (eg. fan-in aggregators) via `allow_unknown_state_fields=True`.
        if self._graph is not None and access is None and not allow_unknown:
            raise StateWriteError(
                "UNKNOWN_FIELD",
                f"unknown state field: {node_id}.{field}",
                details={"nodeId": str(node_id), "field": str(field)},
            )

        # Enforce write access when known.
        if access is not None and not self._origin_allows_access(ctx.origin, access):
            raise StateWriteError(
                "FORBIDDEN",
                f"state field not writable: {node_id}.{field} ({access.value})",
                details={
                    "nodeId": str(node_id),
                    "field": str(field),
                    "access": access.value,
                    "origin": ctx.origin.value,
                },
            )

        if node is None:
            return value
        try:
            meta_dict = dict(meta or {})
            meta_dict.setdefault("origin", ctx.origin.value)
            meta_dict.setdefault("source", ctx.resolved_source)
            r = node.validate_state(str(field), value, ts_ms=int(ts_ms), meta=meta_dict)
            if asyncio.iscoroutine(r):
                return await r
            return r
        except StateWriteError:
            raise
        except ValueError as exc:
            raise StateWriteError("INVALID_VALUE", str(exc)) from exc
        except Exception as exc:
            raise StateWriteError("INVALID_VALUE", str(exc)) from exc

    @staticmethod
    def _coerce_state_value(value: Any) -> Any:
        """
        Best-effort conversion of state values into JSON-friendly primitives.

        This prevents accidental persistence of pydantic RootModel/BaseModel objects
        as strings like "root=0.5", which then breaks runtime numeric coercion.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [ServiceBus._coerce_state_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): ServiceBus._coerce_state_value(v) for k, v in value.items()}

        # Enum-like objects.
        try:
            import enum

            if isinstance(value, enum.Enum):
                return ServiceBus._coerce_state_value(value.value)
        except Exception:
            pass

        # Pydantic v2 models/root models.
        try:
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                dumped = model_dump(mode="json")
                return ServiceBus._coerce_state_value(dumped)
        except Exception:
            pass

        # Generic RootModel-like `root` attribute.
        try:
            if hasattr(value, "root"):
                return ServiceBus._coerce_state_value(getattr(value, "root"))
        except Exception:
            pass

        return value

    @staticmethod
    def _coerce_inbound_ts_ms(ts_raw: Any, *, default: int) -> int:
        """
        Best-effort coercion of inbound timestamps to milliseconds.

        We accept a few common cases:
        - missing/invalid -> `default`
        - seconds since epoch -> convert to ms (heuristic by magnitude)
        - microseconds/nanoseconds since epoch -> downscale to ms (heuristic)
        """
        try:
            if ts_raw is None:
                return int(default)
            # Handle float seconds (common in ad-hoc scripts).
            if isinstance(ts_raw, float):
                ts = int(ts_raw)
            elif isinstance(ts_raw, str):
                ts = int(ts_raw.strip() or "0")
            else:
                ts = int(ts_raw)
        except Exception:
            return int(default)

        if ts <= 0:
            return int(default)

        # Heuristic: epoch seconds are ~1e9, epoch ms are ~1e12 (2026).
        # Use a conservative threshold to avoid misclassifying historical ms.
        if ts < 100_000_000_000:
            # Interpret as seconds.
            return int(ts * 1000)

        # Heuristic: epoch microseconds ~1e15, nanoseconds ~1e18.
        if ts >= 100_000_000_000_000_000:
            return int(ts // 1_000_000)
        if ts >= 100_000_000_000_000:
            return int(ts // 1000)

        return int(ts)

    @staticmethod
    def _extract_ts_field(payload: dict[str, Any]) -> Any:
        # Back-compat: tolerate alternative keys used by ad-hoc writers.
        if "ts" in payload:
            return payload.get("ts")
        if "ts_ms" in payload:
            return payload.get("ts_ms")
        if "tsMs" in payload:
            return payload.get("tsMs")
        return None

    async def _deliver_state_local(
        self, node_id: str, field: str, value: Any, ts_ms: int, meta_dict: dict[str, Any]
    ) -> None:
        """
        Apply a local (in-process) state update to listeners and to the node's `on_state`.

        This is used for local writes where KV watcher callbacks are skipped (self-echo).
        """
        node = self._nodes.get(node_id)
        if node is None:
            return
        await node.on_state(field, value, ts_ms=ts_ms)

        # Intra-service state fanout via state edges (local -> local).
        await self._route_state_edges(node_id=node_id, field=field, value=value, ts_ms=ts_ms, meta_dict=meta_dict)

    async def _route_state_edges(
        self, *, node_id: str, field: str, value: Any, ts_ms: int, meta_dict: dict[str, Any]
    ) -> None:
        if bool(meta_dict.get("_noStateFanout")):
            return
        targets = self._intra_state_out.get((str(node_id), str(field))) or []
        if not targets:
            return
        try:
            hops = int(meta_dict.get("_fanoutHops") or 0)
        except Exception:
            hops = 0
        if hops >= 8:
            return
        for to_node, to_field, _edge in list(targets):
            if str(to_node) == str(node_id) and str(to_field) == str(field):
                continue
            access = self._state_access_by_node_field.get((str(to_node), str(to_field)))
            if access not in (F8StateAccess.rw, F8StateAccess.wo):
                continue
            try:
                await self._publish_state(
                    str(to_node),
                    str(to_field),
                    value,
                    ts_ms=ts_ms,
                    origin=StateWriteOrigin.external,
                    source="state_edge",
                    meta={"fromNodeId": str(node_id), "fromField": str(field), "_fanoutHops": hops + 1},
                )
            except Exception:
                continue

    async def _apply_rungraph_state_values(self, graph: F8RuntimeGraph) -> None:
        """
        Materialize per-node `stateValues` into KV (and dispatch locally).

        The studio compiler stores UI property values into `stateValues` so
        runtimes can start with the same configuration. We write them into KV
        so:
          - downstream nodes (and UIs) can observe state immediately
          - intra-service state edges can fanout correctly
        """
        for n in list(getattr(graph, "nodes", None) or []):
            if str(getattr(n, "serviceId", "")) != self.service_id:
                continue
            node_id = str(getattr(n, "nodeId", "") or "").strip()
            if not node_id:
                continue
            values = getattr(n, "stateValues", None) or {}
            if not isinstance(values, dict) or not values:
                continue
            for k, v in list(values.items()):
                field = str(k or "").strip()
                if not field:
                    continue
                access = self._state_access_by_node_field.get((node_id, field))
                if access not in (F8StateAccess.rw, F8StateAccess.wo):
                    continue
                if (node_id, field) in self._cross_state_targets:
                    continue
                try:
                    await self._publish_state(
                        node_id,
                        field,
                        v,
                        origin=StateWriteOrigin.rungraph,
                        source="rungraph",
                        meta={"via": "rungraph", "_noStateFanout": True},
                    )
                except Exception:
                    continue

    async def _initial_sync_intra_state_edges(self, graph: F8RuntimeGraph) -> None:
        """
        Best-effort initial sync for intra-service state edges.

        If a state edge is added but the upstream value does not change after
        deploy, we still want the downstream field to receive the current value.
        """
        for edge in list(getattr(graph, "edges", None) or []):
            if getattr(edge, "kind", None) != F8EdgeKindEnum.state:
                continue
            if (
                str(getattr(edge, "fromServiceId", "")) != self.service_id
                or str(getattr(edge, "toServiceId", "")) != self.service_id
            ):
                continue
            if not edge.fromOperatorId or not edge.toOperatorId:
                continue
            from_node = str(edge.fromOperatorId)
            from_field = str(edge.fromPort)
            to_node = str(edge.toOperatorId)
            to_field = str(edge.toPort)
            access = self._state_access_by_node_field.get((to_node, to_field))
            if access not in (F8StateAccess.rw, F8StateAccess.wo):
                continue
            try:
                found_from, v_from, _ts_from = await self.get_state_with_ts(from_node, from_field)
            except Exception:
                continue
            if not found_from:
                continue
            try:
                found_to, v_to, _ts_to = await self.get_state_with_ts(to_node, to_field)
            except Exception:
                found_to, v_to = False, None
            # A state edge means "downstream follows upstream". Even if the
            # downstream has a local/default value, we still apply the upstream
            # value once on deploy so the edge has an effect without requiring
            # a subsequent upstream change.
            if found_to and v_to == v_from:
                continue
            try:
                await self._publish_state(
                    to_node,
                    to_field,
                    v_from,
                    origin=StateWriteOrigin.external,
                    source="state_edge_init",
                    meta={"fromNodeId": from_node, "fromField": from_field, "_fanoutHops": 1},
                )
            except Exception:
                continue

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
            ts = self._coerce_inbound_ts_ms(self._extract_ts_field(payload), default=0)
            self._state_cache[(node_id, field)] = (v, ts)
            if self._debug_state:
                print(
                    "state_debug[%s] get_state kv node=%s field=%s ts=%s" % (self.service_id, node_id, field, str(ts))
                )
            return v
        self._state_cache[(node_id, field)] = (payload, 0)
        return payload

    async def get_state_with_ts(self, node_id: str, field: str) -> tuple[bool, Any, int | None]:
        """
        Return (found, value, ts_ms) for a state key.

        Unlike `get_state`, this can distinguish between "missing" and a stored
        `None` value because the KV entry payload always includes metadata.
        """
        node_id = ensure_token(node_id, label="node_id")
        field = str(field)
        cached = self._state_cache.get((node_id, field))
        if cached is not None:
            return True, cached[0], cached[1]
        key = kv_key_node_state(node_id=node_id, field=field)
        raw = await self._transport.kv_get(key)
        if not raw:
            if self._debug_state:
                print("state_debug[%s] get_state_with_ts miss node=%s field=%s" % (self.service_id, node_id, field))
            return False, None, None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._state_cache[(node_id, field)] = (raw, 0)
            return True, raw, 0
        if isinstance(payload, dict) and "value" in payload:
            v = payload.get("value")
            ts = self._coerce_inbound_ts_ms(self._extract_ts_field(payload), default=0)
            self._state_cache[(node_id, field)] = (v, ts)
            if self._debug_state:
                print(
                    "state_debug[%s] get_state_with_ts kv node=%s field=%s ts=%s"
                    % (self.service_id, node_id, field, str(ts))
                )
            return True, v, ts
        self._state_cache[(node_id, field)] = (payload, 0)
        return True, payload, 0

    # ---- rungraph -------------------------------------------------------
    async def set_rungraph(self, graph: F8RuntimeGraph) -> None:
        """
        Publish a full rungraph snapshot for this service.
        """
        payload = graph.model_dump(mode="json", by_alias=True)
        meta = dict(payload.get("meta") or {})
        meta["ts"] = int(now_ms())
        payload["meta"] = meta
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        await self._transport.kv_put(self._rungraph_key, raw)
        # Endpoint-only mode: apply immediately (no KV watch).
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
            await self._validate_rungraph_or_raise(graph)
        except Exception:
            return
        try:
            for n in list(getattr(graph, "nodes", None) or []):
                # Service/container nodes use `nodeId == serviceId`.
                if getattr(n, "operatorClass", None) is None and str(getattr(n, "nodeId", "")) != str(
                    getattr(n, "serviceId", "")
                ):
                    raise ValueError("invalid rungraph: service node requires nodeId == serviceId")
        except Exception:
            return

        self._graph = graph
        # Reset cross-state ordering on graph changes.
        self._cross_state_last_ts.clear()
        # Cache local node state access for enforcement and filtering.
        self._state_access_by_node_field.clear()
        try:
            for n in list(getattr(graph, "nodes", None) or []):
                if str(getattr(n, "serviceId", "") or "") != self.service_id:
                    continue
                node_id = str(getattr(n, "nodeId", "") or "")
                if not node_id:
                    continue
                for sf in list(getattr(n, "stateFields", None) or []):
                    name = str(getattr(sf, "name", "") or "").strip()
                    if not name:
                        continue
                    access = getattr(sf, "access", None)
                    if isinstance(access, F8StateAccess):
                        self._state_access_by_node_field[(node_id, name)] = access
        except Exception:
            self._state_access_by_node_field.clear()
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
        await self._apply_rungraph_state_values(graph)
        await self._seed_builtin_identity_state(graph)
        await self._initial_sync_intra_state_edges(graph)

        for hook in list(self._rungraph_hooks):
            try:
                r = hook.on_rungraph(graph)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                continue
        # Cross-state watches and initial sync are intentionally installed AFTER
        # rungraph hooks so local runtime nodes are registered first. This avoids
        # dropping initial remote values due to missing nodes/fields.
        await self._sync_cross_state_watches()

    async def _seed_builtin_identity_state(self, graph: F8RuntimeGraph) -> None:
        """
        Seed readonly identity fields (`svcId`, `operatorId`) into KV for local nodes.

        These fields are declared as `ro` in specs, so they cannot be set via
        rungraph `stateValues` or external `set_state`. We write them directly
        to KV (without dispatching to nodes) so UIs and cross-service consumers
        can discover identities for routing/commands.
        """
        ts = int(now_ms())
        for n in list(getattr(graph, "nodes", None) or []):
            if str(getattr(n, "serviceId", "")) != self.service_id:
                continue
            node_id = str(getattr(n, "nodeId", "") or "").strip()
            if not node_id:
                continue
            try:
                if self._state_access_by_node_field.get((node_id, "svcId")) is not None:
                    await self._put_state_kv_unvalidated(
                        node_id=node_id,
                        field="svcId",
                        value=str(getattr(n, "serviceId", "") or self.service_id),
                        ts_ms=ts,
                        meta={"source": "system", "origin": "system", "builtin": True},
                    )
                # `operatorId` is only meaningful for operator nodes (not service/container nodes).
                if (
                    getattr(n, "operatorClass", None) is not None
                    and self._state_access_by_node_field.get((node_id, "operatorId")) is not None
                ):
                    await self._put_state_kv_unvalidated(
                        node_id=node_id,
                        field="operatorId",
                        value=str(getattr(n, "nodeId", "") or node_id),
                        ts_ms=ts,
                        meta={"source": "system", "origin": "system", "builtin": True},
                    )
            except Exception:
                continue

    async def _put_state_kv_unvalidated(
        self, *, node_id: str, field: str, value: Any, ts_ms: int, meta: dict[str, Any] | None
    ) -> None:
        """
        Write state to KV without access validation and without dispatching to nodes.

        This is reserved for runtime-owned fields that are declared `ro` but must
        exist in KV for discovery/monitoring.
        """
        node_id = ensure_token(node_id, label="node_id")
        field = str(field or "").strip()
        if not field:
            return
        ts = int(ts_ms or now_ms())
        value = self._coerce_state_value(value)
        key = kv_key_node_state(node_id=node_id, field=field)
        payload = {"value": value, "actor": self.service_id, "ts": ts, **(dict(meta or {}))}
        await self._transport.kv_put(key, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        self._state_cache[(node_id, field)] = (value, int(ts))

    async def _validate_rungraph_or_raise(self, graph: F8RuntimeGraph) -> None:
        """
        Validate the rungraph before applying it.

        Raises to reject illegal graphs.
        """
        # Build access map from the candidate graph (do not rely on currently-applied graph).
        access_map: dict[tuple[str, str], F8StateAccess] = {}
        for n in list(getattr(graph, "nodes", None) or []):
            if str(getattr(n, "serviceId", "")) != self.service_id:
                continue
            node_id = str(getattr(n, "nodeId", "") or "")
            if not node_id:
                continue
            for sf in list(getattr(n, "stateFields", None) or []):
                name = str(getattr(sf, "name", "") or "").strip()
                if not name:
                    continue
                a = getattr(sf, "access", None)
                if isinstance(a, F8StateAccess):
                    access_map[(node_id, name)] = a

        # Basic local invariants.
        for n in list(getattr(graph, "nodes", None) or []):
            if str(getattr(n, "serviceId", "")) != self.service_id:
                continue
            node_id = str(getattr(n, "nodeId", "") or "")
            if not node_id:
                raise ValueError("missing nodeId")
            access_by_name: dict[str, F8StateAccess] = {}
            for sf in list(getattr(n, "stateFields", None) or []):
                name = str(getattr(sf, "name", "") or "").strip()
                if not name:
                    continue
                a = getattr(sf, "access", None)
                if isinstance(a, F8StateAccess):
                    access_by_name[name] = a

            values = getattr(n, "stateValues", None) or {}
            if isinstance(values, dict):
                for k in list(values.keys()):
                    key = str(k)
                    a = access_by_name.get(key)
                    if a is None:
                        raise ValueError(f"unknown state value: {node_id}.{key}")
                    if a == F8StateAccess.ro:
                        raise ValueError(f"read-only state cannot be set by rungraph: {node_id}.{key}")

        # Cross-state edges must not target read-only fields.
        for e in list(getattr(graph, "edges", None) or []):
            if getattr(e, "kind", None) != F8EdgeKindEnum.state:
                continue
            if str(getattr(e, "toServiceId", "")) != self.service_id:
                continue
            to_node = str(getattr(e, "toOperatorId", "") or "")
            to_field = str(getattr(e, "toPort", "") or "")
            if not to_node or not to_field:
                continue
            a = access_map.get((to_node, to_field))
            if a == F8StateAccess.ro:
                raise ValueError(f"state edge targets non-writable field: {to_node}.{to_field} ({a.value})")

        # Custom validators (service-specific).
        for hook in list(self._rungraph_hooks):
            try:
                r = hook.validate_rungraph(graph)
                if asyncio.iscoroutine(r):
                    await r
            except Exception as exc:
                raise ValueError(str(exc)) from exc

    async def _rebuild_routes(self) -> None:
        graph = self._graph
        if graph is None:
            return

        self._data_inputs.clear()
        self._intra_state_out.clear()

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

        # Intra-service state fanout: local state edges (used for fan-in/UI reflection).
        intra_state_out: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
        for edge in graph.edges:
            if edge.kind != F8EdgeKindEnum.state:
                continue
            if str(edge.fromServiceId) != self.service_id or str(edge.toServiceId) != self.service_id:
                continue
            if not edge.fromOperatorId or not edge.toOperatorId:
                continue
            intra_state_out.setdefault((str(edge.fromOperatorId), str(edge.fromPort)), []).append(
                (str(edge.toOperatorId), str(edge.toPort), edge)
            )
        self._intra_state_out = intra_state_out

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
        self._update_cross_state_bindings(graph)
        await self._stop_unused_cross_state_watches()

    def _update_cross_state_bindings(self, graph: F8RuntimeGraph) -> None:
        """
        Build cross-state binding tables from the current graph without starting watches.

        Watches are started after rungraph hooks register nodes (see `_apply_rungraph_bytes`).
        """
        want: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
        targets: set[tuple[str, str]] = set()
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
            targets.add((local_node, local_field))

        self._cross_state_in_by_key = want
        self._cross_state_targets = targets

    async def _stop_unused_cross_state_watches(self) -> None:
        """
        Stop KV watches that are no longer needed based on current bindings.
        """
        want = self._cross_state_in_by_key
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

    # ---- data routing ---------------------------------------------------
    async def emit_data(self, node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if not self._active:
            return
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
        if not self._active:
            return
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
        if not self._active:
            return None
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
                    )
        finally:
            stack.discard(key)

    async def _on_cross_data_msg(self, subject: str, payload: bytes) -> None:
        if not self._active:
            return
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
        # the system can be configured for pull-based or push-based data delivery.
        self._buffer_input(
            to_node=str(to_node),
            to_port=str(to_port),
            value=value,
            ts_ms=int(ts_ms),
            edge=edge,
            ctx_id=None,
        )
        if self._data_delivery in ("push", "both"):
            node = self._nodes.get(str(to_node))
            if node is not None:
                try:
                    if isinstance(node, DataReceivableNode):
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            node.on_data(str(to_port), value, ts_ms=int(ts_ms)),  # type: ignore[misc]
                            name=f"service_bus:on_data:{to_node}:{to_port}",
                        )
                except Exception:
                    pass

    def _buffer_input(
        self,
        to_node: str,
        to_port: str,
        value: Any,
        *,
        ts_ms: int,
        edge: F8Edge | None,
        ctx_id: str | int | None,
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

        return

    # ---- cross-state ----------------------------------------------------
    async def _sync_cross_state_watches(self) -> None:
        """
        Cross-state binding via remote KV watch (read remote, apply to local).
        """
        # NOTE: bindings are computed in `_rebuild_routes` so `stateValues` application
        # can skip cross-state target fields.
        want = self._cross_state_in_by_key

        # Start missing watches and perform best-effort initial sync for every watched key.
        for peer, remote_key in want.keys():
            bucket = kv_bucket_for_service(peer)

            if (peer, remote_key) not in self._remote_state_watches:
                async def _cb(key: str, val: bytes, *, _peer: str = peer) -> None:
                    await self._on_remote_state_kv(_peer, key, val, is_initial=False)

                self._remote_state_watches[(peer, remote_key)] = await self._transport.kv_watch_in_bucket(
                    bucket, remote_key, cb=_cb
                )

            # Initial sync: fetch current value once so new targets receive the current value
            # even if the upstream doesn't change after deploy.
            try:
                raw = await self._transport.kv_get_in_bucket(bucket, remote_key)
                if raw:
                    await self._on_remote_state_kv(peer, remote_key, raw, is_initial=True)
            except Exception:
                pass

    async def _on_remote_state_kv(self, peer_service_id: str, key: str, value: bytes, *, is_initial: bool) -> None:
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
            ts = self._coerce_inbound_ts_ms(self._extract_ts_field(payload), default=now_ms())
        else:
            v = payload
            ts = now_ms()

        if self._debug_state:
            try:
                v_s = repr(v)
                if len(v_s) > 160:
                    v_s = v_s[:157] + "..."
                print(
                    "state_debug[%s] cross_state_in peer=%s key=%s ts=%s initial=%s value=%s targets=%s"
                    % (
                        self.service_id,
                        str(peer_service_id),
                        str(key),
                        str(ts),
                        "1" if bool(is_initial) else "0",
                        v_s,
                        str(len(targets)),
                    )
                )
            except Exception:
                pass

        for local_node_id, local_field, _edge in targets:
            access = self._state_access_by_node_field.get((str(local_node_id), str(local_field)))
            if access == F8StateAccess.ro:
                continue
            try:
                meta_in = payload if isinstance(payload, dict) else {}
                try:
                    hops = int(meta_in.get("_fanoutHops") or 0) + 1
                except Exception:
                    hops = 1
                if hops >= 8:
                    continue

                # Cross-service state edges are directional bindings: downstream follows
                # upstream. We only guard against out-of-order remote updates.
                try:
                    last_ts = self._cross_state_last_ts.get((str(local_node_id), str(local_field)))
                    if not is_initial and last_ts is not None and int(ts) < int(last_ts):
                        if self._debug_state:
                            try:
                                print(
                                    "state_debug[%s] cross_state_skip_old_remote node=%s field=%s ts_last=%s ts_remote=%s peer=%s key=%s"
                                    % (
                                        self.service_id,
                                        str(local_node_id),
                                        str(local_field),
                                        str(last_ts),
                                        str(ts),
                                        str(peer_service_id),
                                        str(key),
                                    )
                                )
                            except Exception:
                                pass
                        continue
                except Exception:
                    pass

                meta_out = {
                    "peerServiceId": str(peer_service_id),
                    "remoteKey": str(key),
                    "_fanoutHops": hops,
                    **{k: vv for k, vv in dict(meta_in).items() if k not in ("value", "actor", "ts", "source")},
                }
                v2 = await self._validate_state_update(
                    node_id=str(local_node_id),
                    field=str(local_field),
                    value=v,
                    ts_ms=int(ts),
                    meta={"source": "cross_state", **meta_out},
                    ctx=StateWriteContext(origin=StateWriteOrigin.external, source="cross_state"),
                )
                v2 = self._coerce_state_value(v2)

                # Skip duplicates (common when KV watch emits the current value and we
                # also perform an explicit initial `kv_get`).
                try:
                    cached = self._state_cache.get((str(local_node_id), str(local_field)))
                    if cached is not None and int(ts) <= int(cached[1]) and cached[0] == v2:
                        if self._debug_state:
                            try:
                                print(
                                    "state_debug[%s] cross_state_skip_duplicate to=%s.%s ts=%s peer=%s remote_key=%s"
                                    % (
                                        self.service_id,
                                        str(local_node_id),
                                        str(local_field),
                                        str(ts),
                                        str(peer_service_id),
                                        str(key),
                                    )
                                )
                            except Exception:
                                pass
                        continue
                except Exception:
                    pass

                if access is None:
                    await self.apply_state_local(
                        str(local_node_id),
                        str(local_field),
                        v2,
                        ts_ms=ts,
                        meta={"source": "cross_state", "origin": "external", **meta_out},
                    )
                else:
                    await self._publish_state(
                        str(local_node_id),
                        str(local_field),
                        v2,
                        ts_ms=ts,
                        origin=StateWriteOrigin.external,
                        source="cross_state",
                        meta=meta_out,
                    )
                if self._debug_state:
                    try:
                        v2s = repr(v2)
                        if len(v2s) > 160:
                            v2s = v2s[:157] + "..."
                        print(
                            "state_debug[%s] cross_state_apply to=%s.%s ts=%s peer=%s remote_key=%s value=%s"
                            % (
                                self.service_id,
                                str(local_node_id),
                                str(local_field),
                                str(ts),
                                str(peer_service_id),
                                str(key),
                                v2s,
                            )
                        )
                    except Exception:
                        pass
                try:
                    self._cross_state_last_ts[(str(local_node_id), str(local_field))] = int(ts)
                except Exception:
                    pass
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
        for subject in list(self._data_route_subs.keys()):
            if subject in want_subjects:
                continue
            sub = self._data_route_subs.pop(subject, None)
            if sub is None:
                continue
            try:
                await sub.unsubscribe()
            except Exception:
                pass

        for subject in want_subjects:
            if subject in self._data_route_subs:
                continue

            async def _cb(s: str, p: bytes) -> None:
                await self._on_cross_data_msg(s, p)

            handle = await self._transport.subscribe(subject, cb=_cb)
            self._data_route_subs[subject] = handle
