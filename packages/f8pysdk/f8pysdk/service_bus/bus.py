from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast

from nats.js.api import StorageType  # type: ignore[import-not-found]

from ..capabilities import (
    BusAttachableNode,
    ClosableNode,
    RungraphHook,
    ServiceHook,
    StatefulNode,
)
from ..generated import F8Edge, F8RuntimeGraph, F8StateAccess
from ..nats_naming import (
    ensure_token,
    kv_key_ready,
    kv_bucket_for_service,
    kv_key_node_state,
    kv_key_rungraph,
)
from ..nats_transport import NatsTransport, NatsTransportConfig
from .micro import _ServiceBusMicroEndpoints
from .payload import coerce_inbound_ts_ms, extract_ts_field, parse_state_key
from .cross_state import (
    on_remote_state_kv as _on_remote_state_kv_impl,
    stop_unused_cross_state_watches as _stop_unused_cross_state_watches_impl,
    sync_cross_state_watches as _sync_cross_state_watches_impl,
    update_cross_state_bindings as _update_cross_state_bindings_impl,
)
from .state_publish import (
    coerce_state_value as _coerce_state_value,
    publish_state as _publish_state_impl,
    publish_state_runtime as _publish_state_runtime_impl,
    validate_state_update as _validate_state_update_impl,
)
from .routing_data import _InputBuffer
from .routing_data import (
    buffer_input as _buffer_input_impl,
    compute_and_buffer_for_input as _compute_and_buffer_for_input_impl,
    emit_data as _emit_data_impl,
    ensure_input_available as _ensure_input_available_impl,
    is_stale as _is_stale_impl,
    on_cross_data_msg as _on_cross_data_msg_impl,
    pull_data as _pull_data_impl,
    push_input as _push_input_impl,
    subscribe_subject as _subscribe_subject_impl,
    sync_subscriptions as _sync_subscriptions_impl,
    unsubscribe_subject as _unsubscribe_subject_impl,
)
from .rungraph_apply import (
    apply_rungraph_bytes as _apply_rungraph_bytes_impl,
    apply_rungraph_state_values as _apply_rungraph_state_values_impl,
    initial_sync_intra_state_edges as _initial_sync_intra_state_edges_impl,
    rebuild_routes as _rebuild_routes_impl,
    seed_builtin_identity_state as _seed_builtin_identity_state_impl,
    set_rungraph as _set_rungraph_impl,
    validate_rungraph_or_raise as _validate_rungraph_or_raise_impl,
)
from .state_write import StateWriteContext, StateWriteError, StateWriteOrigin, StateWriteSource
from .lifecycle import (
    announce_ready as _announce_ready_impl,
    apply_active as _apply_active_impl,
    notify_after_ready as _notify_after_ready_impl,
    notify_after_stop as _notify_after_stop_impl,
    notify_before_ready as _notify_before_ready_impl,
    notify_before_stop as _notify_before_stop_impl,
    set_active as _set_active_impl,
    start as _start_impl,
    stop as _stop_impl,
)
from .state_read import StateRead


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


class _ServiceBusNode(StatefulNode, BusAttachableNode, Protocol):
    """
    Local-only node contract for ServiceBus registration.
    """

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
                    kv_storage=config.kv_storage,
                    delete_bucket_on_connect=bool(config.delete_bucket_on_start),
                    delete_bucket_on_close=bool(config.delete_bucket_on_stop),
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

    async def set_active(
        self,
        active: bool,
        *,
        source: StateWriteSource | str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        await _set_active_impl(self, active, source=source, meta=meta)

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
        await _start_impl(self)

    async def stop(self) -> None:
        await _stop_impl(self)

    async def _announce_ready(self, ready: bool, *, reason: str) -> None:
        await _announce_ready_impl(self, ready, reason=reason)

    async def _notify_before_ready(self) -> None:
        await _notify_before_ready_impl(self)

    async def _notify_after_ready(self) -> None:
        await _notify_after_ready_impl(self)

    async def _notify_before_stop(self) -> None:
        await _notify_before_stop_impl(self)

    async def _notify_after_stop(self) -> None:
        await _notify_after_stop_impl(self)

    async def _apply_active(
        self, active: bool, *, persist: bool, source: StateWriteSource | str | None, meta: dict[str, Any] | None
    ) -> None:
        await _apply_active_impl(self, active, persist=persist, source=source, meta=meta)

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
        return await _subscribe_subject_impl(self, subject, queue=queue, cb=cb)

    async def unsubscribe_subject(self, handle: Any) -> None:
        await _unsubscribe_subject_impl(self, handle)
    # ---- KV state -------------------------------------------------------

    async def _publish_state(
        self,
        node_id: str,
        field: str,
        value: Any,
        *,
        origin: StateWriteOrigin,
        ts_ms: int | None = None,
        source: StateWriteSource | str | None = None,
        meta: dict[str, Any] | None = None,
        deliver_local: bool = True,
    ) -> None:
        await _publish_state_impl(
            self,
            node_id,
            field,
            value,
            origin=origin,
            ts_ms=ts_ms,
            source=source,
            meta=meta,
            deliver_local=deliver_local,
        )

    async def publish_state_runtime(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        await _publish_state_runtime_impl(self, node_id, field, value, ts_ms=ts_ms)

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
        return await _validate_state_update_impl(
            self,
            node_id=node_id,
            field=field,
            value=value,
            ts_ms=ts_ms,
            meta=meta,
            ctx=ctx,
        )

    @staticmethod
    def _coerce_state_value(value: Any) -> Any:
        return _coerce_state_value(value)

    @staticmethod
    def _coerce_inbound_ts_ms(ts_raw: Any, *, default: int) -> int:
        return coerce_inbound_ts_ms(ts_raw, default=int(default))

    @staticmethod
    def _extract_ts_field(payload: dict[str, Any]) -> Any:
        return extract_ts_field(payload)

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
                    source=StateWriteSource.state_edge_intra,
                    meta={"fromNodeId": str(node_id), "fromField": str(field)},
                )
            except Exception:
                continue

    async def _apply_rungraph_state_values(self, graph: F8RuntimeGraph) -> None:
        await _apply_rungraph_state_values_impl(self, graph)

    async def _initial_sync_intra_state_edges(self, graph: F8RuntimeGraph) -> None:
        await _initial_sync_intra_state_edges_impl(self, graph)

    async def get_state(self, node_id: str, field: str) -> StateRead:
        node_id = ensure_token(node_id, label="node_id")
        field = str(field)

        cached = self._state_cache.get((node_id, field))
        if cached is not None:
            return StateRead(found=True, value=cached[0], ts_ms=cached[1])

        key = kv_key_node_state(node_id=node_id, field=field)
        raw = await self._transport.kv_get(key)
        if not raw:
            if self._debug_state:
                print("state_debug[%s] get_state miss node=%s field=%s" % (self.service_id, node_id, field))
            return StateRead(found=False, value=None, ts_ms=None)

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            # Preserve old `get_state_with_ts` behavior: treat unparseable values
            # as "found" and return raw bytes.
            self._state_cache[(node_id, field)] = (raw, 0)
            return StateRead(found=True, value=raw, ts_ms=0)

        if isinstance(payload, dict) and "value" in payload:
            v = payload.get("value")
            ts = self._coerce_inbound_ts_ms(self._extract_ts_field(payload), default=0)
            self._state_cache[(node_id, field)] = (v, ts)
            if self._debug_state:
                print("state_debug[%s] get_state kv node=%s field=%s ts=%s" % (self.service_id, node_id, field, str(ts)))
            return StateRead(found=True, value=v, ts_ms=ts)

        self._state_cache[(node_id, field)] = (payload, 0)
        return StateRead(found=True, value=payload, ts_ms=0)

    # ---- rungraph -------------------------------------------------------
    async def set_rungraph(self, graph: F8RuntimeGraph) -> None:
        await _set_rungraph_impl(self, graph)

    async def _apply_rungraph_bytes(self, raw: bytes) -> None:
        await _apply_rungraph_bytes_impl(self, raw)

    async def _seed_builtin_identity_state(self, graph: F8RuntimeGraph) -> None:
        await _seed_builtin_identity_state_impl(self, graph)

    async def _validate_rungraph_or_raise(self, graph: F8RuntimeGraph) -> None:
        await _validate_rungraph_or_raise_impl(self, graph)

    async def _rebuild_routes(self) -> None:
        await _rebuild_routes_impl(self)

    def _update_cross_state_bindings(self, graph: F8RuntimeGraph) -> None:
        _update_cross_state_bindings_impl(self, graph)

    async def _stop_unused_cross_state_watches(self) -> None:
        await _stop_unused_cross_state_watches_impl(self)

    # ---- data routing ---------------------------------------------------
    async def emit_data(self, node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        await _emit_data_impl(self, node_id, port, value, ts_ms=ts_ms)

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
        return await _pull_data_impl(self, node_id, port, ctx_id=ctx_id)

    async def _ensure_input_available(self, *, node_id: str, port: str, ctx_id: str | int | None = None) -> None:
        await _ensure_input_available_impl(self, node_id=node_id, port=port, ctx_id=ctx_id)

    async def _compute_and_buffer_for_input(
        self,
        *,
        node_id: str,
        port: str,
        ctx_id: str | int | None,
        stack: set[tuple[str, str]],
    ) -> None:
        await _compute_and_buffer_for_input_impl(
            self, node_id=node_id, port=port, ctx_id=ctx_id, stack=stack
        )

    async def _on_cross_data_msg(self, subject: str, payload: bytes) -> None:
        await _on_cross_data_msg_impl(self, subject, payload)

    @staticmethod
    def _is_stale(edge: F8Edge | None, ts_ms: int) -> bool:
        return _is_stale_impl(edge, ts_ms)

    def _push_input(self, to_node: str, to_port: str, value: Any, *, ts_ms: int, edge: F8Edge | None = None) -> None:
        _push_input_impl(self, to_node, to_port, value, ts_ms=ts_ms, edge=edge)

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
        _buffer_input_impl(self, to_node, to_port, value, ts_ms=ts_ms, edge=edge, ctx_id=ctx_id)

    # ---- cross-state ----------------------------------------------------
    async def _sync_cross_state_watches(self) -> None:
        await _sync_cross_state_watches_impl(self)

    async def _on_remote_state_kv(self, peer_service_id: str, key: str, value: bytes, *, is_initial: bool) -> None:
        await _on_remote_state_kv_impl(self, peer_service_id, key, value, is_initial=is_initial)

    @staticmethod
    def _parse_state_key(key: str) -> tuple[str, str] | None:
        return parse_state_key(key)

    # ---- subscriptions --------------------------------------------------
    async def _sync_subscriptions(self, want_subjects: set[str]) -> None:
        await _sync_subscriptions_impl(self, want_subjects)
