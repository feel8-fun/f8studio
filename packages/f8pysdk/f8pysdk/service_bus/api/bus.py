from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, Generic, Protocol, TYPE_CHECKING, TypeVar

from ...capabilities import (
    BusAttachableNode,
    ClosableNode,
    RungraphHook,
    ServiceHook,
    StatefulNode,
)
from ...generated import F8Edge, F8RuntimeGraph, F8StateAccess
from ...nats_naming import (
    ensure_token,
    kv_key_ready,
    kv_bucket_for_service,
    kv_key_node_state,
    kv_key_rungraph,
)
from ...nats_transport import NatsTransport, NatsTransportConfig
from ..payload import coerce_inbound_ts_ms, extract_ts_field
from ..state_publish import (
    publish_state as _publish_state_impl,
)
from ..routing_data import _InputBuffer
from ..routing_data import (
    emit_data as _emit_data_impl,
    pull_data as _pull_data_impl,
    subscribe_subject as _subscribe_subject_impl,
    unsubscribe_subject as _unsubscribe_subject_impl,
)
from ..rungraph_apply import (
    set_rungraph as _set_rungraph_impl,
)
from ..state_write import StateWriteOrigin, StateWriteSource
from ..lifecycle import (
    set_active as _set_active_impl,
    start as _start_impl,
    stop as _stop_impl,
)
from ..state_read import StateRead
from .config import DataDeliveryMode, ServiceBusConfig, _debug_state_enabled

if TYPE_CHECKING:
    from ..micro import _ServiceBusMicroEndpoints


log = logging.getLogger(__name__)

_K = TypeVar("_K")
_V = TypeVar("_V")


def _coerce_data_delivery_mode(value: Any) -> DataDeliveryMode | None:
    text = str(value or "").strip().lower()
    if text in ("pull", "push", "both"):
        return text
    return None


class _CappedOrderedDict(OrderedDict[_K, _V], Generic[_K, _V]):
    """
    Ordered mapping with max-entry cap.

    - `get` and `__getitem__` refresh recency.
    - `__setitem__` enforces max size.
    """

    def __init__(self, *, max_entries: int) -> None:
        super().__init__()
        self._max_entries = max(0, int(max_entries))

    def __getitem__(self, key: _K) -> _V:
        value = super().__getitem__(key)
        super().move_to_end(key)
        return value

    def get(self, key: _K, default: _V | None = None) -> _V | None:
        if key in self:
            return self[key]
        return default

    def __setitem__(self, key: _K, value: _V) -> None:
        exists = key in self
        super().__setitem__(key, value)
        if exists:
            super().move_to_end(key)
        if self._max_entries > 0:
            while len(self) > self._max_entries:
                self.popitem(last=False)


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
        mode = _coerce_data_delivery_mode(config.data_delivery)
        if mode is None:
            if self._debug_state or log.isEnabledFor(logging.WARNING):
                log.warning("Invalid data_delivery=%r; defaulting to 'pull'", config.data_delivery)
            mode = "pull"
        self._data_delivery = mode
        self._state_sync_concurrency = max(1, int(config.state_sync_concurrency))
        self._state_cache_max_entries = max(0, int(config.state_cache_max_entries))
        self._data_input_max_buffers = max(0, int(config.data_input_max_buffers))
        self._data_input_default_queue_size = max(1, int(config.data_input_default_queue_size))

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
        self._data_inputs: _CappedOrderedDict[tuple[str, str], _InputBuffer] = _CappedOrderedDict(
            max_entries=self._data_input_max_buffers
        )

        # Intra-service state fanout (state edges within the same service).
        self._intra_state_out: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}

        # Cross-state binding (remote KV -> local node.on_state + local KV mirror).
        self._cross_state_in_by_key: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
        self._remote_state_watches: dict[tuple[str, str], Any] = {}
        self._cross_state_targets: set[tuple[str, str]] = set()
        self._cross_state_last_ts: dict[tuple[str, str], int] = {}

        self._state_cache: _CappedOrderedDict[tuple[str, str], tuple[Any, int]] = _CappedOrderedDict(
            max_entries=self._state_cache_max_entries
        )
        self._state_access_by_node_field: dict[tuple[str, str], F8StateAccess] = {}
        self._data_route_subs: dict[str, Any] = {}
        self._custom_subs: list[Any] = []

        self._rungraph_hooks: list[RungraphHook] = []
        self._service_hooks: list[ServiceHook] = []

        # Error dedupe for rungraph apply boundaries.
        self._rungraph_apply_error_once: set[str] = set()
        # Generic error dedupe for high-frequency paths (watchers/fanout/loops).
        self._error_once: set[str] = set()

        # Process-level termination request (set via `svc.<serviceId>.terminate`).
        # Service entrypoints may `await bus.wait_terminate()` to exit gracefully.
        self._terminate_event = asyncio.Event()

    async def wait_terminate(self) -> None:
        await self._terminate_event.wait()

    def set_data_delivery(self, value: Any, *, source: str = "service") -> None:
        """
        Update data delivery behavior at runtime (service-controlled).
        """
        mode = _coerce_data_delivery_mode(value)
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
            except Exception as exc:
                log.debug("failed to schedule node close node_id=%s", node_id, exc_info=exc)

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

    async def publish_state_external(
        self,
        node_id: str,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        source: StateWriteSource | str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Publish a state update as an external/user write.

        This method intentionally does not allow callers to choose `origin`.
        `source` is allowed for diagnostics, but does not affect access control.
        """
        await _publish_state_impl(
            self,
            node_id,
            field,
            value,
            ts_ms=ts_ms,
            origin=StateWriteOrigin.external,
            source=source or StateWriteSource.endpoint,
            meta=dict(meta or {}),
        )

    async def publish_state_runtime(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        await _publish_state_impl(
            self,
            node_id,
            field,
            value,
            origin=StateWriteOrigin.runtime,
            source=StateWriteSource.runtime,
            ts_ms=ts_ms,
        )

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
            ts = coerce_inbound_ts_ms(extract_ts_field(payload), default=0)
            self._state_cache[(node_id, field)] = (v, ts)
            if self._debug_state:
                print(
                    "state_debug[%s] get_state kv node=%s field=%s ts=%s" % (self.service_id, node_id, field, str(ts))
                )
            return StateRead(found=True, value=v, ts_ms=ts)

        self._state_cache[(node_id, field)] = (payload, 0)
        return StateRead(found=True, value=payload, ts_ms=0)

    # ---- rungraph -------------------------------------------------------
    async def set_rungraph(self, graph: F8RuntimeGraph) -> None:
        await _set_rungraph_impl(self, graph)

    # ---- data routing ---------------------------------------------------
    async def publish(self, subject: str, payload: bytes) -> None:
        """Publish a message to a subject."""
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
        """Subscribe to a subject."""
        return await self._transport.subscribe(str(subject), queue=queue, cb=cb)

    async def emit_data(self, node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        await _emit_data_impl(self, node_id, port, value, ts_ms=ts_ms)

    async def pull_data(self, node_id: str, port: str, *, ctx_id: str | int | None = None) -> Any:
        return await _pull_data_impl(self, node_id, port, ctx_id=ctx_id)
