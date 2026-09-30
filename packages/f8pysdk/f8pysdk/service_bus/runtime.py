from __future__ import annotations

from ..generated import runtime_keys

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

from ..capabilities import (
    BusAttachableNode,
    ClosableNode,
    RungraphHook,
    ServiceHook,
    StatefulNode,
)
from ..codec import encode_obj
from ..command import CommandExecutionResult, CommandOutputPolicy
from ..data import CrossPublishPolicy, DataDeliveryMode
from ..generated import F8RuntimeGraph
from ..f8_naming import ensure_token
from ..rungraph_fingerprint import build_rungraph_deploy_fingerprint
from ..service_runtime_tools.deploy.readiness import rungraph_deploy_request_status_key
from ..runtime_transport import RuntimeTransport
from ..zenoh_transport import ZenohTransport, ZenohTransportConfig
from ..zenoh_naming import zenoh_data_key, zenoh_state_path_key
from ..state import StateRead, StateWriteOrigin, StateWriteSource
from ..time_utils import now_ms
from .config import ServiceBusConfig, _debug_state_enabled
from .data.router import DataRouter
from .internal.command import CommandGateway, CommandInvocation, CommandInvokeOptions
from ..monitoring import MonitorCollector, MonitorCollectorConfig
from .state.pipeline import publish_state as _publish_state_impl
from .state.router import StateRouter
from .state.store import StateStore
from .workflow.lifecycle import set_active as _set_active_impl
from .workflow.lifecycle import start as _start_impl
from .workflow.lifecycle import stop as _stop_impl
from .workflow.rungraph import set_rungraph as _set_rungraph_impl

if TYPE_CHECKING:
    from .internal.control_endpoints import ServiceControlEndpointServer
    from .state.options import StatePublishOptions


log = logging.getLogger(__name__)


def _coerce_cross_publish_policy(value: Any) -> CrossPublishPolicy | None:
    text = str(value or "").strip().lower()
    if text in ("routed", "all", "none"):
        return text
    return None


def _coerce_data_delivery_mode(value: Any) -> DataDeliveryMode | None:
    text = str(value or "").strip().lower()
    if text in ("buffered", "callback"):
        return text
    return None


class _ServiceBusNode(StatefulNode, BusAttachableNode, Protocol):
    """
    Local-only node contract for ServiceBus registration.
    """


class ServiceBusComponentFactory(Protocol):
    """
    Explicit component builder for `ServiceBus` owner subsystems.

    This keeps component wiring out of `ServiceBus.__init__` so tests and
    future alternate runtimes can supply focused implementations explicitly.
    """

    def create_data_router(
        self,
        *,
        bus: "ServiceBus",
        cross_publish_policy: CrossPublishPolicy,
        data_delivery: DataDeliveryMode,
        input_max_buffers: int,
        default_queue_size: int,
        output_debug_max_ports: int,
        output_debug_history_size: int,
    ) -> DataRouter: ...

    def create_state_store(
        self,
        *,
        bus: "ServiceBus",
        cache_max_entries: int,
    ) -> StateStore: ...

    def create_state_router(
        self,
        *,
        bus: "ServiceBus",
        store: StateStore,
    ) -> StateRouter: ...

    def create_command_gateway(
        self,
        *,
        bus: "ServiceBus",
        nodes: dict[str, _ServiceBusNode],
    ) -> CommandGateway: ...

    def create_monitor_collector(
        self,
        *,
        bus: "ServiceBus",
        config: MonitorCollectorConfig,
    ) -> MonitorCollector: ...


@dataclass(frozen=True)
class DefaultServiceBusComponentFactory:
    def create_data_router(
        self,
        *,
        bus: "ServiceBus",
        cross_publish_policy: CrossPublishPolicy,
        data_delivery: DataDeliveryMode,
        input_max_buffers: int,
        default_queue_size: int,
        output_debug_max_ports: int,
        output_debug_history_size: int,
    ) -> DataRouter:
        return DataRouter(
            bus,
            cross_publish_policy=cross_publish_policy,
            data_delivery=data_delivery,
            input_max_buffers=input_max_buffers,
            default_queue_size=default_queue_size,
            output_debug_max_ports=output_debug_max_ports,
            output_debug_history_size=output_debug_history_size,
        )

    def create_state_store(
        self,
        *,
        bus: "ServiceBus",
        cache_max_entries: int,
    ) -> StateStore:
        return StateStore(bus, cache_max_entries=cache_max_entries)

    def create_state_router(
        self,
        *,
        bus: "ServiceBus",
        store: StateStore,
    ) -> StateRouter:
        return StateRouter(bus, store=store)

    def create_command_gateway(
        self,
        *,
        bus: "ServiceBus",
        nodes: dict[str, _ServiceBusNode],
    ) -> CommandGateway:
        return CommandGateway(bus=bus, nodes=nodes)

    def create_monitor_collector(
        self,
        *,
        bus: "ServiceBus",
        config: MonitorCollectorConfig,
    ) -> MonitorCollector:
        return MonitorCollector(bus, config)


class ServiceBus:
    """
    Service bus (clean, protocol-first).

    - Shared RuntimeTransport connection (Zenoh by default; mem is for tests).
    - Rungraph, lifecycle, and command requests are applied via backend-neutral control endpoints.
    - Builds intra/cross routing tables for data edges.
    - Provides a latest-value state API backed by service-owned KV/state snapshots.
    - Local data delivery is configured explicitly as buffered or callback.
    - Pull-based consumers may trigger intra-service computation via `compute_output(...)`.
    """

    def __init__(
        self,
        config: ServiceBusConfig,
        *,
        transport: RuntimeTransport | None = None,
        component_factory: ServiceBusComponentFactory | None = None,
    ) -> None:
        config = config.normalized()
        self._config = config
        self._bus_backend = config.bus_backend
        self.service_id = ensure_token(config.service_id, label="service_id")
        self._runtime_instance_id = uuid.uuid4().hex
        self._service_name = str(config.service_name or "") or self.service_id
        self._service_class = str(config.service_class or "")
        self._debug_state = _debug_state_enabled()
        self._active = True
        self._ready = False
        cross_publish_policy = _coerce_cross_publish_policy(config.cross_publish_policy)
        if cross_publish_policy is None:
            raise ValueError(
                f"Invalid cross_publish_policy={config.cross_publish_policy!r}; expected 'routed', 'all', or 'none'."
            )
        mode = _coerce_data_delivery_mode(config.data_delivery)
        if mode is None:
            raise ValueError(f"Invalid data_delivery={config.data_delivery!r}; expected 'callback' or 'buffered'.")
        self._state_sync_concurrency = max(1, int(config.state_sync_concurrency))
        self._state_cache_max_entries = max(0, int(config.state_cache_max_entries))
        self._data_input_max_buffers = max(0, int(config.data_input_max_buffers))
        self._data_input_default_queue_size = max(1, int(config.data_input_default_queue_size))
        self._data_output_debug_max_ports = max(0, int(config.data_output_debug_max_ports))
        self._data_output_debug_history_size = max(1, min(int(config.data_output_debug_history_size), 128))

        if transport is None:
            if config.bus_backend == "zenoh":
                self._transport = ZenohTransport(
                    ZenohTransportConfig(
                        service_id=self.service_id,
                        runtime_instance_id=self._runtime_instance_id,
                        announce_service_liveliness=True,
                        config_path=config.zenoh_config_path,
                        connect=config.zenoh_connect,
                        listen=config.zenoh_listen,
                        shm_pool_bytes=config.zenoh_shm_pool_bytes,
                    )
                )
            elif config.bus_backend == "mem":
                from ..testing.in_memory_transport import InMemoryCluster, InMemoryTransport

                self._transport = InMemoryTransport(cluster=InMemoryCluster())
            else:
                raise ValueError(f"Invalid bus_backend={config.bus_backend!r}; expected 'zenoh' or 'mem'.")
        else:
            self._transport = transport

        self._nodes: dict[str, _ServiceBusNode] = {}
        self._graph: F8RuntimeGraph | None = None
        self._rungraph_fingerprint = ""

        self._rungraph_key = runtime_keys.rungraph(self.service_id)
        self._rungraph_status_key = runtime_keys.rungraph_status(self.service_id)
        self._ready_key = runtime_keys.ready(self.service_id)
        self._control_endpoints: ServiceControlEndpointServer | None = None
        self._component_factory = component_factory if component_factory is not None else DefaultServiceBusComponentFactory()

        self._data_router = self._component_factory.create_data_router(
            bus=self,
            cross_publish_policy=cross_publish_policy,
            data_delivery=mode,
            input_max_buffers=self._data_input_max_buffers,
            default_queue_size=self._data_input_default_queue_size,
            output_debug_max_ports=self._data_output_debug_max_ports,
            output_debug_history_size=self._data_output_debug_history_size,
        )
        self._state_store = self._component_factory.create_state_store(
            bus=self,
            cache_max_entries=self._state_cache_max_entries,
        )
        self._state_router = self._component_factory.create_state_router(bus=self, store=self._state_store)
        self._command_gateway = self._component_factory.create_command_gateway(bus=self, nodes=self._nodes)

        self._rungraph_hooks: list[RungraphHook] = []
        self._service_hooks: list[ServiceHook] = []

        # Error dedupe for rungraph apply boundaries.
        self._rungraph_apply_error_once: set[str] = set()
        # Generic error dedupe for high-frequency paths (watchers/fanout/loops).
        self._error_once: set[str] = set()
        self._state_publish_seq = 0
        self._rungraph_apply_lock = asyncio.Lock()
        self._rungraph_apply_tasks: set[asyncio.Task[None]] = set()
        self._rungraph_req_fingerprints: dict[str, str] = {}
        self._rungraph_inflight_aliases: dict[str, set[str]] = {}

        # Process-level termination request (set via `svc.<serviceId>.terminate`).
        # Service entrypoints may `await bus.wait_terminate()` to exit gracefully.
        self._terminate_event = asyncio.Event()
        self._monitor_collector = self._component_factory.create_monitor_collector(
            bus=self,
            config=MonitorCollectorConfig(
                enabled=bool(config.monitor_enabled),
                interval_ms=max(200, int(config.monitor_interval_ms)),
                window_ms=max(1000, int(config.monitor_window_ms)),
                gpu_enabled=bool(config.monitor_gpu_enabled),
            ),
        )
        self._exec_emitter: Callable[[str, str, str | int], Awaitable[None]] | None = None
        if self._monitor_collector.enabled:
            self._monitor_record_emit = self._record_emit_metrics_enabled
            self._monitor_record_wait = self._record_wait_metrics_enabled
            self._monitor_record_input = self._record_input_metrics_enabled
            self._monitor_record_drop = self._record_drop_metrics_enabled
            self._monitor_record_local_only_emit = self._record_local_only_emit_metrics_enabled
            self._monitor_record_routed_cross_emit = self._record_routed_cross_emit_metrics_enabled
            self._monitor_record_suppressed_cross_publish = self._record_suppressed_cross_publish_metrics_enabled
            self._monitor_record_callback_delivery = self._record_callback_delivery_metrics_enabled
            self._monitor_record_buffer_pull_delivery = self._record_buffer_pull_delivery_metrics_enabled
        else:
            self._monitor_record_emit = self._noop_record_emit
            self._monitor_record_wait = self._noop_record_wait
            self._monitor_record_input = self._noop_record_input
            self._monitor_record_drop = self._noop_record_drop
            self._monitor_record_local_only_emit = self._noop_record_local_only_emit
            self._monitor_record_routed_cross_emit = self._noop_record_routed_cross_emit
            self._monitor_record_suppressed_cross_publish = self._noop_record_suppressed_cross_publish
            self._monitor_record_callback_delivery = self._noop_record_callback_delivery
            self._monitor_record_buffer_pull_delivery = self._noop_record_buffer_pull_delivery

        self._started = False
        self._closed = False

    async def wait_terminate(self) -> None:
        await self._terminate_event.wait()

    def _next_state_publish_seq(self) -> int:
        self._state_publish_seq += 1
        return int(self._state_publish_seq)

    @staticmethod
    def _noop_record_emit(node_id: str, port: str, ts: int) -> None:
        del node_id, port, ts

    @staticmethod
    def _noop_record_wait(wait_ms: float) -> None:
        del wait_ms

    @staticmethod
    def _noop_record_input(node_id: str, port: str, ts: int) -> None:
        del node_id, port, ts

    @staticmethod
    def _noop_record_drop(dropped_count: int) -> None:
        del dropped_count

    @staticmethod
    def _noop_record_local_only_emit() -> None:
        return

    @staticmethod
    def _noop_record_routed_cross_emit() -> None:
        return

    @staticmethod
    def _noop_record_suppressed_cross_publish() -> None:
        return

    @staticmethod
    def _noop_record_callback_delivery() -> None:
        return

    @staticmethod
    def _noop_record_buffer_pull_delivery() -> None:
        return

    def _record_emit_metrics_enabled(self, node_id: str, port: str, ts: int) -> None:
        now_ts = int(now_ms())
        self._monitor_collector.record_processed(port=str(port), emit_ts_ms=int(ts), now_ts_ms=now_ts)
        self._monitor_collector.record_emit_completed(node_id=str(node_id), now_ts_ms=now_ts)

    def _record_wait_metrics_enabled(self, wait_ms: float) -> None:
        self._monitor_collector.record_wait_ms(wait_ms=wait_ms)

    def _record_input_metrics_enabled(self, node_id: str, port: str, ts: int) -> None:
        self._monitor_collector.record_observed(port=str(port))
        self._monitor_collector.record_input_sample_ts(node_id=str(node_id), sample_ts_ms=int(ts))

    def _record_drop_metrics_enabled(self, dropped_count: int) -> None:
        self._monitor_collector.record_dropped(dropped_count=int(dropped_count))

    def _record_local_only_emit_metrics_enabled(self) -> None:
        self._monitor_collector.record_local_only_emit()

    def _record_routed_cross_emit_metrics_enabled(self) -> None:
        self._monitor_collector.record_routed_cross_emit()

    def _record_suppressed_cross_publish_metrics_enabled(self) -> None:
        self._monitor_collector.record_suppressed_cross_publish()

    def _record_callback_delivery_metrics_enabled(self) -> None:
        self._monitor_collector.record_callback_delivery()

    def _record_buffer_pull_delivery_metrics_enabled(self) -> None:
        self._monitor_collector.record_buffer_pull_delivery()

    def record_monitor_processed(self, *, port: str, ts_ms: int | None = None) -> None:
        now_ts = int(ts_ms) if ts_ms is not None else int(now_ms())
        self._monitor_collector.record_processed(port=str(port), emit_ts_ms=0, now_ts_ms=now_ts)

    def record_monitor_timing(
        self,
        *,
        port: str,
        process_ms: float,
        latency_ms: float,
        ts_ms: int | None = None,
    ) -> None:
        self._monitor_collector.record_timing(
            port=str(port),
            process_ms=float(process_ms),
            latency_ms=float(latency_ms),
            ts_ms=ts_ms,
        )

    @property
    def service_name(self) -> str:
        return self._service_name

    @property
    def service_class(self) -> str:
        return self._service_class

    @property
    def runtime_instance_id(self) -> str:
        return self._runtime_instance_id

    @property
    def config(self) -> ServiceBusConfig:
        return self._config

    @property
    def bus_backend(self) -> str:
        return self._bus_backend

    @property
    def cross_publish_policy(self) -> CrossPublishPolicy:
        return self._data_router.cross_publish_policy

    @property
    def data_delivery(self) -> DataDeliveryMode:
        return self._data_router.data_delivery

    @property
    def data_router(self) -> DataRouter:
        return self._data_router

    @property
    def state_store(self) -> StateStore:
        return self._state_store

    @property
    def state_router(self) -> StateRouter:
        return self._state_router

    def set_cross_publish_policy(self, value: Any, *, source: str = "service") -> None:
        policy = _coerce_cross_publish_policy(value)
        if policy is None:
            return
        if policy == self.cross_publish_policy:
            return
        self._data_router.set_cross_publish_policy(policy)
        if self._debug_state:
            print(f"state_debug[{self.service_id}] cross_publish_policy={policy} source={source}")

    def set_data_delivery(self, value: Any, *, source: str = "service") -> None:
        """
        Update data delivery behavior at runtime (service-controlled).
        """
        mode = _coerce_data_delivery_mode(value)
        if mode is None:
            raise ValueError(f"Invalid data_delivery={value!r}; expected 'callback' or 'buffered'.")
        if mode == self.data_delivery:
            return
        self._data_router.set_data_delivery(mode)
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

    @property
    def active(self) -> bool:
        return bool(self._active)

    def has_rungraph(self) -> bool:
        return self._graph is not None

    @property
    def command_gateway(self) -> CommandGateway:
        return self._command_gateway

    @property
    def monitor_collector(self) -> MonitorCollector:
        return self._monitor_collector

    def report_error(
        self,
        node_id: str,
        code: str,
        message: str,
        severity: str = "error",
        fingerprint: str | None = None,
        ts_ms: int | None = None,
    ) -> None:
        self._monitor_collector.report_error(
            node_id=node_id,
            code=code,
            message=message,
            severity=severity,
            fingerprint=fingerprint,
            ts_ms=ts_ms,
        )

    def clear_error(self, node_id: str, fingerprint: str | None = None, ts_ms: int | None = None) -> None:
        self._monitor_collector.clear_error(node_id=node_id, fingerprint=fingerprint, ts_ms=ts_ms)

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
        if self._graph is not None:
            self._command_gateway.refresh_bindings()

    def detach_node(self, node_id: str) -> _ServiceBusNode | None:
        node_id = ensure_token(node_id, label="node_id")
        node = self._nodes.pop(node_id, None)
        self._data_router.remove_node_inputs(node_id)
        if self._graph is not None:
            self._command_gateway.refresh_bindings()
        return node

    def unregister_node(self, node_id: str) -> None:
        node = self.detach_node(node_id)
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
        if self._closed:
            raise RuntimeError("ServiceBus is not restartable after stop(); create a new instance")
        if self._started:
            return
        await _start_impl(self)
        self._started = True

    async def stop(self) -> None:
        if self._closed:
            return
        tasks = list(self._rungraph_apply_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._rungraph_apply_tasks.clear()
        self._rungraph_inflight_aliases.clear()
        await _stop_impl(self)
        self._started = False
        self._closed = True

    async def subscribe_key(
        self,
        key_expr: str,
        *,
        queue: str | None = None,
        cb: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> Any:
        return await self._data_router.subscribe_key(key_expr, queue=queue, cb=cb)

    async def unsubscribe_key(self, handle: Any) -> None:
        await self._data_router.unsubscribe_key(handle)

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

        Canonical semantics:
        - persistence: the value is validated and written to KV/state cache
        - local delivery: same-process consumers are notified immediately
        - fanout: intra-service state edges fan out unless typed publish options disable them
        - cross-service propagation: remote services observe the persisted value through state watches

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

    async def publish_state_runtime(
        self,
        node_id: str,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        force_publish: bool = False,
    ) -> None:
        """
        Publish a runtime-owned state update through the same validated/persisted state chain.

        Use this for SDK/runtime writes that should persist and participate in
        the normal local state delivery/fanout behavior.
        """
        await _publish_state_impl(
            self,
            node_id,
            field,
            value,
            origin=StateWriteOrigin.runtime,
            source=StateWriteSource.runtime,
            ts_ms=ts_ms,
            options=self._runtime_state_publish_options(force_publish=bool(force_publish)),
        )

    @staticmethod
    def _runtime_state_publish_options(*, force_publish: bool) -> "StatePublishOptions | None":
        if not force_publish:
            return None
        from .state.options import StatePublishOptions

        return StatePublishOptions(force_publish=True)

    async def invoke_command(
        self,
        node_id: str,
        call: str,
        args: Any = None,
        *,
        meta: dict[str, Any] | None = None,
        output_policy: CommandOutputPolicy = CommandOutputPolicy.none,
        output_ts_ms: int | None = None,
        output_meta: dict[str, Any] | None = None,
    ) -> CommandExecutionResult:
        """
        Invoke a local registered commandable node through the canonical SDK command path.

        - Declared commands accept scalar/list/dict inputs and normalize them by parameter definition.
        - Undeclared commands require object-shaped args because there is no schema for positional mapping.
        - Defaults to reply-first behavior without hidden output state writeback.
        - `output_policy` controls whether hidden command output state is also written back.
        - The return value is structured so callers can inspect failures without parsing logs.
        """
        node_id_s = ensure_token(node_id, label="node_id")
        call_s = str(call or "").strip()
        if not call_s:
            raise ValueError("call is empty")
        return await self._command_gateway.invoke(
            invocation=CommandInvocation(node_id=node_id_s, call=call_s, args=args),
            options=CommandInvokeOptions(
                call_meta=dict(meta or {}),
                output_policy=output_policy,
                output_ts_ms=output_ts_ms,
                output_meta=dict(output_meta or {}),
            ),
        )

    async def get_state(self, node_id: str, field: str) -> StateRead:
        return await self._state_store.read_state(node_id, field)

    def get_state_cached(self, node_id: str, field: str, default: Any = None) -> Any:
        """
        Synchronous cached state snapshot read without KV/network IO.
        """
        return self._state_store.get_cached_value(node_id, field, default)

    async def set_rungraph(self, graph: F8RuntimeGraph) -> None:
        await _set_rungraph_impl(self, graph)

    async def submit_rungraph(
        self,
        graph: F8RuntimeGraph,
        *,
        req_id: str,
        source: str = "control",
        target_fingerprint: str = "",
        force_apply: bool = False,
    ) -> None:
        """
        Accept a remote rungraph deployment request and apply it asynchronously.

        The control endpoint should return quickly after this method succeeds;
        deployment progress and final result are published to the retained
        rungraph status key.
        """
        req_id_s = str(req_id or "").strip()
        if not req_id_s:
            raise ValueError("req_id is empty")
        source_s = str(source or "control").strip() or "control"
        fingerprint = str(target_fingerprint or "").strip() or build_rungraph_deploy_fingerprint(graph)
        existing_fingerprint = self._rungraph_req_fingerprints.get(req_id_s)
        if existing_fingerprint is not None and existing_fingerprint != fingerprint:
            raise ValueError("req_id already used for a different rungraph fingerprint")
        self._rungraph_req_fingerprints[req_id_s] = fingerprint
        if not bool(force_apply) and self._rungraph_fingerprint and self._rungraph_fingerprint == fingerprint:
            self._schedule_rungraph_status_publish(
                graph,
                req_id=req_id_s,
                phase="applied",
                source=source_s,
                target_fingerprint=fingerprint,
                applied_fingerprint=fingerprint,
            )
            return
        aliases = self._rungraph_inflight_aliases.get(fingerprint)
        if aliases is not None:
            aliases.add(req_id_s)
            self._schedule_rungraph_status_publish(
                graph,
                req_id=req_id_s,
                phase="accepted",
                source=source_s,
                target_fingerprint=fingerprint,
            )
            return
        self._rungraph_inflight_aliases[fingerprint] = {req_id_s}
        self._schedule_rungraph_status_publish(
            graph,
            req_id=req_id_s,
            phase="accepted",
            source=source_s,
            target_fingerprint=fingerprint,
        )
        task = asyncio.create_task(
            self._rungraph_apply_worker(graph, target_fingerprint=fingerprint, source=source_s),
            name=f"service_bus:set_rungraph:{self.service_id}:{req_id_s}",
        )
        self._track_rungraph_task(task)

    def _track_rungraph_task(self, task: asyncio.Task[None]) -> None:
        self._rungraph_apply_tasks.add(task)
        task.add_done_callback(self._on_rungraph_apply_task_done)

    def _on_rungraph_apply_task_done(self, task: asyncio.Task[None]) -> None:
        self._rungraph_apply_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            log.error("rungraph apply task failed service_id=%s", self.service_id, exc_info=exc)

    async def _rungraph_apply_worker(self, graph: F8RuntimeGraph, *, target_fingerprint: str, source: str) -> None:
        async with self._rungraph_apply_lock:
            aliases = set(self._rungraph_inflight_aliases.get(target_fingerprint) or set())
            self._schedule_rungraph_status_publish_for_aliases(
                graph,
                req_ids=aliases,
                phase="applying",
                source=source,
                target_fingerprint=target_fingerprint,
            )
            try:
                await self.set_rungraph(graph)
            except asyncio.CancelledError:
                cancelled_aliases = set(self._rungraph_inflight_aliases.pop(target_fingerprint, aliases))
                await self._publish_rungraph_status_for_aliases(
                    graph,
                    req_ids=cancelled_aliases,
                    phase="failed",
                    source=source,
                    target_fingerprint=target_fingerprint,
                    error_message="rungraph apply cancelled",
                )
                raise
            except Exception as exc:
                failed_aliases = set(self._rungraph_inflight_aliases.pop(target_fingerprint, aliases))
                self._schedule_rungraph_status_publish_for_aliases(
                    graph,
                    req_ids=failed_aliases,
                    phase="failed",
                    source=source,
                    target_fingerprint=target_fingerprint,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                log.error("rungraph async apply failed service_id=%s", self.service_id, exc_info=exc)
                return
            applied_aliases = set(self._rungraph_inflight_aliases.pop(target_fingerprint, aliases))
            applied_fingerprint = self._rungraph_fingerprint or build_rungraph_deploy_fingerprint(graph)
            self._schedule_rungraph_status_publish_for_aliases(
                graph,
                req_ids=applied_aliases,
                phase="applied",
                source=source,
                target_fingerprint=target_fingerprint,
                applied_fingerprint=applied_fingerprint,
            )

    async def _publish_rungraph_status_for_aliases(
        self,
        graph: F8RuntimeGraph,
        *,
        req_ids: set[str],
        phase: str,
        source: str,
        target_fingerprint: str,
        applied_fingerprint: str = "",
        error_message: str = "",
    ) -> None:
        for req_id in sorted(str(item) for item in req_ids if str(item or "").strip()):
            await self._publish_rungraph_status(
                graph,
                req_id=req_id,
                phase=phase,
                source=source,
                target_fingerprint=target_fingerprint,
                applied_fingerprint=applied_fingerprint,
                error_message=error_message,
            )

    def _schedule_rungraph_status_publish_for_aliases(
        self,
        graph: F8RuntimeGraph,
        *,
        req_ids: set[str],
        phase: str,
        source: str,
        target_fingerprint: str,
        applied_fingerprint: str = "",
        error_message: str = "",
    ) -> None:
        for req_id in sorted(str(item) for item in req_ids if str(item or "").strip()):
            self._schedule_rungraph_status_publish(
                graph,
                req_id=req_id,
                phase=phase,
                source=source,
                target_fingerprint=target_fingerprint,
                applied_fingerprint=applied_fingerprint,
                error_message=error_message,
            )

    def _schedule_rungraph_status_publish(
        self,
        graph: F8RuntimeGraph,
        *,
        req_id: str,
        phase: str,
        source: str,
        target_fingerprint: str = "",
        applied_fingerprint: str = "",
        error_message: str = "",
    ) -> None:
        task = asyncio.create_task(
            self._publish_rungraph_status(
                graph,
                req_id=req_id,
                phase=phase,
                source=source,
                target_fingerprint=target_fingerprint,
                applied_fingerprint=applied_fingerprint,
                error_message=error_message,
            ),
            name=f"service_bus:rungraph_status:{self.service_id}:{req_id}:{phase}",
        )
        self._track_rungraph_task(task)

    async def _publish_rungraph_status(
        self,
        graph: F8RuntimeGraph,
        *,
        req_id: str,
        phase: str,
        source: str,
        target_fingerprint: str = "",
        applied_fingerprint: str = "",
        error_message: str = "",
    ) -> None:
        graph_id = str(graph.graphId or "")
        revision = str(graph.revision or "")
        phase_s = str(phase or "").strip()
        payload = {
            "schemaVersion": "f8.rungraphDeployStatus/2",
            "serviceId": self.service_id,
            "reqId": str(req_id or ""),
            "graphId": graph_id,
            "revision": revision,
            "phase": phase_s,
            "ok": phase_s == "applied",
            "source": str(source or ""),
            "errorMessage": str(error_message or ""),
            "ts": int(now_ms()),
            "targetFingerprint": str(target_fingerprint or ""),
            "appliedFingerprint": str(applied_fingerprint or ""),
            "runtimeInstanceId": self.runtime_instance_id,
        }
        raw = encode_obj(payload)
        status_keys = (
            self._rungraph_status_key,
            rungraph_deploy_request_status_key(self.service_id, str(req_id or "")),
        )
        for key in status_keys:
            try:
                await asyncio.wait_for(self._transport.retained_put(key, raw), timeout=1.0)
            except (asyncio.TimeoutError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                log.error(
                    "publish rungraph status failed service_id=%s req_id=%s phase=%s key=%s",
                    self.service_id,
                    req_id,
                    phase_s,
                    key,
                    exc_info=exc,
                )

    async def publish(self, key: str, payload: bytes) -> None:
        """Publish a message to a Zenoh key."""
        if not self._active:
            return
        await self._transport.publish(str(key), bytes(payload))

    async def subscribe(
        self,
        key_expr: str,
        *,
        queue: str | None = None,
        cb: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> Any:
        """Subscribe to a Zenoh key expression."""
        return await self._transport.subscribe(str(key_expr), queue=queue, cb=cb)

    async def emit_data(
        self,
        node_id: str,
        port: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        ctx_id: str | int | None = None,
    ) -> None:
        """
        Emit one output sample from a local node port.

        Canonical semantics:
        - local delivery: local routed consumers are satisfied first according to `data_delivery`
        - cross-service publish: controlled separately by `cross_publish_policy`
        - persistence: data samples are transient and are not written to KV state

        `emit_data(...)` is the public data-output path. Pull-triggered local
        recompute uses an internal local-only routing option so `pull_data(...)`
        never turns into hidden cross-service publication.
        """
        node_id_s = ensure_token(node_id, label="node_id")
        port_s = ensure_token(port, label="port_id")
        await self._data_router.emit_data(node_id_s, port_s, value, ts_ms=ts_ms, ctx_id=ctx_id)

    def set_exec_emitter(self, emitter: Callable[[str, str, str | int], Awaitable[None]] | None) -> None:
        self._exec_emitter = emitter

    async def emit_exec(self, node_id: str, port: str, *, exec_id: str | int) -> None:
        emitter = self._exec_emitter
        if emitter is None:
            return
        node_id_s = ensure_token(node_id, label="node_id")
        port_s = ensure_token(port, label="port_id")
        await emitter(node_id_s, port_s, exec_id=exec_id)

    async def pull_data(self, node_id: str, port: str, *, ctx_id: str | int | None = None) -> Any:
        """
        Read the current buffered input value for a local node input port.

        In buffered modes this may trigger same-service upstream `compute_output(...)`
        when the input has no fresh sample yet. That recompute satisfies local
        consumers only; it does not publish cross-service data.
        """
        node_id_s = ensure_token(node_id, label="node_id")
        port_s = ensure_token(port, label="port_id")
        return await self._data_router.pull_data(node_id_s, port_s, ctx_id=ctx_id)

    def data_input_zenoh_key(self, node_id: str, port: str) -> str | None:
        """
        Resolve the Zenoh stream key feeding a local typed stream input port.

        Media/binary data ports use this runtime-resolved key internally instead
        of exposing transport-specific media state fields to user graphs.
        """
        node_id_s = ensure_token(node_id, label="node_id")
        port_s = ensure_token(port, label="port_id")
        return self._data_router.input_stream_key(node_id=node_id_s, port=port_s)
