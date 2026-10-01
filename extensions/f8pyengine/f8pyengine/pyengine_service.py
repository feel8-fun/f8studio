from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from f8pysdk.app import ServiceApp
from f8pysdk.capabilities import ExecutableNode, ServiceHookBase
from f8pysdk.executors.exec_flow import ExecFlowExecutor
from f8pysdk.executors.exec_flow import validate_exec_topology_or_raise
from f8pysdk.f8_naming import ensure_token
from f8pysdk.registry import Registry
from f8pysdk.runtime import ServiceRuntime
from f8pysdk.specs import F8RuntimeGraph

from .auto_sampler import AutoSamplerManager
from .constants import SERVICE_CLASS
from .pyengine_node_registry import register_pyengine_specs

logger = logging.getLogger(__name__)
_SERVICE_HOOK_ERRORS = (LookupError, RuntimeError, TypeError, ValueError)
_EXEC_NODE_REGISTRY_ERRORS = (LookupError, RuntimeError, TypeError, ValueError)


class PyEngineService(ServiceHookBase):
    """
    Canonical entry wiring for `f8.pyengine`.

    - registers pyengine runtime node specs
    - attaches ExecFlowExecutor
    - binds exec-capable nodes from the rungraph
    - pauses/resumes executor from ServiceBus lifecycle events
    """

    def __init__(self) -> None:
        self._executor: ExecFlowExecutor | None = None
        self._exec_node_ids: set[str] = set()
        self._runtime: ServiceRuntime | None = None
        self._auto_sampler: AutoSamplerManager | None = None

    async def setup(self, runtime: ServiceRuntime) -> None:
        runtime.bus.set_data_delivery("buffered", source="service")
        executor = ExecFlowExecutor(runtime.bus)
        auto_sampler = AutoSamplerManager(runtime.bus)
        self._executor = executor
        self._runtime = runtime
        self._auto_sampler = auto_sampler
        runtime.bus.set_exec_emitter(executor.trigger_exec_nowait)
        runtime.bus.register_rungraph_hook(self)
        runtime.bus.register_service_hook(self)

    def _teardown_call(self, label: str, step: Callable[[], None]) -> None:
        try:
            step()
        except _SERVICE_HOOK_ERRORS:
            logger.exception("%s failed during teardown", label)

    async def _teardown_await(self, label: str, step: Callable[[], Awaitable[None]]) -> None:
        try:
            await step()
        except _SERVICE_HOOK_ERRORS:
            logger.exception("%s failed during teardown", label)

    async def teardown(self, runtime: ServiceRuntime) -> None:
        executor = self._executor
        auto_sampler = self._auto_sampler
        self._teardown_call("unregister_rungraph_hook", lambda: runtime.bus.unregister_rungraph_hook(self))
        self._teardown_call("unregister_service_hook", lambda: runtime.bus.unregister_service_hook(self))
        runtime.bus.set_exec_emitter(None)
        self._runtime = None
        self._auto_sampler = None
        if auto_sampler is not None:
            await self._teardown_await("auto sampler close", auto_sampler.close)
        if executor is None:
            return
        await self._teardown_await("set_active(False)", lambda: executor.set_active(False))
        await self._teardown_await("stop_all_entrypoints", executor.stop_all_entrypoints)

    async def _sync_exec_nodes(self, runtime: ServiceRuntime, graph: F8RuntimeGraph) -> None:
        want: set[str] = set()
        for n in list(graph.nodes or []):
            if n.serviceClass != SERVICE_CLASS:
                continue
            exec_in = list(n.execInPorts or [])
            exec_out = list(n.execOutPorts or [])
            if not exec_in and not exec_out:
                continue
            try:
                want.add(ensure_token(str(n.nodeId), label="nodeId"))
            except ValueError:
                logger.warning("skip invalid exec node id in rungraph: %r", n.nodeId)
                continue

        executor = self._executor
        if executor is None:
            return

        for node_id in sorted(self._exec_node_ids - want):
            try:
                executor.unregister_node(node_id)
            except _EXEC_NODE_REGISTRY_ERRORS:
                logger.exception("unregister exec node failed: %s", node_id)
            self._exec_node_ids.discard(node_id)

        for node_id in sorted(want):
            node = runtime.bus.get_node(node_id)
            if node is None:
                continue
            if not isinstance(node, ExecutableNode):
                logger.warning("skip node without on_exec: %s", node_id)
                continue
            try:
                executor.register_node(node)
                self._exec_node_ids.add(node_id)
            except _EXEC_NODE_REGISTRY_ERRORS:
                logger.exception("register exec node failed: %s", node_id)
                continue

    async def on_rungraph(self, graph: F8RuntimeGraph) -> None:
        runtime = self._runtime
        executor = self._executor
        if runtime is None or executor is None:
            return
        await self._sync_exec_nodes(runtime, graph)
        await executor.apply_rungraph(graph)
        auto_sampler = self._auto_sampler
        if auto_sampler is not None:
            await auto_sampler.sync_rungraph(graph)

    async def validate_rungraph(self, graph: F8RuntimeGraph) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        validate_exec_topology_or_raise(graph, service_id=runtime.bus.service_id)

    async def on_activate(self, _bus: Any, _meta: dict[str, Any]) -> None:
        auto_sampler = self._auto_sampler
        if auto_sampler is not None:
            await auto_sampler.set_active(True)
        executor = self._executor
        if executor is None:
            return
        await executor.set_active(True)

    async def on_deactivate(self, _bus: Any, _meta: dict[str, Any]) -> None:
        auto_sampler = self._auto_sampler
        if auto_sampler is not None:
            await auto_sampler.set_active(False)
        executor = self._executor
        if executor is None:
            return
        await executor.set_active(False)


def build_app() -> ServiceApp:
    registry = Registry()
    register_pyengine_specs(registry)
    hooks = PyEngineService()
    return ServiceApp(
        service_class=SERVICE_CLASS,
        registry=registry,
        setup=hooks.setup,
        teardown=hooks.teardown,
    )
