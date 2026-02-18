from __future__ import annotations

import logging
from typing import Any

from f8pysdk.executors.exec_flow import ExecFlowExecutor
from f8pysdk.executors.exec_flow import validate_exec_topology_or_raise
from f8pysdk.generated import F8RuntimeGraph
from f8pysdk.capabilities import ExecutableNode, ServiceHookBase
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_runtime import ServiceRuntime
from f8pysdk.service_cli import ServiceCliTemplate

from .constants import SERVICE_CLASS
from .pyengine_node_registry import register_pyengine_specs

logger = logging.getLogger(__name__)


class PyEngineService(ServiceCliTemplate, ServiceHookBase):
    """
    Fill-in-the-blanks service program for `f8.pyengine`.

    This is the canonical entry wiring:
    - register pyengine runtime node specs
    - attach ExecFlowExecutor (exec runtime)
    - bind exec-capable nodes from the rungraph
    - pause/resume executor via ServiceBus lifecycle events
    """

    def __init__(self) -> None:
        self._executor: ExecFlowExecutor | None = None
        self._exec_node_ids: set[str] = set()
        self._runtime: ServiceRuntime | None = None

    @property
    def service_class(self) -> str:
        return SERVICE_CLASS

    def register_specs(self, registry: RuntimeNodeRegistry) -> None:
        register_pyengine_specs(registry)

    async def setup(self, runtime: ServiceRuntime) -> None:
        executor = ExecFlowExecutor(runtime.bus)
        self._executor = executor
        self._runtime = runtime
        runtime.bus.register_rungraph_hook(self)
        runtime.bus.register_service_hook(self)

    async def teardown(self, runtime: ServiceRuntime) -> None:
        executor = self._executor
        try:
            runtime.bus.unregister_rungraph_hook(self)
        except Exception:
            logger.exception("unregister_rungraph_hook failed")
        try:
            runtime.bus.unregister_service_hook(self)
        except Exception:
            logger.exception("unregister_service_hook failed")
        self._runtime = None
        if executor is None:
            return
        try:
            await executor.stop_entrypoint()
        except Exception:
            logger.exception("stop_entrypoint failed during teardown")

    async def _sync_exec_nodes(self, runtime: ServiceRuntime, graph: F8RuntimeGraph) -> None:
        want: set[str] = set()
        for n in list(graph.nodes or []):
            if n.serviceClass != self.service_class:
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

        # If the current entrypoint node instance is being hot-replaced (same nodeId, new object),
        # stop it first so `apply_rungraph()` can restart it cleanly.
        try:
            entry_id = executor.current_entrypoint_node_id()
        except Exception:
            logger.exception("read current_entrypoint_node_id failed")
            entry_id = None
        if entry_id and entry_id in want:
            try:
                current_obj = executor.get_registered_node(entry_id)
                new_obj = runtime.bus.get_node(entry_id)
                if current_obj is not None and new_obj is not None and current_obj is not new_obj:
                    try:
                        await executor.stop_entrypoint()
                    except Exception:
                        logger.exception("stop_entrypoint failed for hot-replaced node")
            except Exception:
                logger.exception("compare hot-replaced entrypoint failed")

        for node_id in sorted(self._exec_node_ids - want):
            try:
                executor.unregister_node(node_id)
            except Exception:
                logger.exception("unregister exec node failed: %s", node_id)
            self._exec_node_ids.discard(node_id)

        # Always (re-)register nodes in `want` so hot-recreated runtime node instances
        # are picked up by the executor without requiring a nodeId change.
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
            except Exception:
                logger.exception("register exec node failed: %s", node_id)
                continue

    async def on_rungraph(self, graph: F8RuntimeGraph) -> None:
        runtime = self._runtime
        executor = self._executor
        if runtime is None or executor is None:
            return
        await self._sync_exec_nodes(runtime, graph)
        await executor.apply_rungraph(graph)

    async def validate_rungraph(self, graph: F8RuntimeGraph) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        validate_exec_topology_or_raise(graph, service_id=runtime.bus.service_id)

    async def on_activate(self, _bus: Any, _meta: dict[str, Any]) -> None:
        executor = self._executor
        if executor is None:
            return
        await executor.set_active(True)

    async def on_deactivate(self, _bus: Any, _meta: dict[str, Any]) -> None:
        executor = self._executor
        if executor is None:
            return
        await executor.set_active(False)
