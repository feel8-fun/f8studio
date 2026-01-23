from __future__ import annotations

from f8pysdk.executors.exec_flow import ExecFlowExecutor
from f8pysdk.generated import F8RuntimeGraph
from f8pysdk.capabilities import ExecutableNode
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_runtime import ServiceRuntime
from f8pysdk.service_cli import ServiceCliTemplate

from .constants import SERVICE_CLASS
from .pyengine_node_registry import register_pyengine_specs


class PyEngineService(ServiceCliTemplate):
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

    @property
    def service_class(self) -> str:
        return SERVICE_CLASS

    def register_specs(self, registry: RuntimeNodeRegistry) -> None:
        register_pyengine_specs(registry)

    async def setup(self, runtime: ServiceRuntime) -> None:
        executor = ExecFlowExecutor(runtime.bus)
        self._executor = executor

        async def _on_rungraph(graph: F8RuntimeGraph) -> None:
            await self._sync_exec_nodes(runtime, graph)
            await executor.apply_rungraph(graph)

        runtime.bus.add_rungraph_listener(_on_rungraph)

        async def _on_lifecycle(active: bool, _meta: dict[str, object]) -> None:
            await executor.set_active(active)

        runtime.bus.add_lifecycle_listener(_on_lifecycle)

    async def teardown(self, runtime: ServiceRuntime) -> None:
        executor = self._executor
        if executor is None:
            return
        try:
            await executor.stop_entrypoint()
        except Exception:
            pass

    async def _sync_exec_nodes(self, runtime: ServiceRuntime, graph: F8RuntimeGraph) -> None:
        want: set[str] = set()
        for n in list(graph.nodes or []):
            try:
                if n.serviceClass != self.service_class:
                    continue
                exec_in = n.execInPorts
                exec_out = n.execOutPorts
                if not exec_in and not exec_out:
                    continue
                want.add(ensure_token(n.nodeId, label="nodeId"))
            except Exception:
                continue

        executor = self._executor
        if executor is None:
            return

        for node_id in sorted(self._exec_node_ids - want):
            try:
                executor.unregister_node(node_id)
            except Exception:
                pass
            self._exec_node_ids.discard(node_id)

        for node_id in sorted(want - self._exec_node_ids):
            node = runtime.bus.get_node(node_id)
            if node is None:
                continue
            if not isinstance(node, ExecutableNode):
                print(f"pyengine: skip node without on_exec: {node_id}")
                continue
            try:
                executor.register_node(node)
                self._exec_node_ids.add(node_id)
            except Exception:
                continue
