from __future__ import annotations

from dataclasses import dataclass, field

from f8pysdk import F8RuntimeGraph
from f8pysdk.runtime import ServiceBus, ensure_token

from .engine_executor import EngineExecutor


@dataclass
class EngineBinder:
    """
    Bind a `ServiceBus` (all nodes) to an `EngineExecutor` (exec-capable nodes only).

    Node selection rule (as requested):
    - A node is considered "exec-capable" iff it declares at least one exec port
      (`execInPorts` or `execOutPorts`) in the rungraph.

    The binder is rungraph-driven and expects nodes to be materialized/registered
    into the bus before it runs (i.e. register this listener after `ServiceHost`).
    """

    bus: ServiceBus
    executor: EngineExecutor
    service_class: str
    _exec_node_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.service_class = str(self.service_class or "").strip()
        if not self.service_class:
            raise ValueError("service_class must be non-empty")
        self.bus.add_rungraph_listener(self._on_rungraph)

    async def _on_rungraph(self, graph: F8RuntimeGraph) -> None:
        await self._sync_exec_nodes(graph)
        await self.executor.apply_rungraph(graph)

    async def _sync_exec_nodes(self, graph: F8RuntimeGraph) -> None:
        want: set[str] = set()
        for n in list(graph.nodes or []):
            try:
                if str(getattr(n, "serviceClass", "")) != self.service_class:
                    continue
                exec_in = list(getattr(n, "execInPorts", None) or [])
                exec_out = list(getattr(n, "execOutPorts", None) or [])
                if not exec_in and not exec_out:
                    continue
                want.add(ensure_token(str(getattr(n, "nodeId", "")), label="nodeId"))
            except Exception:
                continue

        for node_id in sorted(self._exec_node_ids - want):
            try:
                self.executor.unregister_node(node_id)
            except Exception:
                pass
            self._exec_node_ids.discard(node_id)

        for node_id in sorted(want - self._exec_node_ids):
            node = self.bus.get_node(node_id)
            if node is None:
                continue
            if not hasattr(node, "on_exec"):
                # The rungraph declares exec ports but the runtime node does not implement exec.
                # Keep the bus registration (state/data still work), but skip executor binding.
                print(f"engine_binder: skip node without on_exec: {node_id}")
                continue
            try:
                self.executor.register_node(node)
                self._exec_node_ids.add(node_id)
            except Exception:
                continue

