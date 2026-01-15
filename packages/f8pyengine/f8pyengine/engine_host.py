from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from f8pysdk import F8RuntimeGraph
from f8pysdk.runtime import ServiceOperatorRuntimeRegistry, ServiceRuntime, ensure_token

from .engine_executor import EngineExecutor


@dataclass(frozen=True)
class EngineHostConfig:
    service_class: str = "f8.pyengine"


class EngineHost:
    """
    Engine-side host that materializes runtime nodes from `F8RuntimeGraph` and wires them to:
    - `ServiceRuntime` for data/state routing
    - `EngineExecutor` for exec routing / source lifecycle
    """

    def __init__(
        self,
        runtime: ServiceRuntime,
        executor: EngineExecutor,
        *,
        config: EngineHostConfig,
        registry: ServiceOperatorRuntimeRegistry | None = None,
    ) -> None:
        self._runtime = runtime
        self._executor = executor
        self._config = config
        self._registry = registry or ServiceOperatorRuntimeRegistry.instance()
        self._nodes: dict[str, Any] = {}

    async def apply_topology(self, graph: F8RuntimeGraph) -> None:
        # Only materialize executable operator nodes. Container/service nodes may exist in the
        # runtime graph for metadata/telemetry/state but are not engine-executed.
        want_nodes = [
            n
            for n in (graph.nodes or [])
            if str(getattr(n, "serviceClass", "")) == self._config.service_class
            and getattr(n, "operatorClass", None)
        ]
        want_ids = {str(n.nodeId) for n in want_nodes}

        for node_id in list(self._nodes.keys()):
            if node_id in want_ids:
                continue
            try:
                self._executor.unregister_node(node_id)
            except Exception:
                pass
            try:
                self._runtime.unregister_node(node_id)
            except Exception:
                pass
            self._nodes.pop(node_id, None)

        for n in want_nodes:
            node_id = ensure_token(str(n.nodeId), label="nodeId")
            if node_id in self._nodes:
                continue
            initial_state = {}
            try:
                state_values = getattr(n, "stateValues", None) or {}
                if isinstance(state_values, dict):
                    for k, v in state_values.items():
                        try:
                            initial_state[str(k)] = getattr(v, "root", v)
                        except Exception:
                            initial_state[str(k)] = v
            except Exception:
                initial_state = {}
            try:
                node = self._registry.create(node_id=node_id, node=n, initial_state=initial_state)
            except Exception:
                node = None
            if node is None:
                continue
            self._nodes[node_id] = node
            try:
                self._runtime.register_node(node)
                self._executor.register_node(node)
            except Exception:
                self._nodes.pop(node_id, None)
                try:
                    self._runtime.unregister_node(node_id)
                except Exception:
                    pass
                try:
                    self._executor.unregister_node(node_id)
                except Exception:
                    pass
                continue
