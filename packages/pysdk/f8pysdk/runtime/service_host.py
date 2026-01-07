from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..graph.operator_graph import OperatorGraph
from .service_operator_runtime_registry import ServiceOperatorRuntimeRegistry
from .service_runtime import ServiceRuntime


@dataclass(frozen=True)
class ServiceHostConfig:
    """
    Generic push-based service host.

    - One process hosts exactly one service instance (service_id).
    - `service_class` selects which runtime registry is used to build nodes.
    """

    service_class: str


class ServiceHost:
    """
    Generic service host that binds a `ServiceRuntime` to per-operator runtime nodes.

    - topology drives creation/removal of local runtime nodes
    - runtime pushes data into nodes (`on_data`) and manages cross-edge routing
    """

    def __init__(
        self,
        runtime: ServiceRuntime,
        *,
        config: ServiceHostConfig,
        registry: ServiceOperatorRuntimeRegistry | None = None,
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._registry = registry or ServiceOperatorRuntimeRegistry.instance()

        self._nodes: dict[str, Any] = {}
        self._runtime.add_topology_listener(self._on_topology)

    async def _on_topology(self, graph: OperatorGraph) -> None:
        try:
            await self.apply_topology(graph)
        except Exception:
            return

    async def apply_topology(self, graph: OperatorGraph) -> None:
        """
        Register/unregister local runtime nodes based on the latest topology snapshot.
        """
        want_ids = set(map(str, graph.nodes.keys()))
        for node_id in list(self._nodes.keys()):
            if node_id in want_ids:
                continue
            try:
                self._runtime.unregister_node(node_id)
            except Exception:
                pass
            self._nodes.pop(node_id, None)

        service_class = str(self._config.service_class or "").strip()

        for node_id, inst in graph.nodes.items():
            nid = str(node_id)
            if nid in self._nodes:
                continue
            try:
                inst_service_class = str(getattr(inst.spec, "serviceClass", "") or "").strip()
            except Exception:
                inst_service_class = ""
            if inst_service_class and service_class and inst_service_class != service_class:
                continue
            try:
                node = self._registry.create(
                    node_id=nid,
                    spec=inst.spec,
                    initial_state=dict(inst.state or {}),
                )
            except Exception:
                # Fallback: generic no-op runtime node.
                node = None
            if node is None:
                continue
            self._nodes[nid] = node
            try:
                self._runtime.register_node(node)
            except Exception:
                continue

        await self._seed_state_defaults(graph)

    async def _seed_state_defaults(self, graph: OperatorGraph) -> None:
        """
        Ensure KV has at least the topology-provided state values.
        """
        for node_id, inst in graph.nodes.items():
            for k, v in (inst.state or {}).items():
                try:
                    existing = await self._runtime.get_state(str(node_id), str(k))
                except Exception:
                    existing = None
                if existing is not None:
                    continue
                try:
                    await self._runtime.set_state_with_meta(str(node_id), str(k), v, source="topology")
                except Exception:
                    continue
