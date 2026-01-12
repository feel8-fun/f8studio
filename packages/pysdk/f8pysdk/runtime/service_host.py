from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..generated import F8JsonValue, F8RuntimeGraph, F8RuntimeNode
from .service_operator_runtime_registry import ServiceOperatorRuntimeRegistry
from .service_runtime import ServiceRuntime


def _unwrap_json_value(v: Any) -> Any:
    if v is None:
        return None
    try:
        if isinstance(v, F8JsonValue):
            return v.root
    except Exception:
        pass
    try:
        root = getattr(v, "root", None)
        if root is not None:
            return root
    except Exception:
        pass
    return v


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
    Push-based service host that binds a `ServiceRuntime` to per-node runtime implementations.

    - Topology drives creation/removal of local runtime nodes.
    - Runtime pushes data into nodes (`on_data`) and manages cross-edge routing.
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

    async def _on_topology(self, graph: F8RuntimeGraph) -> None:
        try:
            await self.apply_topology(graph)
        except Exception:
            return

    async def apply_topology(self, graph: F8RuntimeGraph) -> None:
        """
        Register/unregister local runtime nodes based on the latest topology snapshot.
        """
        service_class = str(self._config.service_class or "").strip()

        want_nodes: list[F8RuntimeNode] = []
        for n in graph.nodes:
            try:
                if service_class and str(n.serviceClass) != service_class:
                    continue
                want_nodes.append(n)
            except Exception:
                continue

        want_ids = {str(n.nodeId) for n in want_nodes}

        for node_id in list(self._nodes.keys()):
            if node_id in want_ids:
                continue
            try:
                self._runtime.unregister_node(node_id)
            except Exception:
                pass
            self._nodes.pop(node_id, None)

        for n in want_nodes:
            node_id = str(n.nodeId)
            if node_id in self._nodes:
                continue
            initial_state = self._node_initial_state(n)
            try:
                node = self._registry.create(node_id=node_id, node=n, initial_state=initial_state)
            except Exception:
                node = None
            if node is None:
                continue
            self._nodes[node_id] = node
            try:
                self._runtime.register_node(node)
            except Exception:
                self._nodes.pop(node_id, None)
                continue

        await self._seed_state_defaults(want_nodes)

    @staticmethod
    def _node_initial_state(n: F8RuntimeNode) -> dict[str, Any]:
        values = getattr(n, "stateValues", None) or {}
        out: dict[str, Any] = {}
        if not isinstance(values, dict):
            return out
        for k, v in values.items():
            out[str(k)] = _unwrap_json_value(v)
        return out

    async def _seed_state_defaults(self, nodes: list[F8RuntimeNode]) -> None:
        """
        Ensure KV has at least the topology-provided initial state values.
        """
        for n in nodes:
            node_id = str(n.nodeId)
            for k, v in self._node_initial_state(n).items():
                try:
                    existing = await self._runtime.get_state(node_id, str(k))
                except Exception:
                    existing = None
                if existing is not None:
                    continue
                try:
                    await self._runtime.set_state_with_meta(node_id, str(k), v, source="topology")
                except Exception:
                    continue

