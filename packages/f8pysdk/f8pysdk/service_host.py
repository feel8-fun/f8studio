from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .generated import F8JsonValue, F8RuntimeGraph, F8RuntimeNode, F8StateAccess
from .runtime_node_registry import RuntimeNodeRegistry
from .service_bus import ServiceBus


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
    Push-based service host that binds a `ServiceBus` to per-node runtime implementations.

    - Rungraph drives creation/removal of local runtime nodes.
    - Runtime buffers data edges; exec/data evaluation is driven by the engine layer.
    """

    def __init__(
        self,
        bus: ServiceBus,
        *,
        config: ServiceHostConfig,
        registry: RuntimeNodeRegistry | None = None,
    ) -> None:
        self._bus = bus
        self._config = config
        self._registry = registry or RuntimeNodeRegistry.instance()

        self._service_node: Any | None = None
        self._operator_nodes: dict[str, Any] = {}
        self._bus.add_rungraph_listener(self._on_rungraph)

    async def start(self) -> None:
        """
        Ensure the service node exists before any rungraph arrives.

        Rungraph-provided service nodes are treated as state snapshots only.
        """
        if self._service_node is not None:
            return
        service_class = str(self._config.service_class or "").strip()
        if not service_class:
            raise ValueError("ServiceHostConfig.service_class must be non-empty")
        node_id = str(self._bus.service_id).strip()
        try:
            node = self._registry.create_service_node(service_class=service_class, node_id=node_id, initial_state={})
        except Exception:
            node = None
        if node is None:
            return
        self._service_node = node
        try:
            self._bus.register_node(node)
        except Exception:
            self._service_node = None
            return

    async def _on_rungraph(self, graph: F8RuntimeGraph) -> None:
        try:
            await self.apply_rungraph(graph)
        except Exception:
            return

    async def apply_rungraph(self, graph: F8RuntimeGraph) -> None:
        """
        Register/unregister local runtime nodes based on the latest rungraph snapshot.
        """
        if self._service_node is None:
            try:
                await self.start()
            except Exception:
                return
        service_class = str(self._config.service_class or "").strip()

        want_operator_nodes: list[F8RuntimeNode] = []
        service_snapshot: F8RuntimeNode | None = None
        for n in graph.nodes:
            try:
                if service_class and str(n.serviceClass) != service_class:
                    continue
                if getattr(n, "operatorClass", None) is None:
                    # Service/container node snapshot (state only).
                    if str(getattr(n, "nodeId", "")) == str(self._bus.service_id):
                        service_snapshot = n
                    continue
                want_operator_nodes.append(n)
            except Exception:
                continue

        want_ids = {str(n.nodeId) for n in want_operator_nodes}

        for node_id in list(self._operator_nodes.keys()):
            if node_id in want_ids:
                continue
            try:
                self._bus.unregister_node(node_id)
            except Exception:
                pass
            self._operator_nodes.pop(node_id, None)

        for n in want_operator_nodes:
            node_id = str(n.nodeId)
            if node_id in self._operator_nodes:
                continue
            initial_state = self._node_initial_state(n)
            try:
                node = self._registry.create(node_id=node_id, node=n, initial_state=initial_state)
            except Exception:
                node = None
            if node is None:
                continue
            self._operator_nodes[node_id] = node
            try:
                self._bus.register_node(node)
            except Exception:
                self._operator_nodes.pop(node_id, None)
                continue

        if service_snapshot is not None:
            await self._seed_service_state_defaults(service_snapshot)
        await self._seed_state_defaults(want_operator_nodes)

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
        Reconcile rungraph-provided initial state values into KV.

        If KV already has a value and differs, prefer the rungraph value and write it back
        with a fresh timestamp (current time).
        """
        for n in nodes:
            node_id = str(n.nodeId)
            try:
                access_by_name = {str(sf.name): sf.access for sf in list(getattr(n, "stateFields", None) or [])}
            except Exception:
                access_by_name = {}
            for k, v in self._node_initial_state(n).items():
                # Never seed/overwrite read-only state from rungraph; it is runtime-owned.
                try:
                    if access_by_name.get(str(k)) == F8StateAccess.ro:
                        continue
                except Exception:
                    pass
                try:
                    existing = await self._bus.get_state(node_id, str(k))
                except Exception:
                    existing = None
                if existing is not None and existing == v:
                    continue
                try:
                    await self._bus.set_state_with_meta(
                        node_id,
                        str(k),
                        v,
                        source="rungraph",
                        meta={"rungraphReconcile": True},
                    )
                except Exception:
                    continue

    async def _seed_service_state_defaults(self, n: F8RuntimeNode) -> None:
        """
        Apply rungraph-provided `stateValues` for the service node into KV.
        """
        node_id = str(self._bus.service_id)
        try:
            access_by_name = {str(sf.name): sf.access for sf in list(getattr(n, "stateFields", None) or [])}
        except Exception:
            access_by_name = {}
        for k, v in self._node_initial_state(n).items():
            # Never seed/overwrite read-only state from rungraph; it is runtime-owned.
            try:
                if access_by_name.get(str(k)) == F8StateAccess.ro:
                    continue
            except Exception:
                pass
            try:
                existing = await self._bus.get_state(node_id, str(k))
            except Exception:
                existing = None
            if existing is not None and existing == v:
                continue
            try:
                await self._bus.set_state_with_meta(
                    node_id,
                    str(k),
                    v,
                    source="rungraph",
                    meta={"rungraphReconcile": True, "serviceNode": True},
                )
            except Exception:
                continue
