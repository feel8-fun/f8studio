from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .generated import F8RuntimeGraph, F8RuntimeNode
from .json_unwrap import unwrap_json_value
from .runtime_node import OperatorNode, RuntimeNode
from .runtime_node_registry import RuntimeNodeRegistry
from .service_bus.bus import ServiceBus

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

        self._service_node: RuntimeNode | None = None
        self._operator_nodes: dict[str, OperatorNode] = {}
        self._bus.register_rungraph_hook(self)

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
                if n.operatorClass is None:
                    # Service/container node snapshot (state only).
                    if n.nodeId == str(self._bus.service_id):
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
                # If the node definition changed (ports/state), re-create the runtime node so
                # the local instance matches the rungraph snapshot. Otherwise, edited ports
                # may show in UI but not work at runtime.
                try:
                    existing = self._operator_nodes.get(node_id)
                    if existing is not None and self._needs_recreate(existing, n):
                        try:
                            self._bus.unregister_node(node_id)
                        except Exception:
                            pass
                        self._operator_nodes.pop(node_id, None)
                    else:
                        continue
                except Exception:
                    continue
            initial_state = self._node_initial_state(n)
            try:
                node = self._registry.create(node_id=node_id, node=n, initial_state=initial_state)
            except Exception:
                node = None
            if node is None:
                continue
            if not isinstance(node, OperatorNode):
                continue
            # Make runtime node metadata match the rungraph snapshot explicitly.
            # This keeps change detection deterministic and avoids "ghost fields".
            try:
                node.data_in_ports = [str(p.name) for p in (n.dataInPorts or [])]
                node.data_out_ports = [str(p.name) for p in (n.dataOutPorts or [])]
                node.state_fields = [str(s.name) for s in (n.stateFields or [])]
                node.exec_in_ports = [str(x) for x in (n.execInPorts or [])]
                node.exec_out_ports = [str(x) for x in (n.execOutPorts or [])]
            except Exception:
                pass
            self._operator_nodes[node_id] = node
            try:
                self._bus.register_node(node)
            except Exception:
                self._operator_nodes.pop(node_id, None)
                continue

    async def on_rungraph(self, graph: F8RuntimeGraph) -> None:
        await self.apply_rungraph(graph)

    async def validate_rungraph(self, graph: F8RuntimeGraph) -> None:
        _ = graph

    @staticmethod
    def _needs_recreate(node: OperatorNode, snapshot: F8RuntimeNode) -> bool:
        """
        Return True if the local runtime node should be re-created for the given snapshot.

        Nodes are cached by nodeId; without this check, editing ports in Studio can
        yield a rungraph that routes to ports the old instance doesn't expose.
        """
        desired_in = [str(p.name) for p in (snapshot.dataInPorts or [])]
        desired_out = [str(p.name) for p in (snapshot.dataOutPorts or [])]
        desired_state = [str(sf.name) for sf in (snapshot.stateFields or [])]

        current_in = [str(x) for x in list(node.data_in_ports or [])]
        current_out = [str(x) for x in list(node.data_out_ports or [])]
        current_state = [str(x) for x in list(node.state_fields or [])]

        if current_in != desired_in or current_out != desired_out or current_state != desired_state:
            return True

        desired_exec_in = [str(x) for x in (snapshot.execInPorts or [])]
        desired_exec_out = [str(x) for x in (snapshot.execOutPorts or [])]
        current_exec_in = [str(x) for x in list(node.exec_in_ports or [])]
        current_exec_out = [str(x) for x in list(node.exec_out_ports or [])]
        if current_exec_in != desired_exec_in or current_exec_out != desired_exec_out:
            return True

        return False

    @staticmethod
    def _node_initial_state(n: F8RuntimeNode) -> dict[str, Any]:
        out: dict[str, Any] = {}
        values = n.stateValues or {}
        if not values:
            return out
        for k, v in dict(values).items():
            out[str(k)] = unwrap_json_value(v)
        return out
