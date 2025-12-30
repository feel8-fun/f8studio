from __future__ import annotations

from typing import Any

from f8pysdk import EdgeSpec, EdgeKind, EdgeScope, EdgeStrategy
from f8pysdk import OperatorAccess, OperatorDataPort, OperatorSpec, OperatorStateField

from .operator_instance import OperatorInstance
from .spec_registry import OperatorSpecRegistry


class OperatorGraph:
    """In-memory graph of operator instances with exec/data/state edges."""

    def __init__(self) -> None:
        self.nodes: dict[str, OperatorInstance] = {}
        self.exec_edges: list[EdgeSpec] = []
        self.data_edges: list[EdgeSpec] = []
        self.state_edges: list[EdgeSpec] = []

    # Node management -----------------------------------------------------
    def add_node(self, instance: OperatorInstance) -> None:
        if instance.id in self.nodes:
            raise ValueError(f"Node {instance.id} already exists")
        self.nodes[instance.id] = instance

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.exec_edges = [edge for edge in self.exec_edges if edge.from_ != node_id and edge.to != node_id]
        self.data_edges = [edge for edge in self.data_edges if edge.from_ != node_id and edge.to != node_id]
        self.state_edges = [edge for edge in self.state_edges if edge.from_ != node_id and edge.to != node_id]

    # Edge helpers --------------------------------------------------------
    def connect_exec(
        self,
        source_id: str,
        out_port: str,
        target_id: str,
        in_port: str,
        *,
        scope: EdgeScope = EdgeScope.intra,
    ) -> EdgeSpec:
        self._ensure_node(source_id)
        self._ensure_node(target_id)
        self._validate_exec_port(self.nodes[source_id], out_port, direction="out")
        self._validate_exec_port(self.nodes[target_id], in_port, direction="in")

        if any(edge.from_ == source_id and edge.fromPort == out_port for edge in self.exec_edges):
            raise ValueError(f"exec out port {out_port} on {source_id} already has a link")
        if any(edge.to == target_id and edge.toPort == in_port for edge in self.exec_edges):
            raise ValueError(f"exec in port {in_port} on {target_id} already has a link")

        edge = EdgeSpec(
            from_=source_id,
            fromPort=out_port,
            to=target_id,
            toPort=in_port,
            kind=EdgeKind.exec,
            scope=scope,
            strategy=EdgeStrategy.latest,
        )
        self.exec_edges.append(edge)
        return edge

    def connect_data(
        self,
        source_id: str,
        out_port: str,
        target_id: str,
        in_port: str,
        *,
        scope: EdgeScope = EdgeScope.intra,
        strategy: EdgeStrategy = EdgeStrategy.latest,
        queue_size: int | None = None,
        timeout_ms: int | None = None,
    ) -> EdgeSpec:
        self._ensure_node(source_id)
        self._ensure_node(target_id)
        source_port = self._validate_data_port(self.nodes[source_id], out_port, direction="out")
        target_port = self._validate_data_port(self.nodes[target_id], in_port, direction="in")

        if source_port.type != target_port.type:
            raise ValueError(
                f"data port type mismatch: {source_id}.{out_port} ({source_port.type}) -> "
                f"{target_id}.{in_port} ({target_port.type})"
            )
        if any(edge.to == target_id and edge.toPort == in_port for edge in self.data_edges):
            raise ValueError(f"data in port {in_port} on {target_id} already has a link")

        edge = EdgeSpec(
            from_=source_id,
            fromPort=out_port,
            to=target_id,
            toPort=in_port,
            kind=EdgeKind.data,
            scope=scope,
            strategy=strategy,
            queueSize=queue_size,
            timeoutMs=timeout_ms,
        )
        self.data_edges.append(edge)
        return edge

    def connect_state(
        self, source_id: str, source_field: str, target_id: str, target_field: str, *, scope: EdgeScope = EdgeScope.intra
    ) -> EdgeSpec:
        self._ensure_node(source_id)
        self._ensure_node(target_id)
        source_def = self._validate_state_field(self.nodes[source_id], source_field)
        target_def = self._validate_state_field(self.nodes[target_id], target_field)

        if target_def.access == OperatorAccess.ro:
            raise ValueError(f"target state field {target_field} on {target_id} is read-only")
        if any(edge.to == target_id and edge.toPort == target_field for edge in self.state_edges):
            raise ValueError(f"state field {target_field} on {target_id} already has a link")

        edge = EdgeSpec(
            from_=source_id,
            fromPort=source_field,
            to=target_id,
            toPort=target_field,
            kind=EdgeKind.state,
            scope=scope,
            strategy=EdgeStrategy.hold,
        )
        self.state_edges.append(edge)
        return edge

    def disconnect_edge(self, edge: EdgeSpec) -> None:
        if edge.kind == EdgeKind.exec:
            self.exec_edges = [e for e in self.exec_edges if e != edge]
        elif edge.kind == EdgeKind.data:
            self.data_edges = [e for e in self.data_edges if e != edge]
        elif edge.kind == EdgeKind.state:
            self.state_edges = [e for e in self.state_edges if e != edge]

    # Serialization -------------------------------------------------------
    def to_dict(self, *, include_ctx: bool = False) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": node.id,
                    "operatorClass": node.operator_class,
                    "spec": node.spec.model_dump(mode="json"),
                    "state": node.state,
                    **({"ctx": node.ctx} if include_ctx else {}),
                }
                for node in self.nodes.values()
            ],
            "edges": [
                edge.model_dump(by_alias=True, mode="json")
                for edge in [*self.exec_edges, *self.data_edges, *self.state_edges]
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, registry: OperatorSpecRegistry | None = None) -> "OperatorGraph":
        graph = cls()
        graph.load_dict(payload, registry=registry)
        return graph

    def load_dict(self, payload: dict[str, Any], *, registry: OperatorSpecRegistry | None = None) -> None:
        self.nodes.clear()
        self.exec_edges.clear()
        self.data_edges.clear()
        self.state_edges.clear()

        for node_data in payload.get("nodes", []):
            operator_class = node_data["operatorClass"]
            spec_data = node_data.get("spec")
            state = node_data.get("state") or {}
            ctx = node_data.get("ctx") or {}

            spec: OperatorSpec | None = None
            if spec_data:
                spec = OperatorSpec.model_validate(spec_data)
            if spec is None and registry:
                try:
                    spec = registry.get(operator_class)
                except Exception:
                    spec = None
            if spec is None:
                raise ValueError(f'Missing spec for operatorClass "{operator_class}"')

            instance = OperatorInstance.from_spec(spec, id=node_data["id"], state=state)
            if ctx:
                instance.ctx.update(ctx)
            self.add_node(instance)

        for edge_data in payload.get("edges", []):
            edge_spec = EdgeSpec.model_validate(edge_data)
            self._connect_from_spec(edge_spec)

    # Internal validators -------------------------------------------------
    def _connect_from_spec(self, edge: EdgeSpec) -> None:
        if edge.kind == EdgeKind.exec:
            self.connect_exec(edge.from_, edge.fromPort, edge.to, edge.toPort, scope=edge.scope)
        elif edge.kind == EdgeKind.data:
            self.connect_data(
                edge.from_,
                edge.fromPort,
                edge.to,
                edge.toPort,
                scope=edge.scope,
                strategy=edge.strategy,
                queue_size=edge.queueSize,
                timeout_ms=edge.timeoutMs,
            )
        elif edge.kind == EdgeKind.state:
            self.connect_state(edge.from_, edge.fromPort, edge.to, edge.toPort, scope=edge.scope)

    def _ensure_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")

    def _validate_exec_port(self, instance: OperatorInstance, port: str, *, direction: str) -> None:
        ports = instance.spec.execOutPorts if direction == "out" else instance.spec.execInPorts
        ports = ports or []
        if port not in ports:
            raise ValueError(f"{direction} exec port {port} missing on {instance.id}")

    def _validate_data_port(self, instance: OperatorInstance, port: str, *, direction: str) -> OperatorDataPort:
        ports = instance.spec.dataOutPorts if direction == "out" else instance.spec.dataInPorts
        ports = ports or []
        for port_def in ports:
            if port_def.name == port:
                return port_def
        raise ValueError(f"{direction} data port {port} missing on {instance.id}")

    def _validate_state_field(self, instance: OperatorInstance, name: str) -> OperatorStateField:
        for field_def in instance.spec.states or []:
            if field_def.name == name:
                return field_def
        raise ValueError(f"state field {name} missing on {instance.id}")
