from __future__ import annotations

from typing import Any, Protocol

from ..generated import (
    F8DataPortSpec,
    F8DataTypeSchema,
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8OperatorSpec,
    F8StateAccess,
    F8StateSpec,
)

from .operator_instance import OperatorInstance
from ..runtime.nats_naming import new_id, ensure_token


class _SpecRegistryLike(Protocol):
    def get(self, spec_key: str) -> F8OperatorSpec: ...


class OperatorGraph:
    """In-memory graph of operator instances with exec/data/state edges."""

    def __init__(self, *, service_id: str) -> None:
        self.service_id = ensure_token(service_id, label="service_id")
        self.nodes: dict[str, OperatorInstance] = {}
        self.exec_edges: list[F8Edge] = []
        self.data_edges: list[F8Edge] = []
        self.state_edges: list[F8Edge] = []

    # Node management -----------------------------------------------------
    def add_node(self, instance: OperatorInstance) -> None:
        if instance.id in self.nodes:
            raise ValueError(f"Node {instance.id} already exists")
        self.nodes[instance.id] = instance

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.exec_edges = [
            edge
            for edge in self.exec_edges
            if edge.fromOperatorId != node_id and edge.toOperatorId != node_id
        ]
        self.data_edges = [
            edge
            for edge in self.data_edges
            if edge.fromOperatorId != node_id and edge.toOperatorId != node_id
        ]
        self.state_edges = [
            edge
            for edge in self.state_edges
            if edge.fromOperatorId != node_id and edge.toOperatorId != node_id
        ]

    # Edge helpers --------------------------------------------------------
    def connect_exec(
        self,
        source_id: str,
        out_port: str,
        target_id: str,
        in_port: str,
        *,
        edge_id: str | None = None,
    ) -> F8Edge:
        self._ensure_node(source_id)
        self._ensure_node(target_id)
        self._validate_exec_port(self.nodes[source_id], out_port, direction="out")
        self._validate_exec_port(self.nodes[target_id], in_port, direction="in")

        if any(edge.fromOperatorId == source_id and edge.fromPort == out_port for edge in self.exec_edges):
            raise ValueError(f"exec out port {out_port} on {source_id} already has a link")
        if any(edge.toOperatorId == target_id and edge.toPort == in_port for edge in self.exec_edges):
            raise ValueError(f"exec in port {in_port} on {target_id} already has a link")

        edge = F8Edge(
            edgeId=edge_id or new_id(),
            fromServiceId=self.service_id,
            fromOperatorId=source_id,
            fromPort=out_port,
            toServiceId=self.service_id,
            toOperatorId=target_id,
            toPort=in_port,
            kind=F8EdgeKindEnum.exec,
            strategy=F8EdgeStrategyEnum.latest,
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
        strategy: F8EdgeStrategyEnum = F8EdgeStrategyEnum.latest,
        queue_size: int | None = None,
        timeout_ms: int | None = None,
        edge_id: str | None = None,
    ) -> F8Edge:
        self._ensure_node(source_id)
        self._ensure_node(target_id)
        source_port = self._validate_data_port(self.nodes[source_id], out_port, direction="out")
        target_port = self._validate_data_port(self.nodes[target_id], in_port, direction="in")

        if self._schema_type(source_port.valueSchema) != self._schema_type(target_port.valueSchema):
            raise ValueError(f"data port type mismatch: {source_id}.{out_port} -> {target_id}.{in_port}")
        if any(edge.toOperatorId == target_id and edge.toPort == in_port for edge in self.data_edges):
            raise ValueError(f"data in port {in_port} on {target_id} already has a link")

        edge = F8Edge(
            edgeId=edge_id or new_id(),
            fromServiceId=self.service_id,
            fromOperatorId=source_id,
            fromPort=out_port,
            toServiceId=self.service_id,
            toOperatorId=target_id,
            toPort=in_port,
            kind=F8EdgeKindEnum.data,
            strategy=strategy,
            queueSize=queue_size,
            timeoutMs=timeout_ms,
        )
        self.data_edges.append(edge)
        return edge

    def connect_state(
        self,
        source_id: str,
        source_field: str,
        target_id: str,
        target_field: str,
        *,
        edge_id: str | None = None,
    ) -> F8Edge:
        self._ensure_node(source_id)
        self._ensure_node(target_id)
        source_def = self._validate_state_field(self.nodes[source_id], source_field)
        target_def = self._validate_state_field(self.nodes[target_id], target_field)

        if target_def.access == F8StateAccess.ro:
            raise ValueError(f"target state field {target_field} on {target_id} is read-only")
        if any(edge.toOperatorId == target_id and edge.toPort == target_field for edge in self.state_edges):
            raise ValueError(f"state field {target_field} on {target_id} already has a link")

        edge = F8Edge(
            edgeId=edge_id or new_id(),
            fromServiceId=self.service_id,
            fromOperatorId=source_id,
            fromPort=source_field,
            toServiceId=self.service_id,
            toOperatorId=target_id,
            toPort=target_field,
            kind=F8EdgeKindEnum.state,
            strategy=F8EdgeStrategyEnum.latest,
        )
        self.state_edges.append(edge)
        return edge

    def disconnect_edge(self, edge: F8Edge) -> None:
        if edge.kind == F8EdgeKindEnum.exec:
            self.exec_edges = [e for e in self.exec_edges if e != edge]
        elif edge.kind == F8EdgeKindEnum.data:
            self.data_edges = [e for e in self.data_edges if e != edge]
        elif edge.kind == F8EdgeKindEnum.state:
            self.state_edges = [e for e in self.state_edges if e != edge]

    # Serialization -------------------------------------------------------
    def to_dict(self, *, include_ctx: bool = False) -> dict[str, Any]:
        return {
            "serviceId": self.service_id,
            "nodes": [
                {
                    "id": node.id,
                    "serviceClass": node.service_class,
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
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        service_id: str,
        registry: _SpecRegistryLike | None = None,
    ) -> "OperatorGraph":
        graph = cls(service_id=service_id)
        graph.load_dict(payload, registry=registry)
        return graph

    def load_dict(self, payload: dict[str, Any], *, registry: _SpecRegistryLike | None = None) -> None:
        self.nodes.clear()
        self.exec_edges.clear()
        self.data_edges.clear()
        self.state_edges.clear()

        for node_data in payload.get("nodes", []):
            operator_class = str(node_data.get("operatorClass") or "").strip()
            service_class = str(node_data.get("serviceClass") or "").strip()
            spec_data = node_data.get("spec")
            state = node_data.get("state") or {}
            ctx = node_data.get("ctx") or {}

            spec: F8OperatorSpec | None = None
            if spec_data:
                spec = F8OperatorSpec.model_validate(spec_data)
                operator_class = str(getattr(spec, "operatorClass", "") or "").strip()
                service_class = str(getattr(spec, "serviceClass", "") or "").strip()
            if spec is None and registry is not None:
                try:
                    from ..schema_helpers import operator_key

                    if not service_class:
                        raise ValueError(f'Missing serviceClass for operatorClass "{operator_class}"')
                    spec = registry.get(operator_key(service_class, operator_class))
                except Exception:
                    spec = None
            if spec is None:
                raise ValueError(f'Missing spec for ({service_class}, {operator_class})')

            instance = OperatorInstance.from_spec(spec, id=node_data["id"], state=state)
            if ctx:
                instance.ctx.update(ctx)
            self.add_node(instance)

        for edge_data in payload.get("edges", []):
            edge_spec = F8Edge.model_validate(edge_data)
            self._connect_from_spec(edge_spec)

    # Internal validators -------------------------------------------------
    def _connect_from_spec(self, edge: F8Edge) -> None:
        cross = str(edge.fromServiceId) != str(edge.toServiceId)

        # Exec edges are only supported intra-service in this SDK.
        if edge.kind == F8EdgeKindEnum.exec and cross:
            return

        if edge.kind == F8EdgeKindEnum.exec:
            self._ensure_node(str(edge.fromOperatorId))
            self._ensure_node(str(edge.toOperatorId))
            self._validate_exec_port(self.nodes[str(edge.fromOperatorId)], str(edge.fromPort), direction="out")
            self._validate_exec_port(self.nodes[str(edge.toOperatorId)], str(edge.toPort), direction="in")
            self.exec_edges.append(edge)
            return

        if edge.kind == F8EdgeKindEnum.data:
            if str(edge.fromServiceId) == self.service_id and edge.fromOperatorId:
                self._ensure_node(str(edge.fromOperatorId))
                self._validate_data_port(self.nodes[str(edge.fromOperatorId)], str(edge.fromPort), direction="out")
            if str(edge.toServiceId) == self.service_id and edge.toOperatorId:
                self._ensure_node(str(edge.toOperatorId))
                self._validate_data_port(self.nodes[str(edge.toOperatorId)], str(edge.toPort), direction="in")
            self.data_edges.append(edge)
            return

        if edge.kind == F8EdgeKindEnum.state:
            if str(edge.fromServiceId) == self.service_id and edge.fromOperatorId:
                self._ensure_node(str(edge.fromOperatorId))
                self._validate_state_field(self.nodes[str(edge.fromOperatorId)], str(edge.fromPort))
            if str(edge.toServiceId) == self.service_id and edge.toOperatorId:
                self._ensure_node(str(edge.toOperatorId))
                target_def = self._validate_state_field(self.nodes[str(edge.toOperatorId)], str(edge.toPort))
                if target_def.access == F8StateAccess.ro:
                    raise ValueError(
                        f"target state field {edge.toPort} on {edge.toOperatorId} is read-only"
                    )
            self.state_edges.append(edge)
            return

    def _ensure_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found")

    def _validate_exec_port(self, instance: OperatorInstance, port: str, *, direction: str) -> None:
        ports = instance.spec.execOutPorts if direction == "out" else instance.spec.execInPorts
        ports = ports or []
        if port not in ports:
            raise ValueError(f"{direction} exec port {port} missing on {instance.id}")

    def _validate_data_port(self, instance: OperatorInstance, port: str, *, direction: str) -> F8DataPortSpec:
        ports = instance.spec.dataOutPorts if direction == "out" else instance.spec.dataInPorts
        ports = ports or []
        for port_def in ports:
            if port_def.name == port:
                return port_def
        raise ValueError(f"{direction} data port {port} missing on {instance.id}")

    def _validate_state_field(self, instance: OperatorInstance, name: str) -> F8StateSpec:
        for field_def in instance.spec.stateFields or []:
            if field_def.name == name:
                return field_def
        raise ValueError(f"state field {name} missing on {instance.id}")

    @staticmethod
    def _schema_type(schema: F8DataTypeSchema) -> str | None:
        root = schema.root
        return getattr(root, "type", None)
