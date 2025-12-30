from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .registry import OperatorRegistry
from ..generated.operator_spec import Access, DataPort, StateField, OperatorSpec

from .instance import OperatorInstance
from ..generated.operator_spec import Access, DataPort, StateField


@dataclass
class ExecEdge:
    source_id: str
    out_port: str
    target_id: str
    in_port: str


@dataclass
class DataEdge:
    source_id: str
    out_port: str
    target_id: str
    in_port: str
    schema: dict[str, Any] | None = None


@dataclass
class StateEdge:
    source_id: str
    source_field: str
    target_id: str
    target_field: str


class OperatorGraph:
    """Simple in-memory graph of operator instances and their links."""

    def __init__(self) -> None:
        self.nodes: dict[str, OperatorInstance] = {}
        self.exec_edges: list[ExecEdge] = []
        self.data_edges: list[DataEdge] = []
        self.state_edges: list[StateEdge] = []

    def add_node(self, instance: OperatorInstance) -> None:
        if instance.id in self.nodes:
            raise ValueError(f'Node {instance.id} already exists')
        self.nodes[instance.id] = instance

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.exec_edges = [
            edge
            for edge in self.exec_edges
            if edge.source_id != node_id and edge.target_id != node_id
        ]
        self.data_edges = [
            edge
            for edge in self.data_edges
            if edge.source_id != node_id and edge.target_id != node_id
        ]
        self.state_edges = [
            edge
            for edge in self.state_edges
            if edge.source_id != node_id and edge.target_id != node_id
        ]

    def connect_exec(self, source_id: str, out_port: str, target_id: str, in_port: str) -> ExecEdge:
        self._ensure_node(source_id)
        self._ensure_node(target_id)
        self._validate_exec_port(self.nodes[source_id], out_port, direction='out')
        self._validate_exec_port(self.nodes[target_id], in_port, direction='in')
        if any(
            edge.source_id == source_id and edge.out_port == out_port for edge in self.exec_edges
        ):
            raise ValueError(f'exec out port {out_port} on {source_id} already has a link')
        if any(edge.target_id == target_id and edge.in_port == in_port for edge in self.exec_edges):
            raise ValueError(f'exec in port {in_port} on {target_id} already has a link')
        edge = ExecEdge(source_id=source_id, out_port=out_port, target_id=target_id, in_port=in_port)
        self.exec_edges.append(edge)
        return edge

    def connect_data(self, source_id: str, out_port: str, target_id: str, in_port: str) -> DataEdge:
        self._ensure_node(source_id)
        self._ensure_node(target_id)
        source_port = self._validate_data_port(self.nodes[source_id], out_port, direction='out')
        target_port = self._validate_data_port(self.nodes[target_id], in_port, direction='in')
        if any(edge.target_id == target_id and edge.in_port == in_port for edge in self.data_edges):
            raise ValueError(f'data in port {in_port} on {target_id} already has a link')
        schema = source_port.schema_ or target_port.schema_
        edge = DataEdge(
            source_id=source_id,
            out_port=out_port,
            target_id=target_id,
            in_port=in_port,
            schema=schema,
        )
        self.data_edges.append(edge)
        return edge

    def disconnect_exec(self, edge: ExecEdge) -> None:
        self.exec_edges = [e for e in self.exec_edges if e != edge]

    def disconnect_data(self, edge: DataEdge) -> None:
        self.data_edges = [e for e in self.data_edges if e != edge]

    def connect_state(
        self, source_id: str, source_field: str, target_id: str, target_field: str
    ) -> StateEdge:
        """
        Link state fields to enable propagation between nodes.

        Only allows reading from ro/rw/init fields; disallows writing into ro fields.
        """
        self._ensure_node(source_id)
        self._ensure_node(target_id)
        source_def = self._validate_state_field(self.nodes[source_id], source_field)
        target_def = self._validate_state_field(self.nodes[target_id], target_field)

        if target_def.access == Access.ro:
            raise ValueError(f'target state field {target_field} on {target_id} is read-only')
        if any(
            edge.target_id == target_id and edge.target_field == target_field
            for edge in self.state_edges
        ):
            raise ValueError(f'state field {target_field} on {target_id} already has a link')
        edge = StateEdge(
            source_id=source_id,
            source_field=source_field,
            target_id=target_id,
            target_field=target_field,
        )
        self.state_edges.append(edge)
        return edge

    def disconnect_state(self, edge: StateEdge) -> None:
        self.state_edges = [e for e in self.state_edges if e != edge]

    def _ensure_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f'Node {node_id} not found')

    def _validate_exec_port(self, instance: OperatorInstance, port: str, *, direction: str) -> None:
        ports = instance.spec.execOutPorts if direction == 'out' else instance.spec.execInPorts
        if ports is None or port not in ports:
            raise ValueError(f'{direction} exec port {port} missing on {instance.id}')

    def _validate_data_port(
        self, instance: OperatorInstance, port: str, *, direction: str
    ) -> DataPort:
        ports = instance.spec.dataOutPorts if direction == 'out' else instance.spec.dataInPorts
        ports = ports or []
        for port_def in ports:
            if port_def.name == port:
                return port_def
        raise ValueError(f'{direction} data port {port} missing on {instance.id}')

    def _validate_state_field(self, instance: OperatorInstance, name: str) -> StateField:
        for field_def in instance.spec.states or []:
            if field_def.name == name:
                return field_def
        raise ValueError(f'state field {name} missing on {instance.id}')

    def to_dict(self, *, include_ctx: bool = False) -> dict[str, Any]:
        """Serialize the graph for persistence/transport."""
        return {
            'nodes': [
                {
                    'id': node.id,
                    'operatorClass': node.operator_class,
                    'spec': node.spec.model_dump(mode='json'),
                    'state': node.state,
                    **({'ctx': node.ctx} if include_ctx else {}),
                }
                for node in self.nodes.values()
            ],
            'execEdges': [edge.__dict__ for edge in self.exec_edges],
            'dataEdges': [edge.__dict__ for edge in self.data_edges],
            'stateEdges': [edge.__dict__ for edge in self.state_edges],
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], *, registry: OperatorRegistry | None = None
    ) -> 'OperatorGraph':
        """Instantiate a graph from a serialized payload."""
        graph = cls()
        graph.load_dict(payload, registry=registry)
        return graph

    def load_dict(self, payload: dict[str, Any], *, registry: OperatorRegistry | None = None) -> None:
        """
        Replace current graph contents from a serialized payload.

        If a registry is provided, specs will be pulled from it when possible; otherwise
        the embedded spec data is validated and used.
        """
        self.nodes.clear()
        self.exec_edges.clear()
        self.data_edges.clear()
        self.state_edges.clear()

        for node_data in payload.get('nodes', []):
            operator_class = node_data['operatorClass']
            spec_data = node_data.get('spec')
            state = node_data.get('state') or {}
            ctx = node_data.get('ctx') or {}

            spec: OperatorSpec | None = None
            if spec_data:
                spec = OperatorSpec.model_validate(spec_data)
            elif registry:
                try:
                    spec = registry.get(operator_class)
                except Exception:
                    spec = None

            if spec is None:
                raise ValueError(f'Missing spec for operatorClass "{operator_class}"')

            instance = OperatorInstance.from_spec(spec, id=node_data['id'], state=state)
            if ctx:
                instance.ctx.update(ctx)
            self.add_node(instance)

        for edge_data in payload.get('execEdges', []):
            self.connect_exec(
                edge_data['source_id'],
                edge_data['out_port'],
                edge_data['target_id'],
                edge_data['in_port'],
            )

        for edge_data in payload.get('dataEdges', []):
            edge = self.connect_data(
                edge_data['source_id'],
                edge_data['out_port'],
                edge_data['target_id'],
                edge_data['in_port'],
            )
            if 'schema' in edge_data:
                edge.schema = edge_data['schema']

        for edge_data in payload.get('stateEdges', []):
            self.connect_state(
                edge_data['source_id'],
                edge_data['source_field'],
                edge_data['target_id'],
                edge_data['target_field'],
            )
