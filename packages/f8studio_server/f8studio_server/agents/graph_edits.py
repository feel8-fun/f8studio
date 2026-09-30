"""Resolve compact agent edits through the installed catalog and real graph ports."""

from __future__ import annotations

from f8studio_server.errors import InvalidRequestError

import msgspec

from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import (
    ConnectEdgeOp, CreateNodeOp, GraphEdge, NodeCatalog, NodeLayout,
    OperatorNode, PatchRequest, RefreshInstalledSpecOp, SetNodeStateOp, StudioDocument, replace_node_spec,
)
from f8studio_core.graph.catalog import can_refresh_installed_spec
from f8studio_core.graph.models import GraphEdgeKind, GraphNode, GraphOperation, PortDirection

from ..catalog import CatalogSnapshot


class NewNode(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    node_id: str
    service_class: str
    operator_class: str | None = None
    service_id: str | None = None
    name: str | None = None
    state_values: dict[str, F8JsonValue] = msgspec.field(default_factory=dict)
    x: float = 100
    y: float = 100


class Connection(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    from_node_id: str
    from_port: str
    to_node_id: str
    to_port: str
    kind: GraphEdgeKind = GraphEdgeKind.data


class StateUpdate(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    node_id: str
    field: str
    value: F8JsonValue


class GraphChanges(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    expected_graph_revision: int
    expected_layout_revision: int
    nodes: tuple[NewNode, ...] = ()
    connections: tuple[Connection, ...] = ()
    state_updates: tuple[StateUpdate, ...] = ()


def _port(node: GraphNode, name: str, kind: GraphEdgeKind, direction: PortDirection) -> str:
    matches = [port for port in node.ports
               if port.name == name and port.kind.value == kind.value and port.direction == direction]
    if len(matches) != 1:
        raise InvalidRequestError(f"Expected one {kind.value} {direction.value} port named {name!r} on {node.node_id}")
    return matches[0].port_id


def build_patch(document: StudioDocument, snapshot: CatalogSnapshot, changes: GraphChanges, *, request_id: str) -> PatchRequest:
    if not (changes.nodes or changes.connections or changes.state_updates):
        raise InvalidRequestError("Supply at least one node, connection, or state update")
    catalog = NodeCatalog(services=snapshot.services, operators=snapshot.operators)
    nodes = {node.node_id: node for node in document.nodes}
    operations: list[GraphOperation] = []
    installed_services = {str(spec.serviceClass): spec for spec in snapshot.services}
    installed_operators = {(str(spec.serviceClass), str(spec.operatorClass)): spec for spec in snapshot.operators}
    for node in document.nodes:
        installed = (installed_operators.get((node.service_class, node.operator_class))
                     if isinstance(node, OperatorNode) else installed_services.get(node.service_class))
        if installed is not None and can_refresh_installed_spec(node, installed):
            operations.append(RefreshInstalledSpecOp(node_id=node.node_id))
            nodes[node.node_id] = replace_node_spec(node, installed)
    for item in sorted(changes.nodes, key=lambda node: node.operator_class is not None):
        if item.node_id in nodes:
            raise InvalidRequestError(f"Node ID already exists: {item.node_id}; use stateUpdates for existing nodes")
        if "code" in item.state_values:
            raise InvalidRequestError("Use code_read, code_analyze, and code_write for Python code")
        node: GraphNode
        if item.operator_class is None:
            node = catalog.create_service_node(node_id=item.node_id, service_class=item.service_class,
                                               name=item.name, state_values=item.state_values)
        else:
            service = next((node for node in nodes.values() if not isinstance(node, OperatorNode)
                            and node.service_id == item.service_id and node.service_class == item.service_class), None)
            if service is None:
                raise InvalidRequestError(f"Service instance not found: {item.service_id} ({item.service_class})")
            node = catalog.create_operator_node(node_id=item.node_id, service_id=service.service_id,
                                                service_class=item.service_class, operator_class=item.operator_class,
                                                name=item.name, state_values=item.state_values)
        nodes[node.node_id] = node
        operations.append(CreateNodeOp(node=node, layout=NodeLayout(node_id=node.node_id, x=item.x, y=item.y)))
    for update in changes.state_updates:
        if update.field == "code":
            raise InvalidRequestError("Use code_read, code_analyze, and code_write for Python code")
        operations.append(SetNodeStateOp(node_id=update.node_id, field=update.field, value=update.value))
    for index, connection in enumerate(changes.connections):
        source = nodes.get(connection.from_node_id)
        target = nodes.get(connection.to_node_id)
        if source is None or target is None:
            raise InvalidRequestError(f"Connection references unknown node: {connection.from_node_id} -> {connection.to_node_id}")
        operations.append(ConnectEdgeOp(edge=GraphEdge(
            edge_id=f"{request_id}:edge:{index}", kind=connection.kind,
            from_node_id=source.node_id, from_port_id=_port(source, connection.from_port, connection.kind, PortDirection.output),
            to_node_id=target.node_id, to_port_id=_port(target, connection.to_port, connection.kind, PortDirection.input),
        )))
    return PatchRequest(request_id=request_id, expected_graph_revision=changes.expected_graph_revision,
                        expected_layout_revision=changes.expected_layout_revision, operations=tuple(operations))
