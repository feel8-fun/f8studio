from __future__ import annotations

import enum
from uuid import uuid4

import msgspec

from f8pysdk.specs import F8DataPortSpec, F8JsonValue, F8OperatorSpec, F8ServiceSpec, F8StateSpec


DOCUMENT_SCHEMA_VERSION = "f8studio-document/1"


class NodeKind(str, enum.Enum):
    service = "service"
    operator = "operator"


class PortKind(str, enum.Enum):
    data = "data"
    state = "state"
    exec = "exec"
    command = "command"


class PortDirection(str, enum.Enum):
    input = "input"
    output = "output"


class GraphEdgeKind(str, enum.Enum):
    data = "data"
    state = "state"
    exec = "exec"


class EdgeStrategy(str, enum.Enum):
    latest = "latest"
    queue = "queue"


class GraphPort(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    port_id: str
    name: str
    runtime_name: str
    kind: PortKind
    direction: PortDirection
    data_spec: F8DataPortSpec | None = None
    state_spec: F8StateSpec | None = None


class ServiceNode(msgspec.Struct, frozen=True, kw_only=True, tag="service", tag_field="kind", rename="camel"):
    node_id: str
    name: str
    service_id: str
    service_class: str
    spec: F8ServiceSpec
    ports: tuple[GraphPort, ...] = ()
    state_values: dict[str, F8JsonValue] = msgspec.field(default_factory=dict)
    enabled: bool = True


class OperatorNode(msgspec.Struct, frozen=True, kw_only=True, tag="operator", tag_field="kind", rename="camel"):
    node_id: str
    name: str
    service_id: str
    service_class: str
    operator_class: str
    spec: F8OperatorSpec
    ports: tuple[GraphPort, ...] = ()
    state_values: dict[str, F8JsonValue] = msgspec.field(default_factory=dict)
    enabled: bool = True


GraphNode = ServiceNode | OperatorNode


class GraphEdge(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    edge_id: str
    from_node_id: str
    from_port_id: str
    to_node_id: str
    to_port_id: str
    kind: GraphEdgeKind
    strategy: EdgeStrategy = EdgeStrategy.latest
    queue_size: int = 16
    timeout_ms: int | None = None


class NodeLayout(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    node_id: str
    x: float
    y: float
    width: float | None = None
    height: float | None = None
    collapsed: bool = False


class StudioDocument(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    schema_version: str
    project_id: str
    graph_id: str
    graph_revision: int
    layout_revision: int
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    layout: tuple[NodeLayout, ...] = ()


class CreateNodeOp(msgspec.Struct, frozen=True, kw_only=True, tag="createNode", tag_field="op", rename="camel"):
    node: GraphNode
    layout: NodeLayout | None = None


class DeleteNodeOp(msgspec.Struct, frozen=True, kw_only=True, tag="deleteNode", tag_field="op", rename="camel"):
    node_id: str


class ConnectEdgeOp(msgspec.Struct, frozen=True, kw_only=True, tag="connectEdge", tag_field="op", rename="camel"):
    edge: GraphEdge


class DisconnectEdgeOp(
    msgspec.Struct, frozen=True, kw_only=True, tag="disconnectEdge", tag_field="op", rename="camel"
):
    edge_id: str


class SetNodeStateOp(msgspec.Struct, frozen=True, kw_only=True, tag="setNodeState", tag_field="op", rename="camel"):
    node_id: str
    field: str
    value: F8JsonValue


class RenameNodeOp(msgspec.Struct, frozen=True, kw_only=True, tag="renameNode", tag_field="op", rename="camel"):
    node_id: str
    name: str


class ReplaceNodeOp(msgspec.Struct, frozen=True, kw_only=True, tag="replaceNode", tag_field="op", rename="camel"):
    node: GraphNode


class BindOperatorServiceOp(
    msgspec.Struct, frozen=True, kw_only=True, tag="bindOperatorService", tag_field="op", rename="camel"
):
    node_id: str
    service_id: str


class SetNodeEnabledOp(
    msgspec.Struct, frozen=True, kw_only=True, tag="setNodeEnabled", tag_field="op", rename="camel"
):
    node_id: str
    enabled: bool


class SetNodeLayoutOp(
    msgspec.Struct, frozen=True, kw_only=True, tag="setNodeLayout", tag_field="op", rename="camel"
):
    layout: NodeLayout


class InsertFragmentOp(
    msgspec.Struct, frozen=True, kw_only=True, tag="insertFragment", tag_field="op", rename="camel"
):
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...] = ()
    layout: tuple[NodeLayout, ...] = ()


GraphOperation = (
    CreateNodeOp
    | DeleteNodeOp
    | ConnectEdgeOp
    | DisconnectEdgeOp
    | SetNodeStateOp
    | RenameNodeOp
    | ReplaceNodeOp
    | BindOperatorServiceOp
    | SetNodeEnabledOp
    | SetNodeLayoutOp
    | InsertFragmentOp
)


class PatchRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    request_id: str
    expected_graph_revision: int
    expected_layout_revision: int
    operations: tuple[GraphOperation, ...]


def new_document(*, project_id: str, graph_id: str | None = None) -> StudioDocument:
    return StudioDocument(
        schema_version=DOCUMENT_SCHEMA_VERSION,
        project_id=project_id,
        graph_id=graph_id or uuid4().hex,
        graph_revision=0,
        layout_revision=0,
    )
