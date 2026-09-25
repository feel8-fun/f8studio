from __future__ import annotations

import hashlib

import msgspec

from f8pysdk.specs import F8JsonValue, F8OperatorSpec, F8ServiceSpec

from .catalog import ports_for_spec
from .codec import canonical_json_bytes
from .models import (
    DOCUMENT_SCHEMA_VERSION,
    GraphEdge,
    NodeLayout,
    OperatorNode,
    ServiceNode,
    StudioDocument,
)
from .validation import validate_document


EXCHANGE_FORMAT = "f8graph"
EXCHANGE_VERSION = 3


class ExchangeMetadata(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    graph_id: str
    project_id: str


class ExchangeDefinitions(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    services: dict[str, F8ServiceSpec] = msgspec.field(default_factory=dict)
    operators: dict[str, F8OperatorSpec] = msgspec.field(default_factory=dict)


class ExchangeService(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    node_id: str
    name: str
    definition_ref: str
    port_ids: dict[str, str] = msgspec.field(default_factory=dict)
    state_values: dict[str, F8JsonValue] = msgspec.field(default_factory=dict)
    enabled: bool = True


class ExchangeOperator(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    node_id: str
    name: str
    service_id: str
    definition_ref: str
    port_ids: dict[str, str] = msgspec.field(default_factory=dict)
    state_values: dict[str, F8JsonValue] = msgspec.field(default_factory=dict)
    enabled: bool = True


class ExchangePresentation(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    layout: tuple[NodeLayout, ...] = ()
    node_order: tuple[str, ...] = msgspec.field(default_factory=tuple, name="nodeOrder")


class GraphExchange(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    format: str
    format_version: int
    metadata: ExchangeMetadata
    definitions: ExchangeDefinitions
    services: dict[str, ExchangeService]
    operators: dict[str, ExchangeOperator]
    connections: tuple[GraphEdge, ...]
    resources: dict[str, F8JsonValue] = msgspec.field(default_factory=dict)
    presentation: ExchangePresentation = msgspec.field(default_factory=ExchangePresentation)


_DECODER = msgspec.json.Decoder(GraphExchange)


def _definition_ref(spec: F8ServiceSpec | F8OperatorSpec) -> str:
    if isinstance(spec, F8ServiceSpec):
        normalized = msgspec.json.decode(msgspec.json.encode(spec), type=F8ServiceSpec)
    else:
        normalized = msgspec.json.decode(msgspec.json.encode(spec), type=F8OperatorSpec)
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def export_graph(document: StudioDocument) -> bytes:
    validate_document(document)
    services: dict[str, ExchangeService] = {}
    operators: dict[str, ExchangeOperator] = {}
    service_definitions: dict[str, F8ServiceSpec] = {}
    operator_definitions: dict[str, F8OperatorSpec] = {}
    for node in document.nodes:
        ref = _definition_ref(node.spec)
        if isinstance(node, ServiceNode):
            service_definitions[ref] = node.spec
            services[node.node_id] = ExchangeService(
                node_id=node.node_id,
                name=node.name,
                definition_ref=ref,
                port_ids=node.port_ids,
                state_values=node.state_values,
                enabled=node.enabled,
            )
        else:
            operator_definitions[ref] = node.spec
            operators[node.node_id] = ExchangeOperator(
                node_id=node.node_id,
                name=node.name,
                service_id=node.service_id,
                definition_ref=ref,
                port_ids=node.port_ids,
                state_values=node.state_values,
                enabled=node.enabled,
            )
    exchange = GraphExchange(
        format=EXCHANGE_FORMAT,
        format_version=EXCHANGE_VERSION,
        metadata=ExchangeMetadata(graph_id=document.graph_id, project_id=document.project_id),
        definitions=ExchangeDefinitions(services=service_definitions, operators=operator_definitions),
        services=services,
        operators=operators,
        connections=document.edges,
        presentation=ExchangePresentation(layout=document.layout, node_order=tuple(node.node_id for node in document.nodes)),
    )
    return canonical_json_bytes(exchange)


def import_graph(payload: bytes | str, *, project_id: str | None = None) -> StudioDocument:
    try:
        exchange = _DECODER.decode(payload)
    except msgspec.DecodeError as exc:
        raise ValueError(f"invalid f8graph document: {exc}") from exc
    if exchange.format != EXCHANGE_FORMAT or exchange.format_version != EXCHANGE_VERSION:
        raise ValueError(f"unsupported graph format: {exchange.format}/{exchange.format_version}")
    if exchange.resources:
        raise ValueError("f8graph resources are not supported yet")
    for ref, spec in exchange.definitions.services.items():
        if ref != _definition_ref(spec):
            raise ValueError(f"service definition hash mismatch: {ref}")
    for ref, spec in exchange.definitions.operators.items():
        if ref != _definition_ref(spec):
            raise ValueError(f"operator definition hash mismatch: {ref}")
    services: list[ServiceNode] = []
    operators: list[OperatorNode] = []
    for key, instance in exchange.services.items():
        if key != instance.node_id:
            raise ValueError(f"service key does not match nodeId: {key}")
        spec = exchange.definitions.services.get(instance.definition_ref)
        if spec is None:
            raise ValueError(f"missing service definition: {instance.definition_ref}")
        services.append(
            ServiceNode(
                node_id=instance.node_id,
                name=instance.name,
                service_id=instance.node_id,
                service_class=spec.serviceClass,
                spec=spec,
                ports=ports_for_spec(spec, instance.port_ids),
                port_ids=instance.port_ids,
                state_values=instance.state_values,
                enabled=instance.enabled,
            )
        )
    for key, instance in exchange.operators.items():
        if key != instance.node_id:
            raise ValueError(f"operator key does not match nodeId: {key}")
        spec = exchange.definitions.operators.get(instance.definition_ref)
        if spec is None:
            raise ValueError(f"missing operator definition: {instance.definition_ref}")
        operators.append(
            OperatorNode(
                node_id=instance.node_id,
                name=instance.name,
                service_id=instance.service_id,
                service_class=spec.serviceClass,
                operator_class=spec.operatorClass,
                spec=spec,
                ports=ports_for_spec(spec, instance.port_ids),
                port_ids=instance.port_ids,
                state_values=instance.state_values,
                enabled=instance.enabled,
            )
        )
    document = StudioDocument(
        schema_version=DOCUMENT_SCHEMA_VERSION,
        project_id=project_id or exchange.metadata.project_id,
        graph_id=exchange.metadata.graph_id,
        graph_revision=0,
        layout_revision=0,
        nodes=(*services, *operators),
        edges=exchange.connections,
        layout=exchange.presentation.layout,
    )
    if exchange.presentation.node_order:
        ordered = {node.node_id: node for node in document.nodes}
        if len(exchange.presentation.node_order) != len(ordered) or set(exchange.presentation.node_order) != set(ordered):
            raise ValueError("presentation.nodeOrder must list each node exactly once")
        document = msgspec.structs.replace(
            document,
            nodes=tuple(ordered[node_id] for node_id in exchange.presentation.node_order),
        )
    validate_document(document)
    return document
