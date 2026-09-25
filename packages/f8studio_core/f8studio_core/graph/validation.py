from __future__ import annotations

import math
from typing import NoReturn

import msgspec

from f8pysdk.f8_naming import ensure_token
from f8pysdk.specs import (
    F8AnyTypeSchema,
    F8ArrayTypeSchema,
    F8BooleanTypeSchema,
    F8ComplexObjectTypeSchema,
    F8DataTypeSchema,
    F8IntegerTypeSchema,
    F8JsonValue,
    F8NullTypeSchema,
    F8NumberTypeSchema,
    F8StateAccess,
    F8StringTypeSchema,
    F8UiControlKind,
    data_port_payload_kind,
)

from .catalog import ports_for_spec
from .models import (
    DOCUMENT_SCHEMA_VERSION,
    GraphEdgeKind,
    GraphNode,
    GraphPort,
    OperatorNode,
    PortDirection,
    PortKind,
    ServiceNode,
    StudioDocument,
)


class GraphValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> NoReturn:
    raise GraphValidationError(code, message)


def _validate_token(value: str, label: str) -> None:
    try:
        ensure_token(value, label=label)
    except ValueError as exc:
        _fail("invalid_id", str(exc))


def _node_port_map(node: GraphNode) -> dict[str, GraphPort]:
    return {port.port_id: port for port in node.ports}


def _edge_port_kind(port: GraphPort) -> GraphEdgeKind:
    if port.kind == PortKind.data:
        return GraphEdgeKind.data
    if port.kind == PortKind.exec:
        return GraphEdgeKind.exec
    return GraphEdgeKind.state


def _validate_schema_value(value: F8JsonValue, schema: F8DataTypeSchema, path: str) -> None:
    if isinstance(schema, F8AnyTypeSchema):
        return
    if isinstance(schema, F8StringTypeSchema):
        if not isinstance(value, str):
            _fail("invalid_state_value", f"{path} must be a string")
        if not isinstance(schema.enum, msgspec.UnsetType) and value not in schema.enum:
            _fail("invalid_state_value", f"{path} must be one of {schema.enum}")
        return
    if isinstance(schema, F8BooleanTypeSchema):
        if not isinstance(value, bool):
            _fail("invalid_state_value", f"{path} must be a boolean")
        if not isinstance(schema.enum, msgspec.UnsetType) and value not in schema.enum:
            _fail("invalid_state_value", f"{path} must be one of {schema.enum}")
        return
    if isinstance(schema, F8NullTypeSchema):
        if value is not None:
            _fail("invalid_state_value", f"{path} must be null")
        return
    if isinstance(schema, (F8NumberTypeSchema, F8IntegerTypeSchema)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            expected = "an integer" if isinstance(schema, F8IntegerTypeSchema) else "a finite number"
            _fail("invalid_state_value", f"{path} must be {expected}")
        if isinstance(schema, F8IntegerTypeSchema) and not isinstance(value, int):
            _fail("invalid_state_value", f"{path} must be an integer")
        numeric = float(value)
        if not isinstance(schema.enum, msgspec.UnsetType) and value not in schema.enum:
            _fail("invalid_state_value", f"{path} must be one of {schema.enum}")
        if not isinstance(schema.minimum, msgspec.UnsetType) and numeric < schema.minimum:
            _fail("invalid_state_value", f"{path} must be at least {schema.minimum}")
        if not isinstance(schema.maximum, msgspec.UnsetType) and numeric > schema.maximum:
            _fail("invalid_state_value", f"{path} must be at most {schema.maximum}")
        if not isinstance(schema.exclusiveMinimum, msgspec.UnsetType) and numeric <= schema.exclusiveMinimum:
            _fail("invalid_state_value", f"{path} must be greater than {schema.exclusiveMinimum}")
        if not isinstance(schema.exclusiveMaximum, msgspec.UnsetType) and numeric >= schema.exclusiveMaximum:
            _fail("invalid_state_value", f"{path} must be less than {schema.exclusiveMaximum}")
        if not isinstance(schema.multipleOf, msgspec.UnsetType):
            quotient = numeric / schema.multipleOf
            if not math.isclose(quotient, round(quotient), rel_tol=1e-9, abs_tol=1e-9):
                _fail("invalid_state_value", f"{path} must be a multiple of {schema.multipleOf}")
        return
    if isinstance(schema, F8ArrayTypeSchema):
        if not isinstance(value, (list, tuple)):
            _fail("invalid_state_value", f"{path} must be an array")
        for index, item in enumerate(value):
            _validate_schema_value(item, schema.items, f"{path}[{index}]")
        return
    assert isinstance(schema, F8ComplexObjectTypeSchema)
    if not isinstance(value, dict):
        _fail("invalid_state_value", f"{path} must be an object")
    required = () if isinstance(schema.required, msgspec.UnsetType) else schema.required
    missing = [name for name in required if name not in value]
    if missing:
        _fail("invalid_state_value", f"{path} is missing required fields: {', '.join(missing)}")
    if schema.additionalProperties is False:
        unexpected = sorted(set(value) - set(schema.properties))
        if unexpected:
            _fail("invalid_state_value", f"{path} has unexpected fields: {', '.join(unexpected)}")
    for name, item in value.items():
        property_schema = schema.properties.get(name)
        if property_schema is not None:
            _validate_schema_value(item, property_schema, f"{path}.{name}")


def _validate_node(node: GraphNode, service_nodes: dict[str, ServiceNode]) -> None:
    _validate_token(node.node_id, "node_id")
    _validate_token(node.service_id, "service_id")
    if not node.name.strip():
        _fail("invalid_node", f"node name must be non-empty: {node.node_id}")
    if isinstance(node, ServiceNode):
        if node.node_id != node.service_id:
            _fail("invalid_service_binding", f"service node {node.node_id} must own service id {node.node_id}")
        if node.service_class != str(node.spec.serviceClass):
            _fail("spec_mismatch", f"serviceClass mismatch for {node.node_id}")
    else:
        service = service_nodes.get(node.service_id)
        if service is None:
            _fail("missing_service", f"operator {node.node_id} references missing service {node.service_id}")
        if service.service_class != node.service_class:
            _fail(
                "service_class_mismatch",
                f"operator {node.node_id} requires {node.service_class}, service {node.service_id} is {service.service_class}",
            )
        if node.service_class != str(node.spec.serviceClass) or node.operator_class != str(node.spec.operatorClass):
            _fail("spec_mismatch", f"operator spec identity mismatch for {node.node_id}")

    standard_port_ids = {port.port_id for port in ports_for_spec(node.spec)}
    if set(node.port_ids) - standard_port_ids:
        _fail("invalid_port_ids", f"port ID map references missing ports on node {node.node_id}")
    if node.ports != ports_for_spec(node.spec, node.port_ids):
        _fail("port_spec_mismatch", f"ports do not match spec for node {node.node_id}")
    port_ids = [port.port_id for port in node.ports]
    if len(port_ids) != len(set(port_ids)):
        _fail("duplicate_port", f"duplicate port id on node {node.node_id}")

    state_fields = {
        str(field.name): field
        for field in ([] if isinstance(node.spec.stateFields, msgspec.UnsetType) else node.spec.stateFields)
    }
    for field in state_fields.values():
        control = field.control
        if isinstance(control, msgspec.UnsetType):
            continue
        if control.kind == F8UiControlKind.custom and (
            isinstance(control.rendererKey, msgspec.UnsetType) or not control.rendererKey.strip()
        ):
            _fail("invalid_control", f"custom control requires rendererKey: {node.node_id}.{field.name}")
        pool = control.optionsFromState
        if isinstance(pool, msgspec.UnsetType):
            continue
        if control.kind not in (F8UiControlKind.select, F8UiControlKind.multiselect):
            _fail("invalid_control", f"optionsFromState requires a selection control: {node.node_id}.{field.name}")
        source = state_fields.get(pool)
        if source is None or not isinstance(source.valueSchema, F8ArrayTypeSchema):
            _fail("invalid_control", f"optionsFromState must reference an array state: {node.node_id}.{field.name}")
    for field_name, value in node.state_values.items():
        field = state_fields.get(field_name)
        if field is None:
            _fail("missing_state_field", f"state field not found: {node.node_id}.{field_name}")
        if field.access == F8StateAccess.ro:
            _fail("readonly_state", f"cannot persist read-only state: {node.node_id}.{field_name}")
        try:
            msgspec.json.encode(value)
        except (TypeError, ValueError) as exc:
            _fail("invalid_state_value", f"state value is not JSON compatible: {node.node_id}.{field_name}: {exc}")
        _validate_schema_value(value, field.valueSchema, f"{node.node_id}.{field_name}")


def _validate_edges(document: StudioDocument, nodes: dict[str, GraphNode]) -> None:
    data_inputs: dict[tuple[str, str], str] = {}
    state_inputs: dict[tuple[str, str], str] = {}
    exec_inputs: dict[tuple[str, str], str] = {}
    exec_outputs: dict[tuple[str, str], str] = {}
    state_adjacency: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for edge in document.edges:
        _validate_token(edge.edge_id, "edge_id")
        source = nodes.get(edge.from_node_id)
        target = nodes.get(edge.to_node_id)
        if source is None or target is None:
            _fail("missing_node", f"edge {edge.edge_id} references a missing node")
        source_port = _node_port_map(source).get(edge.from_port_id)
        target_port = _node_port_map(target).get(edge.to_port_id)
        if source_port is None or target_port is None:
            _fail("missing_port", f"edge {edge.edge_id} references a missing port")
        if source_port.direction != PortDirection.output or target_port.direction != PortDirection.input:
            _fail("invalid_direction", f"edge {edge.edge_id} must connect output to input")
        if _edge_port_kind(source_port) != edge.kind or _edge_port_kind(target_port) != edge.kind:
            _fail("edge_kind_mismatch", f"edge {edge.edge_id} kind does not match its ports")
        if edge.queue_size < 1 or (edge.timeout_ms is not None and edge.timeout_ms < 0):
            _fail("invalid_edge_policy", f"edge {edge.edge_id} has invalid queue/timeout values")
        if edge.kind != GraphEdgeKind.data and (edge.strategy.value != "latest" or edge.queue_size != 16):
            _fail("invalid_edge_policy", f"non-data edge {edge.edge_id} cannot define a queue policy")

        input_key = (target.node_id, target_port.port_id)
        source_text = f"{source.node_id}.{source_port.port_id}"
        if edge.kind == GraphEdgeKind.data:
            previous = data_inputs.get(input_key)
            if previous is not None and previous != source_text:
                _fail("multiple_data_upstreams", f"data input has multiple upstreams: {target.node_id}.{target_port.name}")
            data_inputs[input_key] = source_text
            if source_port.data_spec is None or target_port.data_spec is None:
                _fail("missing_data_spec", f"data edge {edge.edge_id} has no data schema")
            if data_port_payload_kind(source_port.data_spec) != data_port_payload_kind(target_port.data_spec):
                _fail("data_payload_mismatch", f"data payload mismatch on edge {edge.edge_id}")
        elif edge.kind == GraphEdgeKind.exec:
            if not isinstance(source, OperatorNode) or not isinstance(target, OperatorNode):
                _fail("invalid_exec_endpoint", f"exec edge {edge.edge_id} requires operator endpoints")
            if source.service_id != target.service_id:
                _fail("cross_service_exec", f"exec edge {edge.edge_id} crosses services")
            previous_in = exec_inputs.get(input_key)
            output_key = (source.node_id, source_port.port_id)
            previous_out = exec_outputs.get(output_key)
            if previous_in is not None and previous_in != source_text:
                _fail("multiple_exec_upstreams", f"exec input has multiple upstreams: {target.node_id}.{target_port.name}")
            target_text = f"{target.node_id}.{target_port.port_id}"
            if previous_out is not None and previous_out != target_text:
                _fail("multiple_exec_downstreams", f"exec output has multiple downstreams: {source.node_id}.{source_port.name}")
            exec_inputs[input_key] = source_text
            exec_outputs[output_key] = target_text
        else:
            previous = state_inputs.get(input_key)
            if previous is not None and previous != source_text:
                _fail("multiple_state_upstreams", f"state input has multiple upstreams: {target.node_id}.{target_port.name}")
            state_inputs[input_key] = source_text
            if source_port.state_spec is None or target_port.state_spec is None:
                _fail("missing_state_spec", f"state edge {edge.edge_id} has no state schema")
            if source_port.state_spec.access == F8StateAccess.wo:
                _fail("writeonly_state_source", f"state source is write-only: {source.node_id}.{source_port.name}")
            if target_port.state_spec.access == F8StateAccess.ro:
                _fail("readonly_state_target", f"state target is read-only: {target.node_id}.{target_port.name}")
            source_key = (source.node_id, source_port.runtime_name)
            target_key = (target.node_id, target_port.runtime_name)
            state_adjacency.setdefault(source_key, []).append(target_key)

    visiting: set[tuple[str, str]] = set()
    visited: set[tuple[str, str]] = set()

    def visit(key: tuple[str, str]) -> None:
        if key in visiting:
            _fail("state_cycle", f"cyclic state edge detected at {key[0]}.{key[1]}")
        if key in visited:
            return
        visiting.add(key)
        for downstream in state_adjacency.get(key, []):
            visit(downstream)
        visiting.remove(key)
        visited.add(key)

    for source_key in state_adjacency:
        visit(source_key)


def validate_document(document: StudioDocument) -> None:
    if document.schema_version != DOCUMENT_SCHEMA_VERSION:
        _fail("unsupported_document_version", f"unsupported schemaVersion: {document.schema_version}")
    _validate_token(document.project_id, "project_id")
    _validate_token(document.graph_id, "graph_id")
    if document.graph_revision < 0 or document.layout_revision < 0:
        _fail("invalid_revision", "document revisions must be non-negative")

    node_ids = [node.node_id for node in document.nodes]
    edge_ids = [edge.edge_id for edge in document.edges]
    layout_ids = [layout.node_id for layout in document.layout]
    if len(node_ids) != len(set(node_ids)):
        _fail("duplicate_node", "document contains duplicate node ids")
    if len(edge_ids) != len(set(edge_ids)):
        _fail("duplicate_edge", "document contains duplicate edge ids")
    if len(layout_ids) != len(set(layout_ids)):
        _fail("duplicate_layout", "document contains duplicate node layout entries")

    service_nodes = {node.service_id: node for node in document.nodes if isinstance(node, ServiceNode)}
    nodes = {node.node_id: node for node in document.nodes}
    for node in document.nodes:
        _validate_node(node, service_nodes)
    for layout in document.layout:
        if layout.node_id not in nodes:
            _fail("missing_layout_node", f"layout references missing node {layout.node_id}")
        values = (layout.x, layout.y, layout.width, layout.height)
        if any(value is not None and not math.isfinite(value) for value in values):
            _fail("invalid_layout", f"layout contains a non-finite number for {layout.node_id}")
        if layout.width is not None and layout.width <= 0:
            _fail("invalid_layout", f"layout width must be positive for {layout.node_id}")
        if layout.height is not None and layout.height <= 0:
            _fail("invalid_layout", f"layout height must be positive for {layout.node_id}")
    _validate_edges(document, nodes)
