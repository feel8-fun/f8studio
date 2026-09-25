from __future__ import annotations

import hashlib

import msgspec

from f8pysdk.codec import coerce_int
from f8pysdk.command import hidden_command_state_specs
from f8pysdk.rungraph_validation import (
    validate_data_edges_or_raise,
    validate_exec_edges_or_raise,
    validate_state_edge_targets_writable_or_raise,
    validate_state_edges_or_raise,
)
from f8pysdk.specs import (
    F8AutoSampleRequest,
    F8ArrayTypeSchema,
    F8ComplexObjectTypeSchema,
    F8DataPortSpec,
    F8DataTypeSchema,
    F8Edge,
    F8EdgeDirection,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8RuntimeService,
    F8StateAccess,
    F8StateSpec,
    operator_state_fields_with_builtins,
    service_state_fields_with_builtins,
)

from .graph.models import (
    EdgeStrategy,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
    GraphPort,
    OperatorNode,
    ServiceNode,
    StudioDocument,
)
from .graph.codec import canonical_json_bytes
from .graph.validation import validate_document


PATCH_HUB_OPERATOR_CLASS = "f8.patch_hub"
STUDIO_SERVICE_ID = "studio"
PYENGINE_SERVICE_CLASS = "f8.pyengine"
CPPENGINE_SERVICE_CLASS = "f8.cppengine"


class CompiledRuntimeGraphs(msgspec.Struct, frozen=True, kw_only=True):
    global_graph: F8RuntimeGraph
    per_service: dict[str, F8RuntimeGraph]
    warnings: tuple[str, ...] = ()


def _optional_text(value: object) -> str:
    if value is None or isinstance(value, msgspec.UnsetType):
        return ""
    return str(value).strip()


def _spec_data_ports(value: list[F8DataPortSpec] | msgspec.UnsetType) -> list[F8DataPortSpec]:
    return [] if isinstance(value, msgspec.UnsetType) else list(value)


def _spec_state_fields(value: list[F8StateSpec] | msgspec.UnsetType) -> list[F8StateSpec]:
    return [] if isinstance(value, msgspec.UnsetType) else list(value)


def _spec_text_ports(value: list[str] | msgspec.UnsetType) -> list[str]:
    return [] if isinstance(value, msgspec.UnsetType) else list(value)


def _port(node: GraphNode, port_id: str) -> GraphPort:
    for item in node.ports:
        if item.port_id == port_id:
            return item
    raise ValueError(f"port not found after document validation: {node.node_id}.{port_id}")


def _edge_kind(kind: GraphEdgeKind) -> F8EdgeKindEnum:
    if kind == GraphEdgeKind.data:
        return F8EdgeKindEnum.data
    if kind == GraphEdgeKind.state:
        return F8EdgeKindEnum.state
    return F8EdgeKindEnum.exec


def _edge_strategy(strategy: EdgeStrategy) -> F8EdgeStrategyEnum:
    return F8EdgeStrategyEnum.queue if strategy == EdgeStrategy.queue else F8EdgeStrategyEnum.latest


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _runtime_services(graph: F8RuntimeGraph) -> list[F8RuntimeService]:
    return [] if isinstance(graph.services, msgspec.UnsetType) else list(graph.services)


def _runtime_nodes(graph: F8RuntimeGraph) -> list[F8RuntimeNode]:
    return [] if isinstance(graph.nodes, msgspec.UnsetType) else list(graph.nodes)


def _runtime_edges(graph: F8RuntimeGraph) -> list[F8Edge]:
    return [] if isinstance(graph.edges, msgspec.UnsetType) else list(graph.edges)


def semantic_graph_revision(document: StudioDocument) -> str:
    validate_document(document)
    enabled_nodes = {node.node_id: node for node in document.nodes if node.enabled}
    payload = {
        "graphId": document.graph_id,
        "nodes": [
            _semantic_runtime_node(node)
            for node in sorted(enabled_nodes.values(), key=lambda item: item.node_id)
        ],
        "edges": [
            _runtime_edge(edge, enabled_nodes)
            for edge in sorted(document.edges, key=lambda item: item.edge_id)
            if edge.from_node_id in enabled_nodes and edge.to_node_id in enabled_nodes
        ],
        "launch": [
            (node.service_id, None if isinstance(node.spec.launch, msgspec.UnsetType) else node.spec.launch)
            for node in sorted(enabled_nodes.values(), key=lambda item: item.node_id)
            if isinstance(node, ServiceNode)
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _semantic_runtime_node(node: GraphNode) -> F8RuntimeNode:
    runtime = msgspec.json.decode(msgspec.json.encode(_runtime_node(node)), type=F8RuntimeNode)
    states = [
        msgspec.structs.replace(
            field,
            label=msgspec.UNSET,
            description=msgspec.UNSET,
            editPolicy=msgspec.UNSET,
            uiControl=msgspec.UNSET,
            control=msgspec.UNSET,
            showOnNode=msgspec.UNSET,
            redactOnPublish=msgspec.UNSET,
            editorAssist=msgspec.UNSET,
            valueSchema=_semantic_value_schema(field.valueSchema),
        )
        for field in _spec_state_fields(runtime.stateFields)
    ]
    inputs = [
        _semantic_data_port(port)
        for port in _spec_data_ports(runtime.dataInPorts)
    ]
    outputs = [
        _semantic_data_port(port)
        for port in _spec_data_ports(runtime.dataOutPorts)
    ]
    return msgspec.structs.replace(runtime, stateFields=states, dataInPorts=inputs, dataOutPorts=outputs)


def _semantic_value_schema(schema: F8DataTypeSchema) -> F8DataTypeSchema:
    stripped = msgspec.structs.replace(
        schema,
        title=msgspec.UNSET,
        description=msgspec.UNSET,
        examples=msgspec.UNSET,
        field_comment=msgspec.UNSET,
    )
    if isinstance(stripped, F8ArrayTypeSchema):
        return msgspec.structs.replace(stripped, items=_semantic_value_schema(stripped.items))
    if isinstance(stripped, F8ComplexObjectTypeSchema):
        return msgspec.structs.replace(
            stripped,
            properties={name: _semantic_value_schema(value) for name, value in stripped.properties.items()},
        )
    return stripped


def _semantic_data_port(port: F8DataPortSpec) -> F8DataPortSpec:
    payload = port.payload
    if not isinstance(payload, msgspec.UnsetType):
        payload = msgspec.structs.replace(
            payload,
            valueSchema=payload.valueSchema if isinstance(payload.valueSchema, msgspec.UnsetType)
            else _semantic_value_schema(payload.valueSchema),
            metadataSchema=payload.metadataSchema if isinstance(payload.metadataSchema, msgspec.UnsetType)
            else _semantic_value_schema(payload.metadataSchema),
        )
    return msgspec.structs.replace(
        port,
        valueSchema=_semantic_value_schema(port.valueSchema),
        payload=payload,
        description=msgspec.UNSET,
        required=msgspec.UNSET,
        showOnNode=msgspec.UNSET,
    )


def _runtime_state_fields(node: GraphNode) -> list[F8StateSpec]:
    base = _spec_state_fields(node.spec.stateFields)
    commands = [] if isinstance(node.spec.commands, msgspec.UnsetType) else list(node.spec.commands)
    fields = [*base, *hidden_command_state_specs(commands)]
    if isinstance(node, ServiceNode):
        return service_state_fields_with_builtins(fields)
    return operator_state_fields_with_builtins(fields)


def _runtime_node(node: GraphNode) -> F8RuntimeNode:
    writable_fields = {
        str(field.name)
        for field in _spec_state_fields(node.spec.stateFields)
        if field.access != F8StateAccess.ro
    }
    state_values = {name: value for name, value in node.state_values.items() if name in writable_fields}
    if isinstance(node, OperatorNode):
        return F8RuntimeNode(
            nodeId=node.node_id,
            serviceId=node.service_id,
            serviceClass=node.service_class,
            operatorClass=node.operator_class,
            execInPorts=_spec_text_ports(node.spec.execInPorts),
            execOutPorts=_spec_text_ports(node.spec.execOutPorts),
            dataInPorts=_spec_data_ports(node.spec.dataInPorts),
            dataOutPorts=_spec_data_ports(node.spec.dataOutPorts),
            stateFields=_runtime_state_fields(node),
            stateValues=state_values or msgspec.UNSET,
        )
    return F8RuntimeNode(
        nodeId=node.node_id,
        serviceId=node.service_id,
        serviceClass=node.service_class,
        operatorClass=msgspec.UNSET,
        dataInPorts=_spec_data_ports(node.spec.dataInPorts),
        dataOutPorts=_spec_data_ports(node.spec.dataOutPorts),
        stateFields=_runtime_state_fields(node),
        stateValues=state_values or msgspec.UNSET,
    )


def _runtime_edge(edge: GraphEdge, nodes: dict[str, GraphNode]) -> F8Edge:
    source = nodes[edge.from_node_id]
    target = nodes[edge.to_node_id]
    source_port = _port(source, edge.from_port_id)
    target_port = _port(target, edge.to_port_id)
    return F8Edge(
        edgeId=edge.edge_id,
        fromServiceId=source.service_id,
        fromOperatorId=source.node_id if isinstance(source, OperatorNode) else msgspec.UNSET,
        fromPort=source_port.runtime_name,
        toServiceId=target.service_id,
        toOperatorId=target.node_id if isinstance(target, OperatorNode) else msgspec.UNSET,
        toPort=target_port.runtime_name,
        kind=_edge_kind(edge.kind),
        strategy=_edge_strategy(edge.strategy),
        queueSize=edge.queue_size,
        timeoutMs=edge.timeout_ms if edge.timeout_ms is not None else msgspec.UNSET,
        direction=msgspec.UNSET,
    )


def _remove_upstream_state_values(graph: F8RuntimeGraph) -> F8RuntimeGraph:
    bound: dict[str, set[str]] = {}
    for edge in _runtime_edges(graph):
        if edge.kind != F8EdgeKindEnum.state:
            continue
        node_id = _optional_text(edge.toOperatorId) or str(edge.toServiceId)
        field = str(edge.toPort).strip()
        if node_id and field:
            bound.setdefault(node_id, set()).add(field)

    nodes: list[F8RuntimeNode] = []
    for node in _runtime_nodes(graph):
        fields = bound.get(str(node.nodeId))
        if not fields or isinstance(node.stateValues, msgspec.UnsetType):
            nodes.append(node)
            continue
        values = {name: value for name, value in node.stateValues.items() if name not in fields}
        nodes.append(msgspec.structs.replace(node, stateValues=values or msgspec.UNSET))
    return msgspec.structs.replace(graph, nodes=nodes)


def _lower_patch_hubs(graph: F8RuntimeGraph) -> tuple[F8RuntimeGraph, list[str]]:
    hub_ids = {
        str(node.nodeId)
        for node in _runtime_nodes(graph)
        if _optional_text(node.operatorClass) == PATCH_HUB_OPERATOR_CLASS
    }
    if not hub_ids:
        return graph, []

    warnings: list[str] = []
    kept: list[F8Edge] = []
    inbound: dict[tuple[str, str, str], list[F8Edge]] = {}
    outbound: dict[tuple[str, str, str], list[F8Edge]] = {}
    for edge in _runtime_edges(graph):
        source_id = _optional_text(edge.fromOperatorId)
        target_id = _optional_text(edge.toOperatorId)
        source_hub = source_id in hub_ids
        target_hub = target_id in hub_ids
        if not source_hub and not target_hub:
            kept.append(edge)
            continue
        if edge.kind not in {F8EdgeKindEnum.data, F8EdgeKindEnum.state}:
            raise ValueError(f"patch hub only supports data/state edges: {edge.edgeId}")
        if source_hub:
            outbound.setdefault((source_id, edge.kind.value, str(edge.fromPort)), []).append(edge)
        if target_hub:
            inbound.setdefault((target_id, edge.kind.value, str(edge.toPort)), []).append(edge)

    for key, edges in inbound.items():
        if len(edges) > 1:
            raise ValueError(f"patch hub terminal has multiple upstreams: {'.'.join(key)}")

    def resolve(key: tuple[str, str, str], stack: tuple[tuple[str, str, str], ...]) -> F8Edge | None:
        if key in stack:
            raise ValueError("patch hub cycle detected: " + " -> ".join(".".join(item) for item in (*stack, key)))
        edges = inbound.get(key)
        if not edges:
            return None
        source = edges[0]
        source_id = _optional_text(source.fromOperatorId)
        if source_id in hub_ids:
            return resolve((source_id, source.kind.value, str(source.fromPort)), (*stack, key))
        return source

    lowered: list[F8Edge] = []
    for key in sorted(set(inbound) | set(outbound)):
        inputs = inbound.get(key, [])
        outputs = outbound.get(key, [])
        if inputs and not outputs:
            warnings.append(f"patch hub terminal has no downstream consumers: {'.'.join(key)}")
            continue
        if outputs and not inputs:
            warnings.append(f"patch hub terminal has no upstream source: {'.'.join(key)}")
            continue
        source = resolve(key, ())
        if source is None:
            continue
        for target in outputs:
            target_id = _optional_text(target.toOperatorId)
            if target_id in hub_ids:
                continue
            lowered.append(
                F8Edge(
                    edgeId=_stable_id(
                        "patch",
                        source.kind.value,
                        str(source.fromServiceId),
                        _optional_text(source.fromOperatorId),
                        str(source.fromPort),
                        str(target.toServiceId),
                        target_id,
                        str(target.toPort),
                    ),
                    fromServiceId=source.fromServiceId,
                    fromOperatorId=source.fromOperatorId,
                    fromPort=source.fromPort,
                    toServiceId=target.toServiceId,
                    toOperatorId=target.toOperatorId,
                    toPort=target.toPort,
                    kind=target.kind,
                    strategy=target.strategy,
                    queueSize=target.queueSize,
                    timeoutMs=target.timeoutMs,
                    direction=msgspec.UNSET,
                )
            )

    deduplicated: dict[tuple[str, str, str, str, str, str, str], F8Edge] = {}
    for edge in (*kept, *lowered):
        key = (
            edge.kind.value,
            str(edge.fromServiceId),
            _optional_text(edge.fromOperatorId),
            str(edge.fromPort),
            str(edge.toServiceId),
            _optional_text(edge.toOperatorId),
            str(edge.toPort),
        )
        deduplicated[key] = edge
    nodes = [node for node in _runtime_nodes(graph) if str(node.nodeId) not in hub_ids]
    edges = [deduplicated[key] for key in sorted(deduplicated)]
    return msgspec.structs.replace(graph, nodes=nodes, edges=edges), warnings


def _attach_auto_sample_requests(graph: F8RuntimeGraph) -> F8RuntimeGraph:
    services = {str(service.serviceId): service for service in _runtime_services(graph)}
    nodes = {str(node.nodeId): node for node in _runtime_nodes(graph)}
    grouped: dict[tuple[str, str, str], int] = {}
    for edge in _runtime_edges(graph):
        if edge.kind != F8EdgeKindEnum.data or str(edge.toServiceId) != STUDIO_SERVICE_ID:
            continue
        if str(edge.fromServiceId) == str(edge.toServiceId):
            continue
        target_id = _optional_text(edge.toOperatorId)
        source_id = _optional_text(edge.fromOperatorId)
        target = nodes.get(target_id)
        if target is None or not source_id or isinstance(target.stateValues, msgspec.UnsetType):
            continue
        values = target.stateValues
        mode = str(values.get("upstreamSamplingMode", "auto")).strip().lower()
        if mode != "auto":
            continue
        interval = coerce_int(values.get("upstreamSampleIntervalMs", 100), default=100, minimum=8, maximum=5000)
        key = (str(edge.fromServiceId), source_id, str(edge.fromPort))
        grouped[key] = min(grouped.get(key, interval), interval)

    updated = dict(services)
    for (service_id, node_id, port), interval in sorted(grouped.items()):
        service = updated.get(service_id)
        if service is None or str(service.serviceClass) not in {PYENGINE_SERVICE_CLASS, CPPENGINE_SERVICE_CLASS}:
            continue
        requests = [] if isinstance(service.autoSampleRequests, msgspec.UnsetType) else list(service.autoSampleRequests)
        requests.append(
            F8AutoSampleRequest(
                sourceNodeId=node_id,
                sourcePort=port,
                intervalMs=interval,
                deliverLocal=False,
                publishCrossService=True,
            )
        )
        updated[service_id] = msgspec.structs.replace(service, autoSampleRequests=requests)
    return msgspec.structs.replace(
        graph,
        services=[updated[str(service.serviceId)] for service in _runtime_services(graph)],
    )


def split_runtime_graph_by_service(graph: F8RuntimeGraph) -> dict[str, F8RuntimeGraph]:
    nodes: dict[str, list[F8RuntimeNode]] = {}
    edges: dict[str, list[F8Edge]] = {}
    for node in _runtime_nodes(graph):
        nodes.setdefault(str(node.serviceId), []).append(node)
    for edge in _runtime_edges(graph):
        source = str(edge.fromServiceId)
        target = str(edge.toServiceId)
        if source == target:
            edges.setdefault(source, []).append(edge)
        else:
            edges.setdefault(source, []).append(msgspec.structs.replace(edge, direction=F8EdgeDirection.out))
            edges.setdefault(target, []).append(msgspec.structs.replace(edge, direction=F8EdgeDirection.in_))

    result: dict[str, F8RuntimeGraph] = {}
    for service in _runtime_services(graph):
        service_id = str(service.serviceId)
        result[service_id] = F8RuntimeGraph(
            graphId=graph.graphId,
            revision=graph.revision,
            services=[service],
            nodes=nodes.get(service_id, []),
            edges=edges.get(service_id, []),
        )
    return result


def compile_document(document: StudioDocument) -> CompiledRuntimeGraphs:
    validate_document(document)
    enabled_nodes = {node.node_id: node for node in document.nodes if node.enabled}
    enabled_services = {
        node.service_id: node for node in enabled_nodes.values() if isinstance(node, ServiceNode)
    }
    for node in enabled_nodes.values():
        if isinstance(node, OperatorNode) and node.service_id not in enabled_services:
            raise ValueError(f"enabled operator {node.node_id} belongs to a disabled or missing service")

    services = [
        F8RuntimeService(
            serviceId=node.service_id,
            serviceClass=node.service_class,
        )
        for node in sorted(enabled_services.values(), key=lambda item: item.service_id)
    ]
    runtime_nodes = [_semantic_runtime_node(node) for node in sorted(enabled_nodes.values(), key=lambda item: item.node_id)]
    runtime_edges = [
        _runtime_edge(edge, enabled_nodes)
        for edge in sorted(document.edges, key=lambda item: item.edge_id)
        if edge.from_node_id in enabled_nodes and edge.to_node_id in enabled_nodes
    ]
    graph = F8RuntimeGraph(
        graphId=document.graph_id,
        revision=semantic_graph_revision(document),
        services=services,
        nodes=runtime_nodes,
        edges=runtime_edges,
    )
    graph = _remove_upstream_state_values(graph)
    graph, warnings = _lower_patch_hubs(graph)
    graph = _attach_auto_sample_requests(graph)
    validate_exec_edges_or_raise(graph)
    validate_data_edges_or_raise(graph)
    validate_state_edges_or_raise(graph, forbid_cycles=True, forbid_multi_upstream=True)
    validate_state_edge_targets_writable_or_raise(graph)
    return CompiledRuntimeGraphs(
        global_graph=graph,
        per_service=split_runtime_graph_by_service(graph),
        warnings=tuple(warnings),
    )
