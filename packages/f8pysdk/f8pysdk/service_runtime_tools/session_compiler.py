from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from f8pysdk.generated import (
    F8Edge,
    F8EdgeDirection,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8OperatorSpec,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8RuntimeService,
    F8ServiceSpec,
)
from f8pysdk.rungraph_validation import (
    validate_data_edges_or_raise,
    validate_exec_edges_or_raise,
    validate_state_edge_targets_writable_or_raise,
    validate_state_edges_or_raise,
)

from .catalog import ServiceCatalog


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledRuntimeGraphs:
    global_graph: F8RuntimeGraph
    per_service: dict[str, F8RuntimeGraph]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _KeptNode:
    node_id: str
    service_id: str
    is_service_node: bool
    spec: F8ServiceSpec | F8OperatorSpec
    runtime_node: F8RuntimeNode


def _port_kind(name: str) -> F8EdgeKindEnum | None:
    n = str(name or "")
    if n.startswith("[E]") or n.endswith("[E]"):
        return F8EdgeKindEnum.exec
    if n.startswith("[D]") or n.endswith("[D]"):
        return F8EdgeKindEnum.data
    if n.startswith("[S]") or n.endswith("[S]"):
        return F8EdgeKindEnum.state
    return None


def _raw_port_name(name: str) -> str:
    n = str(name or "")
    for prefix in ("[E]", "[D]", "[S]"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    for suffix in ("[E]", "[D]", "[S]"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.strip()


def _is_disabled_node(node_data: dict[str, Any]) -> bool:
    if bool(node_data.get("disabled")):
        return True
    widgets = node_data.get("widgets")
    if isinstance(widgets, dict) and bool(widgets.get("disabled")):
        return True
    return False


def _coerce_spec(spec_payload: Any) -> F8ServiceSpec | F8OperatorSpec:
    if not isinstance(spec_payload, dict):
        raise ValueError("node f8_spec must be an object")
    if "operatorClass" in spec_payload:
        return F8OperatorSpec.model_validate(spec_payload)
    return F8ServiceSpec.model_validate(spec_payload)


def _node_custom_map(node_data: dict[str, Any]) -> dict[str, Any]:
    custom = node_data.get("custom")
    if isinstance(custom, dict):
        return custom
    return {}


def _node_f8_sys_map(node_data: dict[str, Any]) -> dict[str, Any]:
    f8_sys = node_data.get("f8_sys")
    if isinstance(f8_sys, dict):
        return f8_sys
    return {}


def _node_properties_map(node_data: dict[str, Any]) -> dict[str, Any]:
    props = node_data.get("properties")
    if isinstance(props, dict):
        return props
    return {}


def _collect_state_values(spec: F8ServiceSpec | F8OperatorSpec, node_data: dict[str, Any]) -> dict[str, Any] | None:
    custom = _node_custom_map(node_data)
    properties = _node_properties_map(node_data)
    values: dict[str, Any] = {}
    for field in list(spec.stateFields or []):
        name = str(field.name or "").strip()
        if not name:
            continue
        if str(field.access) == "F8StateAccess.ro" or field.access.value == "ro":
            continue
        if name in custom:
            values[name] = custom[name]
            continue
        if name in properties:
            values[name] = properties[name]
    return values or None


def _resolve_operator_service_id(node_id: str, node_data: dict[str, Any]) -> str:
    custom = _node_custom_map(node_data)
    service_id = str(custom.get("svcId") or node_data.get("svcId") or "").strip()
    if not service_id:
        raise ValueError(f"operator node '{node_id}' missing svcId")
    return service_id


def _runtime_service_label(spec: F8ServiceSpec | F8OperatorSpec, node_data: dict[str, Any]) -> str | None:
    if isinstance(spec, F8ServiceSpec):
        custom = _node_custom_map(node_data)
        name = str(custom.get("name") or node_data.get("name") or "").strip()
        if name:
            return name
        return str(spec.label or "").strip() or None
    return None


def _compile_kept_nodes(
    *,
    layout_nodes: dict[str, Any],
    catalog: ServiceCatalog,
    pystudio_service_class: str,
    warnings: list[str],
) -> tuple[dict[str, _KeptNode], list[F8RuntimeService]]:
    kept_by_id: dict[str, _KeptNode] = {}
    runtime_services: dict[str, F8RuntimeService] = {}

    for raw_node_id, raw_node_data in layout_nodes.items():
        node_id = str(raw_node_id or "").strip()
        if not node_id or not isinstance(raw_node_data, dict):
            continue
        f8_sys = _node_f8_sys_map(raw_node_data)
        if bool(f8_sys.get("missingLocked")):
            missing_type = str(f8_sys.get("missingType") or "").strip()
            raise ValueError(
                f"session contains missing dependency node '{node_id}'"
                + (f" (type='{missing_type}')" if missing_type else "")
            )
        if _is_disabled_node(raw_node_data):
            warnings.append(f"skip disabled node: {node_id}")
            continue

        spec = _coerce_spec(raw_node_data.get("f8_spec"))
        service_class = str(spec.serviceClass or "").strip()
        if not service_class:
            raise ValueError(f"node '{node_id}' missing serviceClass in f8_spec")

        if service_class == str(pystudio_service_class):
            warnings.append(f"skip pystudio node: {node_id}")
            continue

        if not catalog.services.has(service_class):
            raise ValueError(f"unknown serviceClass '{service_class}' for node '{node_id}'")

        is_service_node = isinstance(spec, F8ServiceSpec)
        if not is_service_node:
            operator_spec = spec
            operator_class = str(operator_spec.operatorClass or "").strip()
            if not catalog.operators.has(service_class, operator_class):
                raise ValueError(
                    f"unknown operatorClass '{operator_class}' for serviceClass '{service_class}' (node '{node_id}')"
                )

        service_id = node_id if is_service_node else _resolve_operator_service_id(node_id, raw_node_data)
        runtime_node = F8RuntimeNode(
            nodeId=node_id,
            serviceId=service_id,
            serviceClass=service_class,
            operatorClass=(None if is_service_node else str(spec.operatorClass)),
            execInPorts=([] if is_service_node else [str(p) for p in list(spec.execInPorts or [])]),
            execOutPorts=([] if is_service_node else [str(p) for p in list(spec.execOutPorts or [])]),
            dataInPorts=list(spec.dataInPorts or []),
            dataOutPorts=list(spec.dataOutPorts or []),
            stateFields=list(spec.stateFields or []),
            stateValues=_collect_state_values(spec, raw_node_data),
        )
        kept_by_id[node_id] = _KeptNode(
            node_id=node_id,
            service_id=service_id,
            is_service_node=is_service_node,
            spec=spec,
            runtime_node=runtime_node,
        )

        if service_id not in runtime_services:
            runtime_services[service_id] = F8RuntimeService(
                serviceId=service_id,
                serviceClass=service_class,
                label=_runtime_service_label(spec, raw_node_data),
                meta={},
            )
    return kept_by_id, list(runtime_services.values())


def _compile_edges(layout_connections: list[Any], kept_by_id: dict[str, _KeptNode], warnings: list[str]) -> list[F8Edge]:
    edges: list[F8Edge] = []
    for raw_conn in layout_connections:
        if not isinstance(raw_conn, dict):
            continue
        out_ref = raw_conn.get("out")
        in_ref = raw_conn.get("in")
        if not (isinstance(out_ref, (list, tuple)) and len(out_ref) == 2):
            continue
        if not (isinstance(in_ref, (list, tuple)) and len(in_ref) == 2):
            continue

        from_id = str(out_ref[0] or "").strip()
        to_id = str(in_ref[0] or "").strip()
        from_port_raw = str(out_ref[1] or "")
        to_port_raw = str(in_ref[1] or "")

        if from_id not in kept_by_id or to_id not in kept_by_id:
            continue

        kind = _port_kind(from_port_raw)
        if kind is None:
            kind = _port_kind(to_port_raw)
        if kind is None:
            warnings.append(f"skip connection with unknown port kind: {from_id}.{from_port_raw} -> {to_id}.{to_port_raw}")
            continue

        from_node = kept_by_id[from_id]
        to_node = kept_by_id[to_id]
        edges.append(
            F8Edge(
                edgeId=uuid4().hex,
                fromServiceId=from_node.service_id,
                fromOperatorId=(None if from_node.is_service_node else from_node.node_id),
                fromPort=_raw_port_name(from_port_raw),
                toServiceId=to_node.service_id,
                toOperatorId=(None if to_node.is_service_node else to_node.node_id),
                toPort=_raw_port_name(to_port_raw),
                kind=kind,
                strategy=F8EdgeStrategyEnum.latest,
                timeoutMs=None,
                direction=None,
            )
        )
    return edges


def split_runtime_graph_by_service(graph: F8RuntimeGraph) -> dict[str, F8RuntimeGraph]:
    def _with_direction(edge: F8Edge, direction: F8EdgeDirection) -> F8Edge:
        payload = edge.model_dump(mode="json", by_alias=True)
        payload["direction"] = direction
        return F8Edge.model_validate(payload)

    by_service_nodes: dict[str, list[F8RuntimeNode]] = {}
    for node in list(graph.nodes or []):
        by_service_nodes.setdefault(str(node.serviceId), []).append(node)

    by_service_edges: dict[str, list[F8Edge]] = {}
    for edge in list(graph.edges or []):
        from_sid = str(edge.fromServiceId)
        to_sid = str(edge.toServiceId)
        if from_sid == to_sid:
            by_service_edges.setdefault(from_sid, []).append(edge)
            continue
        by_service_edges.setdefault(from_sid, []).append(_with_direction(edge, F8EdgeDirection.out))
        by_service_edges.setdefault(to_sid, []).append(_with_direction(edge, F8EdgeDirection.in_))

    out: dict[str, F8RuntimeGraph] = {}
    for service in list(graph.services or []):
        sid = str(service.serviceId)
        out[sid] = F8RuntimeGraph(
            graphId=graph.graphId,
            revision=graph.revision,
            services=[service],
            nodes=by_service_nodes.get(sid, []),
            edges=by_service_edges.get(sid, []),
            meta=graph.meta,
        )
    return out


def compile_runtime_graphs_from_session_layout(
    *,
    layout: dict[str, Any],
    catalog: ServiceCatalog,
    pystudio_service_class: str = "f8.pystudio",
    graph_id: str = "session",
    revision: str = "1",
) -> CompiledRuntimeGraphs:
    nodes_obj = layout.get("nodes")
    if not isinstance(nodes_obj, dict):
        raise ValueError("session layout missing `nodes` object")
    connections_obj = layout.get("connections")
    if connections_obj is None:
        connection_list: list[Any] = []
    elif isinstance(connections_obj, list):
        connection_list = connections_obj
    else:
        raise ValueError("session layout `connections` must be a list")

    warnings: list[str] = []
    kept_by_id, runtime_services = _compile_kept_nodes(
        layout_nodes=nodes_obj,
        catalog=catalog,
        pystudio_service_class=str(pystudio_service_class),
        warnings=warnings,
    )
    runtime_nodes = [entry.runtime_node for entry in kept_by_id.values()]
    edges = _compile_edges(connection_list, kept_by_id, warnings)

    graph = F8RuntimeGraph(
        graphId=str(graph_id),
        revision=str(revision),
        services=runtime_services,
        nodes=runtime_nodes,
        edges=edges,
    )
    validate_exec_edges_or_raise(graph)
    validate_data_edges_or_raise(graph)
    validate_state_edges_or_raise(graph, forbid_cycles=True, forbid_multi_upstream=True)
    validate_state_edge_targets_writable_or_raise(graph)
    return CompiledRuntimeGraphs(
        global_graph=graph,
        per_service=split_runtime_graph_by_service(graph),
        warnings=tuple(warnings),
    )
