from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from uuid import uuid4

from f8pysdk import (
    F8Edge,
    F8EdgeDirection,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8OperatorSpec,
    F8ServiceSpec,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8RuntimeService,
)
from f8pysdk.nats_naming import ensure_token


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

def _port_name(port: Any) -> str:
    """
    NodeGraphQt `Port` exposes `name()` (method), not `.name` (attribute).
    """
    name_attr = getattr(port, "name", None)
    if callable(name_attr):
        try:
            return str(name_attr() or "")
        except Exception:
            return ""
    return str(name_attr or "")


def _node_name(node: Any) -> str:
    """
    NodeGraphQt `BaseNode` exposes `name()` (method), not `.name` (attribute).
    """
    name_attr = getattr(node, "name", None)
    if callable(name_attr):
        try:
            return str(name_attr() or "")
        except Exception:
            return ""
    return str(name_attr or "")


def _runtime_node_id(node: Any) -> str:
    return ensure_token(str(getattr(node, "id")), label="node_id")


def _runtime_service_id(node: Any) -> str:
    spec = getattr(node, "spec", None)
    # Containers represent service instances themselves: their id is the serviceId.
    if isinstance(spec, F8ServiceSpec):
        return ensure_token(str(getattr(node, "id")), label="service_id")
    # Operators are bound to a container: svcId points at the container id.
    return ensure_token(str(getattr(node, "svcId")), label="service_id")


@dataclass(frozen=True)
class CompiledRuntimeGraphs:
    global_graph: F8RuntimeGraph
    per_service: dict[str, F8RuntimeGraph]


def compile_global_runtime_graph(
    *,
    services: Iterable[Any],
    operators: Iterable[Any],
    service_nodes: Iterable[Any] | None = None,
    graph_id: str | None = None,
    revision: str = "1",
) -> F8RuntimeGraph:
    """
    Compile studio nodes into a single global runtime graph.

    - `services` are container nodes (service instances).
    - `operators` are operator nodes (executable nodes bound to a container).
    """
    gid = ensure_token(graph_id or uuid4().hex, label="graph_id")
    rev = ensure_token(str(revision), label="revision")

    # Service instances (containers + standalone single-node services).
    runtime_services: dict[str, F8RuntimeService] = {}

    def add_runtime_service(node: Any) -> None:
        service_id = _runtime_service_id(node)
        spec = getattr(node, "spec", None)
        if not isinstance(spec, F8ServiceSpec):
            return
        meta: dict[str, Any] = {}
        instance_name = _node_name(node).strip()
        if instance_name:
            meta["name"] = instance_name
        runtime_services[service_id] = F8RuntimeService(
            serviceId=service_id,
            serviceClass=str(spec.serviceClass),
            label=str(getattr(spec, "label", None) or "") or None,
            meta=meta,
        )

    for c in services:
        add_runtime_service(c)
    for s in list(service_nodes or []):
        add_runtime_service(s)

    runtime_nodes: list[F8RuntimeNode] = []
    # Include containers too: containers are service instances and should be present as runtime nodes
    # so they can later own telemetry/state/data ports.
    port_nodes: list[Any] = [n for n in operators] + list(service_nodes or []) + [n for n in services]

    id_map: dict[Any, str] = {}
    svc_map: dict[Any, str] = {}
    kind_map: dict[Any, str] = {}
    for n in port_nodes:
        spec = getattr(n, "spec", None)
        if isinstance(spec, F8OperatorSpec):
            kind_map[n] = "operator"
        elif isinstance(spec, F8ServiceSpec):
            kind_map[n] = "service"
        else:
            continue

        node_id = _runtime_node_id(n)
        service_id = _runtime_service_id(n)
        id_map[n] = node_id
        svc_map[n] = service_id

        state_values: dict[str, Any] = {}
        for f in list(getattr(spec, "stateFields", None) or []):
            name = str(getattr(f, "name", "") or "").strip()
            if not name:
                continue
            try:
                if name not in n.model.properties and name not in n.model.custom_properties:
                    continue
                state_values[name] = n.model.get_property(name)
            except Exception:
                continue

        runtime_nodes.append(
            F8RuntimeNode(
                nodeId=node_id,
                serviceId=service_id,
                serviceClass=str(spec.serviceClass),
                operatorClass=(str(spec.operatorClass) if isinstance(spec, F8OperatorSpec) else None),
                execInPorts=[str(p) for p in list(getattr(spec, "execInPorts", None) or [])],
                execOutPorts=[str(p) for p in list(getattr(spec, "execOutPorts", None) or [])],
                dataInPorts=list(getattr(spec, "dataInPorts", None) or []),
                dataOutPorts=list(getattr(spec, "dataOutPorts", None) or []),
                stateFields=list(getattr(spec, "stateFields", None) or []),
                stateValues=state_values or None,
            )
        )

    edges: list[F8Edge] = []
    for src_node in port_nodes:
        if src_node not in id_map:
            continue
        for out_port in list(src_node.output_ports() or []):
            out_name = _port_name(out_port)
            edge_kind = _port_kind(out_name)
            if edge_kind is None:
                continue
            for in_port in list(out_port.connected_ports() or []):
                dst_node = in_port.node()
                if dst_node not in id_map:
                    continue
                in_name = _port_name(in_port)

                edges.append(
                    F8Edge(
                        edgeId=uuid4().hex,
                        fromServiceId=svc_map[src_node],
                        fromOperatorId=(id_map[src_node] if kind_map.get(src_node) != "container" else None),
                        fromPort=_raw_port_name(out_name),
                        toServiceId=svc_map[dst_node],
                        toOperatorId=(id_map[dst_node] if kind_map.get(dst_node) != "container" else None),
                        toPort=_raw_port_name(in_name),
                        kind=edge_kind,
                        strategy=F8EdgeStrategyEnum.latest,
                        timeoutMs=None,
                        direction=None,
                    )
                )

    return F8RuntimeGraph(
        graphId=gid,
        revision=rev,
        services=list(runtime_services.values()),
        nodes=runtime_nodes,
        edges=edges,
    )


def split_runtime_graph_by_service(graph: F8RuntimeGraph) -> dict[str, F8RuntimeGraph]:
    """
    Produce per-service runtime graphs.

    Cross edges are included, but since the peer service's nodes are absent in
    the per-service node list, they naturally act as "half edges".
    """

    def _with_direction(edge: F8Edge, direction: F8EdgeDirection) -> F8Edge:
        # Avoid mutating shared edge instances across per-service graphs.
        try:
            return edge.model_copy(update={"direction": direction})
        except Exception:
            payload = edge.model_dump(mode="json", by_alias=True)
            payload["direction"] = direction
            return F8Edge.model_validate(payload)

    by_service_nodes: dict[str, list[F8RuntimeNode]] = {}
    for n in graph.nodes:
        by_service_nodes.setdefault(str(n.serviceId), []).append(n)

    by_service_edges: dict[str, list[F8Edge]] = {}
    for e in graph.edges:
        from_sid = str(e.fromServiceId)
        to_sid = str(e.toServiceId)
        if to_sid == from_sid:
            by_service_edges.setdefault(from_sid, []).append(e)
            continue

        # Cross-service edges become half-edges in per-service graphs.
        by_service_edges.setdefault(from_sid, []).append(_with_direction(e, F8EdgeDirection.out))
        by_service_edges.setdefault(to_sid, []).append(_with_direction(e, F8EdgeDirection.in_))

    out: dict[str, F8RuntimeGraph] = {}
    for svc in graph.services:
        sid = str(svc.serviceId)
        out[sid] = F8RuntimeGraph(
            graphId=graph.graphId,
            revision=graph.revision,
            services=[svc],
            nodes=by_service_nodes.get(sid, []),
            edges=by_service_edges.get(sid, []),
        )
    return out


def compile_runtime_graphs_from_studio(studio_graph: Any) -> CompiledRuntimeGraphs:
    """
    Convenience wrapper that extracts container/operator nodes from an
    `F8StudioGraph`.
    """
    all_nodes = list(studio_graph.all_nodes() or [])
    if not hasattr(studio_graph, "_is_container_node") or not hasattr(studio_graph, "_is_operator_node"):
        raise TypeError("studio_graph must be an F8StudioGraph (missing type predicates).")

    services = [n for n in all_nodes if studio_graph._is_container_node(n)]  # type: ignore[attr-defined]
    operators = [n for n in all_nodes if studio_graph._is_operator_node(n)]  # type: ignore[attr-defined]

    # Standalone single-node services (non-container F8ServiceSpec nodes).
    service_nodes = [
        n
        for n in all_nodes
        if hasattr(n, "spec")
        and isinstance(getattr(n, "spec", None), F8ServiceSpec)
        and not studio_graph._is_container_node(n)  # type: ignore[attr-defined]
    ]

    global_graph = compile_global_runtime_graph(
        services=services,
        operators=operators,
        service_nodes=service_nodes,
    )
    return CompiledRuntimeGraphs(global_graph=global_graph, per_service=split_runtime_graph_by_service(global_graph))
