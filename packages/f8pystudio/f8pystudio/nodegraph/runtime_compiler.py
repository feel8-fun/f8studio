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
    F8StateAccess,
    F8StateSpec,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8RuntimeService,
)
from f8pysdk.rungraph_validation import validate_state_edges_or_raise
from f8pysdk.schema_helpers import boolean_schema
from f8pysdk.schema_helpers import string_schema
from f8pysdk.nats_naming import ensure_token

from ..pystudio_node_registry import SERVICE_CLASS as STUDIO_SERVICE_CLASS
from ..pystudio_node_registry import STUDIO_SERVICE_ID


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
    try:
        return str(port.name() or "")
    except Exception:
        return ""


def _node_name(node: Any) -> str:
    """
    NodeGraphQt `BaseNode` exposes `name()` (method), not `.name` (attribute).
    """
    try:
        return str(node.name() or "")
    except Exception:
        return ""


def _runtime_node_id(node: Any) -> str:
    return ensure_token(str(node.id), label="node_id")


def _runtime_service_id(node: Any) -> str:
    try:
        spec = node.spec
    except Exception:
        spec = None
    # Containers represent service instances themselves: their id is the serviceId.
    if isinstance(spec, F8ServiceSpec):
        return ensure_token(str(node.id), label="service_id")
    # Studio operators belong to a fixed local service id.
    if isinstance(spec, F8OperatorSpec) and str(spec.serviceClass or "") == STUDIO_SERVICE_CLASS:
        return STUDIO_SERVICE_ID
    # Operators are bound to a container: svcId points at the container id.
    return ensure_token(str(node.svcId), label="service_id")


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
        try:
            spec = node.spec
        except Exception:
            spec = None
        if not isinstance(spec, F8ServiceSpec):
            return
        meta: dict[str, Any] = {}
        instance_name = _node_name(node).strip()
        if instance_name:
            meta["name"] = instance_name
        runtime_services[service_id] = F8RuntimeService(
            serviceId=service_id,
            serviceClass=str(spec.serviceClass),
            label=str(spec.label or "") or None,
            meta=meta,
        )

    for c in services:
        add_runtime_service(c)
    for s in list(service_nodes or []):
        add_runtime_service(s)

    # If the canvas contains studio operators, ensure the studio service instance exists.
    try:
        has_studio_ops = any(
            isinstance(n.spec, F8OperatorSpec) and str(n.spec.serviceClass or "") == STUDIO_SERVICE_CLASS
            for n in operators
        )
    except Exception:
        has_studio_ops = False
    if has_studio_ops and STUDIO_SERVICE_ID not in runtime_services:
        runtime_services[STUDIO_SERVICE_ID] = F8RuntimeService(
            serviceId=STUDIO_SERVICE_ID,
            serviceClass=STUDIO_SERVICE_CLASS,
            label="PyStudio",
            meta={"name": "PyStudio"},
        )

    runtime_nodes: list[F8RuntimeNode] = []
    # Include containers too: containers are service instances and should be present as runtime nodes
    # so they can later own telemetry/state/data ports.
    port_nodes: list[Any] = [n for n in operators] + list(service_nodes or []) + [n for n in services]

    id_map: dict[Any, str] = {}
    svc_map: dict[Any, str] = {}
    kind_map: dict[Any, str] = {}
    for n in port_nodes:
        try:
            spec = n.spec
        except Exception:
            spec = None
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
        for f in list(spec.stateFields or []):
            name = str(f.name or "").strip()
            if not name:
                continue
            # Do not include read-only state values in the rungraph snapshot.
            # These are runtime-owned and may be updated internally (eg. telemetry).
            if f.access == F8StateAccess.ro:
                continue
            try:
                if name not in n.model.properties and name not in n.model.custom_properties:
                    continue
                state_values[name] = n.model.get_property(name)
            except Exception:
                continue
        # NOTE: values for upstream-driven state fields (bound via state edges)
        # are filtered out after compiling edges, so state propagation always
        # takes precedence over this snapshot on repeated deploys.

        state_fields = list(spec.stateFields or [])

        # Built-in identity fields (readonly) for cross-process routing/commands.
        # - svcId: service instance id
        # - operatorId: operator/node id (operators only; service/container nodes omit it)
        existing = {str(sf.name or "") for sf in state_fields}
        if "svcId" not in existing:
            state_fields.append(
                F8StateSpec(
                    name="svcId",
                    label="Service Id",
                    description="Readonly: current service instance id (svcId).",
                    valueSchema=string_schema(),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                )
            )
        if isinstance(spec, F8OperatorSpec) and "operatorId" not in existing:
            state_fields.append(
                F8StateSpec(
                    name="operatorId",
                    label="Operator Id",
                    description="Readonly: current operator/node id (operatorId).",
                    valueSchema=string_schema(),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                )
            )
        if isinstance(spec, F8ServiceSpec):
            # Runtime-level lifecycle state (service-scoped), persisted in KV by the runtime.
            has_active = any(str(sf.name or "") == "active" for sf in state_fields)
            if not has_active:
                state_fields.append(
                    F8StateSpec(
                        name="active",
                        label="Active",
                        description="Service lifecycle state (activate/deactivate).",
                        valueSchema=boolean_schema(default=True),
                        access=F8StateAccess.rw,
                        showOnNode=True,
                    )
                )

        runtime_nodes.append(
            F8RuntimeNode(
                nodeId=node_id,
                serviceId=service_id,
                serviceClass=str(spec.serviceClass),
                operatorClass=(str(spec.operatorClass) if isinstance(spec, F8OperatorSpec) else None),
                execInPorts=([str(p) for p in list(spec.execInPorts or [])] if isinstance(spec, F8OperatorSpec) else []),
                execOutPorts=([str(p) for p in list(spec.execOutPorts or [])] if isinstance(spec, F8OperatorSpec) else []),
                dataInPorts=list(spec.dataInPorts or []),
                dataOutPorts=list(spec.dataOutPorts or []),
                stateFields=state_fields,
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

    # If a state field is upstream-driven (connected via state edge), do not
    # include its current state value in the rungraph snapshot. This avoids
    # "deploy races" where the snapshot temporarily overrides the edge-driven
    # value when redeploying a running graph.
    upstream_state_by_node: dict[str, set[str]] = {}
    for e in edges:
        if e.kind != F8EdgeKindEnum.state:
            continue
        if e.toOperatorId is None:
            continue
        node_id = str(e.toOperatorId)
        field = str(e.toPort or "").strip()
        if not field:
            continue
        upstream_state_by_node.setdefault(node_id, set()).add(field)

    if upstream_state_by_node:
        filtered_nodes: list[F8RuntimeNode] = []
        for rn in runtime_nodes:
            bound = upstream_state_by_node.get(str(rn.nodeId))
            if not bound or not rn.stateValues:
                filtered_nodes.append(rn)
                continue
            new_values = {k: v for k, v in dict(rn.stateValues).items() if str(k) not in bound}
            if new_values == rn.stateValues:
                filtered_nodes.append(rn)
                continue
            filtered_nodes.append(rn.model_copy(update={"stateValues": new_values or None}))
        runtime_nodes = filtered_nodes

    graph = F8RuntimeGraph(
        graphId=gid,
        revision=rev,
        services=list(runtime_services.values()),
        nodes=runtime_nodes,
        edges=edges,
    )
    # Studio-level validation: reject global cyclic state loops early.
    validate_state_edges_or_raise(graph, forbid_cycles=True, forbid_multi_upstream=True)
    return graph


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
    def _is_disabled(n: Any) -> bool:
        try:
            return bool(n.view.disabled)
        except Exception:
            return False
        return False

    all_nodes = [n for n in list(studio_graph.all_nodes() or []) if not _is_disabled(n)]
    try:
        is_container_node = studio_graph._is_container_node
        is_operator_node = studio_graph._is_operator_node
    except Exception as exc:
        raise TypeError("studio_graph must be an F8StudioGraph (missing type predicates).") from exc

    services = [n for n in all_nodes if is_container_node(n)]
    operators = [n for n in all_nodes if is_operator_node(n)]

    # Standalone single-node services (non-container F8ServiceSpec nodes).
    service_nodes: list[Any] = []
    for n in all_nodes:
        try:
            spec = n.spec
        except Exception:
            continue
        if not isinstance(spec, F8ServiceSpec):
            continue
        if is_container_node(n):
            continue
        service_nodes.append(n)

    global_graph = compile_global_runtime_graph(
        services=services,
        operators=operators,
        service_nodes=service_nodes,
    )
    return CompiledRuntimeGraphs(global_graph=global_graph, per_service=split_runtime_graph_by_service(global_graph))
