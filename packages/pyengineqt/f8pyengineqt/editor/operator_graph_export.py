from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from ..graph.operator_graph import OperatorGraph
from ..graph.operator_instance import OperatorInstance
from ..renderers.generic import GenericNode
from ..services.service_operator_registry import ServiceOperatorSpecRegistry
from f8pysdk import (
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8OperatorSpec,
    F8PrimitiveTypeEnum,
    operator_key,
)
from ..engine.nats_naming import ensure_token


def export_operator_graph(
    node_graph: Any,
    *,
    service_id: str,
    node_filter: Callable[[GenericNode], bool] | None = None,
    edge_meta: Callable[..., dict[str, Any]] | None = None,
) -> OperatorGraph:
    """
    Export a NodeGraphQt graph (or sub-graph) into an OperatorGraph.
    """
    graph = OperatorGraph(service_id=ensure_token(service_id, label="service_id"))

    operator_nodes = [
        n for n in node_graph.all_nodes() if isinstance(n, GenericNode) and (node_filter(n) if node_filter else True)
    ]
    operator_ids = {n.id for n in operator_nodes}

    for node in operator_nodes:
        state: dict[str, Any] = {}
        spec = node.spec
        if not isinstance(spec, F8OperatorSpec):
            try:
                spec = ServiceOperatorSpecRegistry.instance().get(operator_key(spec.serviceClass, spec.operatorClass))
            except Exception:
                continue

        for field in spec.states or []:
            try:
                value = node.get_property(field.name)
                schema_type = getattr(field.valueSchema, "type", None) if field.valueSchema else None
                if schema_type == F8PrimitiveTypeEnum.integer:
                    state[field.name] = int(value) if value is not None and value != "" else None
                elif schema_type == F8PrimitiveTypeEnum.number:
                    state[field.name] = float(value) if value is not None and value != "" else None
                elif schema_type == F8PrimitiveTypeEnum.boolean:
                    state[field.name] = bool(value)
                else:
                    state[field.name] = value
            except Exception:
                pass

        instance = OperatorInstance.from_spec(spec, id=node.id, state=state)

        try:
            pos = node.pos()
            renderer_props = instance.spec.rendererProps or {}
            renderer_props["pos"] = [float(pos[0]), float(pos[1])]
            instance.spec.rendererProps = renderer_props
        except Exception:
            pass

        graph.add_node(instance)

    def raw_name_for_port(target_node: GenericNode, kind: str, direction: str, port_obj: Any) -> str | None:
        mapping = {
            ("exec", "in"): target_node.port_handles.exec_in,
            ("exec", "out"): target_node.port_handles.exec_out,
            ("data", "in"): target_node.port_handles.data_in,
            ("data", "out"): target_node.port_handles.data_out,
            ("state", "in"): target_node.port_handles.state_in,
            ("state", "out"): target_node.port_handles.state_out,
        }.get((kind, direction))
        if not mapping:
            return None
        for raw_name, handle in mapping.items():
            if handle is port_obj:
                return raw_name
        return None

    def _call_edge_meta(source: GenericNode, target: GenericNode, kind: str, local_side: str) -> dict[str, Any]:
        if edge_meta is None:
            return {}
        try:
            # New signature: (src, dst, kind, local_side)
            return edge_meta(source, target, kind, local_side) or {}
        except TypeError:
            # Backward compatibility: (src, dst, kind)
            try:
                return edge_meta(source, target, kind) or {}
            except Exception:
                return {}
        except Exception:
            return {}

    def _data_meta_for(
        source: GenericNode, target: GenericNode, *, local_side: str
    ) -> tuple[F8EdgeStrategyEnum, int | None, int | None, dict[str, Any]]:
        meta = _call_edge_meta(source, target, "data", local_side)
        strategy = F8EdgeStrategyEnum.latest
        queue_size: int | None = None
        timeout_ms: int | None = None
        if isinstance(meta.get("strategy"), F8EdgeStrategyEnum):
            strategy = meta["strategy"]
        try:
            if meta.get("queueSize") is not None:
                queue_size = int(meta["queueSize"])
        except Exception:
            queue_size = None
        try:
            if meta.get("timeoutMs") is not None:
                timeout_ms = int(meta["timeoutMs"])
        except Exception:
            timeout_ms = None
        return strategy, queue_size, timeout_ms, meta

    def _service_ids(meta: dict[str, Any]) -> tuple[str, str] | None:
        from_sid = str(meta.get("fromServiceId") or graph.service_id).strip()
        to_sid = str(meta.get("toServiceId") or graph.service_id).strip()
        if not from_sid or not to_sid:
            return None
        try:
            return ensure_token(from_sid, label="fromServiceId"), ensure_token(to_sid, label="toServiceId")
        except Exception:
            return None

    def _edge_id(kind: str, from_id: str, from_port: str, to_id: str, to_port: str) -> str:
        key = f"{kind}:{from_id}:{from_port}->{to_id}:{to_port}"
        return uuid.uuid5(uuid.NAMESPACE_OID, key).hex

    for source in operator_nodes:
        for raw_out, out_port in source.port_handles.exec_out.items():
            for in_port in out_port.connected_ports():
                target = in_port.node()
                if not isinstance(target, GenericNode):
                    continue
                if target.id not in operator_ids:
                    # Cross-instance exec edges are not supported (skip).
                    continue
                raw_in = raw_name_for_port(target, "exec", "in", in_port)
                if raw_in is None:
                    continue
                try:
                    edge = F8Edge(
                        edgeId=_edge_id("exec", str(source.id), str(raw_out), str(target.id), str(raw_in)),
                        fromServiceId=graph.service_id,
                        fromOperatorId=str(source.id),
                        fromPort=str(raw_out),
                        toServiceId=graph.service_id,
                        toOperatorId=str(target.id),
                        toPort=str(raw_in),
                        kind=F8EdgeKindEnum.exec,
                        strategy=F8EdgeStrategyEnum.latest,
                    )
                    graph._connect_from_spec(edge)  # type: ignore[attr-defined]
                except Exception:
                    pass

        for raw_out, out_port in source.port_handles.data_out.items():
            for in_port in out_port.connected_ports():
                target = in_port.node()
                if not isinstance(target, GenericNode):
                    continue
                raw_in = raw_name_for_port(target, "data", "in", in_port)
                if raw_in is None:
                    continue
                try:
                    strategy, queue_size, timeout_ms, meta = _data_meta_for(source, target, local_side="from")
                    sids = _service_ids(meta)
                    if sids is None:
                        # If service ids are unknown, only export local-only edges.
                        if target.id not in operator_ids:
                            continue
                        sids = (graph.service_id, graph.service_id)
                    from_sid, to_sid = sids

                    edge = F8Edge(
                        edgeId=_edge_id("data", str(source.id), str(raw_out), str(target.id), str(raw_in)),
                        fromServiceId=from_sid,
                        fromOperatorId=str(source.id),
                        fromPort=str(raw_out),
                        toServiceId=to_sid,
                        toOperatorId=str(target.id),
                        toPort=str(raw_in),
                        kind=F8EdgeKindEnum.data,
                        strategy=strategy,
                        queueSize=queue_size,
                        timeoutMs=timeout_ms,
                    )
                    graph._connect_from_spec(edge)  # type: ignore[attr-defined]
                except Exception:
                    pass

        for raw_out, out_port in source.port_handles.state_out.items():
            for in_port in out_port.connected_ports():
                target = in_port.node()
                if not isinstance(target, GenericNode):
                    continue
                raw_in = raw_name_for_port(target, "state", "in", in_port)
                if raw_in is None:
                    continue
                try:
                    meta = _call_edge_meta(source, target, "state", "from")
                    sids = _service_ids(meta)
                    if sids is None:
                        if target.id not in operator_ids:
                            continue
                        sids = (graph.service_id, graph.service_id)
                    from_sid, to_sid = sids
                    edge = F8Edge(
                        edgeId=_edge_id("state", str(source.id), str(raw_out), str(target.id), str(raw_in)),
                        fromServiceId=from_sid,
                        fromOperatorId=str(source.id),
                        fromPort=str(raw_out),
                        toServiceId=to_sid,
                        toOperatorId=str(target.id),
                        toPort=str(raw_in),
                        kind=F8EdgeKindEnum.state,
                        strategy=F8EdgeStrategyEnum.latest,
                    )
                    graph._connect_from_spec(edge)  # type: ignore[attr-defined]
                except Exception:
                    pass

    # Incoming edges where the local node is the target side.
    for target in operator_nodes:
        for raw_in, in_port in target.port_handles.data_in.items():
            for out_port in in_port.connected_ports():
                source = out_port.node()
                if not isinstance(source, GenericNode):
                    continue
                if source.id in operator_ids:
                    continue
                raw_out = raw_name_for_port(source, "data", "out", out_port)
                if raw_out is None:
                    continue
                try:
                    strategy, queue_size, timeout_ms, meta = _data_meta_for(source, target, local_side="to")
                    sids = _service_ids(meta)
                    if sids is None:
                        # If remote service id is unknown, skip exporting this edge.
                        continue
                    from_sid, to_sid = sids
                    edge = F8Edge(
                        edgeId=_edge_id("data", str(source.id), str(raw_out), str(target.id), str(raw_in)),
                        fromServiceId=from_sid,
                        fromOperatorId=str(source.id),
                        fromPort=str(raw_out),
                        toServiceId=to_sid,
                        toOperatorId=str(target.id),
                        toPort=str(raw_in),
                        kind=F8EdgeKindEnum.data,
                        strategy=strategy,
                        queueSize=queue_size,
                        timeoutMs=timeout_ms,
                    )
                    graph._connect_from_spec(edge)  # type: ignore[attr-defined]
                except Exception:
                    pass

        for raw_in, in_port in target.port_handles.state_in.items():
            for out_port in in_port.connected_ports():
                source = out_port.node()
                if not isinstance(source, GenericNode):
                    continue
                if source.id in operator_ids:
                    continue
                raw_out = raw_name_for_port(source, "state", "out", out_port)
                if raw_out is None:
                    continue
                try:
                    meta = _call_edge_meta(source, target, "state", "to")
                    sids = _service_ids(meta)
                    if sids is None:
                        continue
                    from_sid, to_sid = sids
                    edge = F8Edge(
                        edgeId=_edge_id("state", str(source.id), str(raw_out), str(target.id), str(raw_in)),
                        fromServiceId=from_sid,
                        fromOperatorId=str(source.id),
                        fromPort=str(raw_out),
                        toServiceId=to_sid,
                        toOperatorId=str(target.id),
                        toPort=str(raw_in),
                        kind=F8EdgeKindEnum.state,
                        strategy=F8EdgeStrategyEnum.latest,
                    )
                    graph._connect_from_spec(edge)  # type: ignore[attr-defined]
                except Exception:
                    pass

    return graph
