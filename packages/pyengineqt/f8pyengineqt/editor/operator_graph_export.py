from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from ..engine.nats_naming import data_subject, ensure_token
from ..graph.operator_graph import OperatorGraph
from ..graph.operator_instance import OperatorInstance
from ..operators.operator_registry import OperatorSpecRegistry
from ..renderers.generic import GenericNode
from f8pysdk import (
    F8EdgeKindEmum,
    F8EdgeScopeEnum,
    F8EdgeSpec,
    F8EdgeStrategyEnum,
    F8OperatorSpec,
    F8PrimitiveTypeEnum,
    operator_key,
)


def export_operator_graph(
    node_graph: Any,
    *,
    node_filter: Callable[[GenericNode], bool] | None = None,
    edge_meta: Callable[..., dict[str, Any]] | None = None,
) -> OperatorGraph:
    """
    Export a NodeGraphQt graph (or sub-graph) into an OperatorGraph.
    """
    graph = OperatorGraph()

    operator_nodes = [
        n for n in node_graph.all_nodes() if isinstance(n, GenericNode) and (node_filter(n) if node_filter else True)
    ]
    operator_ids = {n.id for n in operator_nodes}

    for node in operator_nodes:
        state: dict[str, Any] = {}
        spec = node.spec
        if not isinstance(spec, F8OperatorSpec):
            try:
                spec = OperatorSpecRegistry.instance().get(operator_key(spec.serviceClass, spec.operatorClass))
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

    def _scope_for(source: GenericNode, target: GenericNode, kind: str, *, local_side: str) -> F8EdgeScopeEnum:
        meta = _call_edge_meta(source, target, kind, local_side)
        scope = meta.get("scope")
        return scope if isinstance(scope, F8EdgeScopeEnum) else F8EdgeScopeEnum.intra

    def _data_meta_for(
        source: GenericNode, target: GenericNode, *, local_side: str
    ) -> tuple[F8EdgeScopeEnum, F8EdgeStrategyEnum, int | None, int | None]:
        meta = _call_edge_meta(source, target, "data", local_side)
        scope = meta.get("scope") if isinstance(meta.get("scope"), F8EdgeScopeEnum) else F8EdgeScopeEnum.intra
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
        return scope, strategy, queue_size, timeout_ms

    def _edge_id(kind: str, from_id: str, from_port: str, to_id: str, to_port: str) -> str:
        key = f"{kind}:{from_id}:{from_port}->{to_id}:{to_port}"
        return uuid.uuid5(uuid.NAMESPACE_OID, key).hex

    seen_cross: set[tuple[str, str, str, str, str]] = set()
    seen_cross_data_out: set[tuple[str, str]] = set()

    def _append_cross(edge: F8EdgeSpec) -> None:
        k = str(edge.kind)
        key = (
            k,
            str(edge.from_),
            str(edge.fromPort),
            str(edge.to),
            str(edge.toPort),
        )
        if key in seen_cross:
            return
        seen_cross.add(key)
        if edge.kind == F8EdgeKindEmum.data:
            graph.data_edges.append(edge)
        elif edge.kind == F8EdgeKindEmum.state:
            graph.state_edges.append(edge)
        elif edge.kind == F8EdgeKindEmum.exec:
            graph.exec_edges.append(edge)

    def _from_service_id(meta: dict[str, Any]) -> str:
        sid = str(meta.get("fromServiceId") or "").strip()
        if not sid:
            return ""
        try:
            return ensure_token(sid, label="from_service_id")
        except Exception:
            return ""

    def _cross_data_subject(meta: dict[str, Any], *, from_node_id: str, out_port: str) -> str:
        sid = _from_service_id(meta)
        if not sid:
            return ""
        try:
            return data_subject(sid, from_node_id=from_node_id, port_id=out_port)
        except Exception:
            return ""

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
                    graph.connect_exec(
                        source.id,
                        raw_out,
                        target.id,
                        raw_in,
                        scope=_scope_for(source, target, "exec", local_side="from"),
                    )
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
                    scope, strategy, queue_size, timeout_ms = _data_meta_for(source, target, local_side="from")
                    if target.id in operator_ids:
                        graph.connect_data(
                            source.id,
                            raw_out,
                            target.id,
                            raw_in,
                            scope=scope,
                            strategy=strategy,
                            queue_size=queue_size,
                            timeout_ms=timeout_ms,
                        )
                    else:
                        # Cross-instance half-edge (outgoing).
                        key = (str(source.id), str(raw_out))
                        if key in seen_cross_data_out:
                            continue
                        seen_cross_data_out.add(key)

                        meta = _call_edge_meta(source, target, "data", "from")
                        subj = _cross_data_subject(meta, from_node_id=str(source.id), out_port=str(raw_out))
                        eid = uuid.uuid5(uuid.NAMESPACE_OID, f"data_out:{source.id}:{raw_out}").hex
                        _append_cross(
                            F8EdgeSpec(
                                from_=str(source.id),
                                fromPort=str(raw_out),
                                to=str(target.id),
                                toPort=str(raw_in),
                                kind=F8EdgeKindEmum.data,
                                scope=F8EdgeScopeEnum.cross,
                                strategy=strategy,
                                queueSize=queue_size,
                                timeoutMs=timeout_ms,
                                edgeId=eid,
                                direction="out",
                                subject=subj,
                                peerServiceId=str(meta.get("toServiceId") or meta.get("peerServiceId") or ""),
                            )
                        )
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
                    scope = _scope_for(source, target, "state", local_side="from")
                    if target.id in operator_ids:
                        graph.connect_state(source.id, raw_out, target.id, raw_in, scope=scope)
                    else:
                        meta = _call_edge_meta(source, target, "state", "from")
                        peer_service_id = str(meta.get("peerServiceId") or "")
                        eid = _edge_id("state", str(source.id), str(raw_out), str(target.id), str(raw_in))
                        _append_cross(
                            F8EdgeSpec(
                                from_=str(source.id),
                                fromPort=str(raw_out),
                                to=str(target.id),
                                toPort=str(raw_in),
                                kind=F8EdgeKindEmum.state,
                                scope=F8EdgeScopeEnum.cross,
                                strategy=F8EdgeStrategyEnum.hold,
                                edgeId=eid,
                                direction="out",
                                peerServiceId=peer_service_id,
                            )
                        )
                except Exception:
                    pass

    # Incoming cross-instance half-edges (local node is the target side).
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
                    scope, strategy, queue_size, timeout_ms = _data_meta_for(source, target, local_side="to")
                    meta = _call_edge_meta(source, target, "data", "to")
                    from_sid = str(meta.get("fromServiceId") or meta.get("peerServiceId") or "")
                    subj = _cross_data_subject(meta, from_node_id=str(source.id), out_port=str(raw_out))
                    eid = _edge_id("data", str(source.id), str(raw_out), str(target.id), str(raw_in))
                    _append_cross(
                        F8EdgeSpec(
                            from_=str(source.id),
                            fromPort=str(raw_out),
                            to=str(target.id),
                            toPort=str(raw_in),
                            kind=F8EdgeKindEmum.data,
                            scope=F8EdgeScopeEnum.cross,
                            strategy=strategy,
                            queueSize=queue_size,
                            timeoutMs=timeout_ms,
                            edgeId=eid,
                            direction="in",
                            subject=subj,
                            peerServiceId=from_sid,
                        )
                    )
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
                    peer_service_id = str(meta.get("peerServiceId") or "")
                    eid = _edge_id("state", str(source.id), str(raw_out), str(target.id), str(raw_in))
                    _append_cross(
                        F8EdgeSpec(
                            from_=str(source.id),
                            fromPort=str(raw_out),
                            to=str(target.id),
                            toPort=str(raw_in),
                            kind=F8EdgeKindEmum.state,
                            scope=F8EdgeScopeEnum.cross,
                            strategy=F8EdgeStrategyEnum.hold,
                            edgeId=eid,
                            direction="in",
                            peerServiceId=peer_service_id,
                        )
                    )
                except Exception:
                    pass

    return graph
