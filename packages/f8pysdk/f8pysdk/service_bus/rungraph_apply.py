from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

from ..generated import F8Edge, F8EdgeKindEnum, F8RuntimeGraph, F8StateAccess
from ..nats_naming import data_subject, ensure_token, kv_key_node_state
from .state_write import StateWriteOrigin
from ..time_utils import now_ms

from .routing_data import precreate_input_buffers_for_cross_in, sync_subscriptions

if TYPE_CHECKING:
    from .bus import ServiceBus


async def set_rungraph(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Publish a full rungraph snapshot for this service.
    """
    payload = graph.model_dump(mode="json", by_alias=True)
    meta = dict(payload.get("meta") or {})
    meta["ts"] = int(now_ms())
    payload["meta"] = meta
    raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    await bus._transport.kv_put(bus._rungraph_key, raw)
    # Endpoint-only mode: apply immediately (no KV watch).
    await apply_rungraph_bytes(bus, raw)


async def apply_rungraph_bytes(bus: "ServiceBus", raw: bytes) -> None:
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return
    try:
        graph = F8RuntimeGraph.model_validate(payload)
    except Exception:
        return
    try:
        await validate_rungraph_or_raise(bus, graph)
    except Exception:
        return
    try:
        for n in list(getattr(graph, "nodes", None) or []):
            # Service/container nodes use `nodeId == serviceId`.
            if getattr(n, "operatorClass", None) is None and str(getattr(n, "nodeId", "")) != str(getattr(n, "serviceId", "")):
                raise ValueError("invalid rungraph: service node requires nodeId == serviceId")
    except Exception:
        return

    bus._graph = graph
    # Reset cross-state ordering on graph changes.
    bus._cross_state_last_ts.clear()
    # Cache local node state access for enforcement and filtering.
    bus._state_access_by_node_field.clear()
    try:
        for n in list(getattr(graph, "nodes", None) or []):
            if str(getattr(n, "serviceId", "") or "") != bus.service_id:
                continue
            node_id = str(getattr(n, "nodeId", "") or "")
            if not node_id:
                continue
            for sf in list(getattr(n, "stateFields", None) or []):
                name = str(getattr(sf, "name", "") or "").strip()
                if not name:
                    continue
                access = getattr(sf, "access", None)
                if isinstance(access, F8StateAccess):
                    bus._state_access_by_node_field[(node_id, name)] = access
    except Exception:
        bus._state_access_by_node_field.clear()
    if bus._debug_state:
        try:
            node_count = len(list(graph.nodes or []))
            edge_count = len(list(graph.edges or []))
            graph_id = str(getattr(graph, "graphId", "") or "")
        except Exception:
            node_count = 0
            edge_count = 0
            graph_id = ""
        print("state_debug[%s] rungraph_applied graph=%s nodes=%s edges=%s" % (bus.service_id, graph_id, str(node_count), str(edge_count)))
    await rebuild_routes(bus)
    await apply_rungraph_state_values(bus, graph)
    await seed_builtin_identity_state(bus, graph)
    await initial_sync_intra_state_edges(bus, graph)

    for hook in list(bus._rungraph_hooks):
        try:
            r = hook.on_rungraph(graph)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            continue
    # Cross-state watches and initial sync are intentionally installed AFTER
    # rungraph hooks so local runtime nodes are registered first. This avoids
    # dropping initial remote values due to missing nodes/fields.
    await bus._sync_cross_state_watches()


async def apply_rungraph_state_values(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Materialize per-node `stateValues` into KV (and dispatch locally).
    """
    for n in list(getattr(graph, "nodes", None) or []):
        if str(getattr(n, "serviceId", "")) != bus.service_id:
            continue
        node_id = str(getattr(n, "nodeId", "") or "").strip()
        if not node_id:
            continue
        values = getattr(n, "stateValues", None) or {}
        if not isinstance(values, dict) or not values:
            continue
        for k, v in list(values.items()):
            field = str(k or "").strip()
            if not field:
                continue
            access = bus._state_access_by_node_field.get((node_id, field))
            if access not in (F8StateAccess.rw, F8StateAccess.wo):
                continue
            if (node_id, field) in bus._cross_state_targets:
                continue
            try:
                await bus._publish_state(
                    node_id,
                    field,
                    v,
                    origin=StateWriteOrigin.rungraph,
                    source="rungraph",
                    meta={"via": "rungraph", "_noStateFanout": True},
                )
            except Exception:
                continue


async def initial_sync_intra_state_edges(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Best-effort initial sync for intra-service state edges.
    """
    for edge in list(getattr(graph, "edges", None) or []):
        if getattr(edge, "kind", None) != F8EdgeKindEnum.state:
            continue
        if str(getattr(edge, "fromServiceId", "")) != bus.service_id or str(getattr(edge, "toServiceId", "")) != bus.service_id:
            continue
        if not edge.fromOperatorId or not edge.toOperatorId:
            continue
        from_node = str(edge.fromOperatorId)
        from_field = str(edge.fromPort)
        to_node = str(edge.toOperatorId)
        to_field = str(edge.toPort)
        access = bus._state_access_by_node_field.get((to_node, to_field))
        if access not in (F8StateAccess.rw, F8StateAccess.wo):
            continue
        try:
            found_from, v_from, _ts_from = await bus.get_state_with_ts(from_node, from_field)
        except Exception:
            continue
        if not found_from:
            continue
        try:
            found_to, v_to, _ts_to = await bus.get_state_with_ts(to_node, to_field)
        except Exception:
            found_to, v_to = False, None
        if found_to and v_to == v_from:
            continue
        try:
            await bus._publish_state(
                to_node,
                to_field,
                v_from,
                origin=StateWriteOrigin.external,
                source="state_edge_init",
                meta={"fromNodeId": from_node, "fromField": from_field, "_fanoutHops": 1},
            )
        except Exception:
            continue


async def seed_builtin_identity_state(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Seed readonly identity fields (`svcId`, `operatorId`) into KV for local nodes.
    """
    ts = int(now_ms())
    for n in list(getattr(graph, "nodes", None) or []):
        if str(getattr(n, "serviceId", "")) != bus.service_id:
            continue
        node_id = str(getattr(n, "nodeId", "") or "").strip()
        if not node_id:
            continue
        try:
            if bus._state_access_by_node_field.get((node_id, "svcId")) is not None:
                await bus._put_state_kv_unvalidated(
                    node_id=node_id,
                    field="svcId",
                    value=str(getattr(n, "serviceId", "") or bus.service_id),
                    ts_ms=ts,
                    meta={"source": "system", "origin": "system", "builtin": True},
                )
            if getattr(n, "operatorClass", None) is not None and bus._state_access_by_node_field.get((node_id, "operatorId")) is not None:
                await bus._put_state_kv_unvalidated(
                    node_id=node_id,
                    field="operatorId",
                    value=str(getattr(n, "nodeId", "") or node_id),
                    ts_ms=ts,
                    meta={"source": "system", "origin": "system", "builtin": True},
                )
        except Exception:
            continue


async def validate_rungraph_or_raise(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Validate the rungraph before applying it.
    """
    access_map: dict[tuple[str, str], F8StateAccess] = {}
    for n in list(getattr(graph, "nodes", None) or []):
        if str(getattr(n, "serviceId", "")) != bus.service_id:
            continue
        node_id = str(getattr(n, "nodeId", "") or "")
        if not node_id:
            continue
        for sf in list(getattr(n, "stateFields", None) or []):
            name = str(getattr(sf, "name", "") or "").strip()
            if not name:
                continue
            a = getattr(sf, "access", None)
            if isinstance(a, F8StateAccess):
                access_map[(node_id, name)] = a

    for n in list(getattr(graph, "nodes", None) or []):
        if str(getattr(n, "serviceId", "")) != bus.service_id:
            continue
        node_id = str(getattr(n, "nodeId", "") or "")
        if not node_id:
            raise ValueError("missing nodeId")
        access_by_name: dict[str, F8StateAccess] = {}
        for sf in list(getattr(n, "stateFields", None) or []):
            name = str(getattr(sf, "name", "") or "").strip()
            if not name:
                continue
            a = getattr(sf, "access", None)
            if isinstance(a, F8StateAccess):
                access_by_name[name] = a

        values = getattr(n, "stateValues", None) or {}
        if isinstance(values, dict):
            for k in list(values.keys()):
                key = str(k)
                a = access_by_name.get(key)
                if a is None:
                    raise ValueError(f"unknown state value: {node_id}.{key}")
                if a == F8StateAccess.ro:
                    raise ValueError(f"read-only state cannot be set by rungraph: {node_id}.{key}")

    for e in list(getattr(graph, "edges", None) or []):
        if getattr(e, "kind", None) != F8EdgeKindEnum.state:
            continue
        if str(getattr(e, "toServiceId", "")) != bus.service_id:
            continue
        to_node = str(getattr(e, "toOperatorId", "") or "")
        to_field = str(getattr(e, "toPort", "") or "")
        if not to_node or not to_field:
            continue
        a = access_map.get((to_node, to_field))
        if a == F8StateAccess.ro:
            raise ValueError(f"state edge targets non-writable field: {to_node}.{to_field} ({a.value})")

    for hook in list(bus._rungraph_hooks):
        try:
            r = hook.validate_rungraph(graph)
            if asyncio.iscoroutine(r):
                await r
        except Exception as exc:
            raise ValueError(str(exc)) from exc


async def rebuild_routes(bus: "ServiceBus") -> None:
    graph = bus._graph
    if graph is None:
        return

    bus._data_inputs.clear()
    bus._intra_state_out.clear()

    # Intra (in-process) routing: local service -> local service.
    intra: dict[tuple[str, str], list[tuple[str, str]]] = {}
    intra_in: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
    for edge in graph.edges:
        if edge.kind != F8EdgeKindEnum.data:
            continue
        if str(edge.fromServiceId) != bus.service_id or str(edge.toServiceId) != bus.service_id:
            continue
        if not edge.fromOperatorId or not edge.toOperatorId:
            continue
        intra.setdefault((str(edge.fromOperatorId), str(edge.fromPort)), []).append((str(edge.toOperatorId), str(edge.toPort)))
        intra_in.setdefault((str(edge.toOperatorId), str(edge.toPort)), []).append((str(edge.fromOperatorId), str(edge.fromPort), edge))
    bus._intra_data_out = intra
    bus._intra_data_in = intra_in

    # Intra-service state fanout: local state edges.
    intra_state_out: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
    for edge in graph.edges:
        if edge.kind != F8EdgeKindEnum.state:
            continue
        if str(edge.fromServiceId) != bus.service_id or str(edge.toServiceId) != bus.service_id:
            continue
        if not edge.fromOperatorId or not edge.toOperatorId:
            continue
        intra_state_out.setdefault((str(edge.fromOperatorId), str(edge.fromPort)), []).append((str(edge.toOperatorId), str(edge.toPort), edge))
    bus._intra_state_out = intra_state_out

    # Cross routing.
    cross_in: dict[str, list[tuple[str, str, F8Edge]]] = {}
    cross_out: dict[tuple[str, str], str] = {}
    for edge in graph.edges:
        if edge.kind != F8EdgeKindEnum.data:
            continue
        if str(edge.fromServiceId) == str(edge.toServiceId):
            continue
        if not edge.fromOperatorId:
            continue

        subject = data_subject(str(edge.fromServiceId), from_node_id=str(edge.fromOperatorId), port_id=str(edge.fromPort))

        if str(edge.toServiceId) == bus.service_id:
            if not edge.toOperatorId:
                continue
            to_node = str(edge.toOperatorId)
            cross_in.setdefault(subject, []).append((to_node, str(edge.toPort), edge))
            continue

        if str(edge.fromServiceId) == bus.service_id:
            from_node = str(edge.fromOperatorId)
            cross_out[(from_node, str(edge.fromPort))] = subject

    bus._cross_in_by_subject = cross_in
    bus._cross_out_subjects = cross_out

    precreate_input_buffers_for_cross_in(bus, cross_in)

    await sync_subscriptions(bus, set(cross_in.keys()))
    bus._update_cross_state_bindings(graph)
    await bus._stop_unused_cross_state_watches()
