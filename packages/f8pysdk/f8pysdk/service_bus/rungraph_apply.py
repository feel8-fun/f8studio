from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

from ..generated import F8Edge, F8EdgeKindEnum, F8RuntimeGraph, F8StateAccess
from ..nats_naming import data_subject, ensure_token, kv_key_node_state
from .state_write import StateWriteOrigin, StateWriteSource
from ..time_utils import now_ms
from ..rungraph_validation import validate_state_edges_or_raise

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
        for n in list(graph.nodes or []):
            # Service/container nodes use `nodeId == serviceId`.
            if n.operatorClass is None and str(n.nodeId) != str(n.serviceId):
                raise ValueError("invalid rungraph: service node requires nodeId == serviceId")
    except Exception:
        return

    bus._graph = graph
    # Reset cross-state ordering on graph changes.
    bus._cross_state_last_ts.clear()
    # Cache local node state access for enforcement and filtering.
    bus._state_access_by_node_field.clear()
    try:
        for n in list(graph.nodes or []):
            if str(n.serviceId or "") != bus.service_id:
                continue
            node_id = str(n.nodeId or "")
            if not node_id:
                continue
            for sf in list(n.stateFields or []):
                name = str(sf.name or "").strip()
                if not name:
                    continue
                access = sf.access
                if isinstance(access, F8StateAccess):
                    bus._state_access_by_node_field[(node_id, name)] = access
    except Exception:
        bus._state_access_by_node_field.clear()
    if bus._debug_state:
        try:
            node_count = len(list(graph.nodes or []))
            edge_count = len(list(graph.edges or []))
            graph_id = str(graph.graphId or "")
        except Exception:
            node_count = 0
            edge_count = 0
            graph_id = ""
        print("state_debug[%s] rungraph_applied graph=%s nodes=%s edges=%s" % (bus.service_id, graph_id, str(node_count), str(edge_count)))
    await rebuild_routes(bus)
    await apply_rungraph_state_values(bus, graph)
    await seed_builtin_identity_state(bus, graph)

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
    # With strong cross-state sync, materialize remote values first (no fanout),
    # then run a single ordered intra-service init propagation.
    await initial_sync_intra_state_edges(bus, graph)


async def apply_rungraph_state_values(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Materialize per-node `stateValues` into KV (and dispatch locally).
    """
    def _unwrap_json_value(v: object) -> object:
        if v is None:
            return None
        # F8JsonValue is a pydantic RootModel wrapper around JSON primitives/containers.
        try:
            from ..generated import F8JsonValue  # local import to keep module import light

            if isinstance(v, F8JsonValue):
                return v.root
        except Exception:
            pass
        try:
            return v.root  # type: ignore[attr-defined]
        except Exception:
            return v

    for n in list(graph.nodes or []):
        if str(n.serviceId) != bus.service_id:
            continue
        node_id = str(n.nodeId or "").strip()
        if not node_id:
            continue
        values = n.stateValues or {}
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
                    _unwrap_json_value(v),
                    origin=StateWriteOrigin.rungraph,
                    source=StateWriteSource.rungraph,
                    meta={"via": "rungraph", "_noStateFanout": True},
                )
            except Exception:
                continue


async def initial_sync_intra_state_edges(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Best-effort initial sync for intra-service state edges.
    """
    # Motivation:
    # - Avoid propagating "soon-to-be-overwritten" intermediate values.
    # - Avoid order-dependence from scanning `graph.edges` linearly.
    #
    # Since we disallow multiple upstreams per downstream state field, we can
    # identify roots (in-degree == 0) and only start propagation from roots
    # whose state already exists.
    edges = list(graph.edges or [])
    out: dict[tuple[str, str], list[tuple[str, str]]] = {}
    inbound: set[tuple[str, str]] = set()
    nodes: set[tuple[str, str]] = set()
    upstream_by_target: dict[tuple[str, str], tuple[str, str]] = {}

    for edge in edges:
        if edge.kind != F8EdgeKindEnum.state:
            continue
        if str(edge.fromServiceId) != bus.service_id or str(edge.toServiceId) != bus.service_id:
            continue
        if not edge.fromOperatorId or not edge.toOperatorId:
            continue
        from_key = (str(edge.fromOperatorId), str(edge.fromPort))
        to_key = (str(edge.toOperatorId), str(edge.toPort))

        # Skip unknown/unwritable targets.
        access = bus._state_access_by_node_field.get(to_key)
        if access not in (F8StateAccess.rw, F8StateAccess.wo):
            continue

        # Enforce single-upstream per target (should already be validated elsewhere).
        prev = upstream_by_target.get(to_key)
        if prev is not None and prev != from_key:
            if bus._debug_state:
                try:
                    print(
                        "state_debug[%s] state_edge_init_skip_multi_upstream to=%s.%s from_a=%s.%s from_b=%s.%s"
                        % (
                            bus.service_id,
                            str(to_key[0]),
                            str(to_key[1]),
                            str(prev[0]),
                            str(prev[1]),
                            str(from_key[0]),
                            str(from_key[1]),
                        )
                    )
                except Exception:
                    pass
            continue
        upstream_by_target[to_key] = from_key

        out.setdefault(from_key, []).append(to_key)
        inbound.add(to_key)
        nodes.add(from_key)
        nodes.add(to_key)

    if not out:
        return

    roots = [k for k in nodes if k not in inbound]
    if not roots:
        if bus._debug_state:
            try:
                print("state_debug[%s] state_edge_init_no_roots (cycle?)" % (bus.service_id,))
            except Exception:
                pass
        return

    ts0 = int(now_ms())

    # Propagate only from roots that have an actual value in KV/cache.
    visited: set[tuple[str, str]] = set()
    for root in list(roots):
        try:
            root_state = await bus.get_state(root[0], root[1])
        except Exception:
            continue
        if not root_state.found:
            continue

        queue: list[tuple[tuple[str, str], object]] = [(root, root_state.value)]
        seen_in_component: set[tuple[str, str]] = set()
        while queue:
            from_key, from_val = queue.pop(0)
            if from_key in seen_in_component:
                continue
            seen_in_component.add(from_key)
            visited.add(from_key)

            for to_key in list(out.get(from_key) or []):
                if to_key in seen_in_component:
                    # Cycle: don't spin.
                    continue

                try:
                    to_state = await bus.get_state(to_key[0], to_key[1])
                except Exception:
                    to_state = None

                if to_state is not None and to_state.found:
                    try:
                        if to_state.value == from_val:
                            queue.append((to_key, to_state.value))
                            continue
                    except Exception:
                        pass

                try:
                    await bus._publish_state(
                        to_key[0],
                        to_key[1],
                        from_val,
                        ts_ms=ts0,
                        origin=StateWriteOrigin.external,
                        source=StateWriteSource.state_edge_intra_init,
                        meta={"fromNodeId": from_key[0], "fromField": from_key[1]},
                    )
                except Exception:
                    continue

                # Continue propagation using the post-validation cached value if available.
                try:
                    cached = bus._state_cache.get(to_key)
                    next_val = cached[0] if cached is not None else from_val
                except Exception:
                    next_val = from_val
                queue.append((to_key, next_val))


async def seed_builtin_identity_state(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Seed readonly identity fields (`svcId`, `operatorId`) into KV for local nodes.
    """
    ts = int(now_ms())
    for n in list(graph.nodes or []):
        if str(n.serviceId) != bus.service_id:
            continue
        node_id = str(n.nodeId or "").strip()
        if not node_id:
            continue
        try:
            if bus._state_access_by_node_field.get((node_id, "svcId")) is not None:
                await bus._publish_state(
                    node_id,
                    "svcId",
                    str(n.serviceId or bus.service_id),
                    origin=StateWriteOrigin.system,
                    source=StateWriteSource.system,
                    ts_ms=ts,
                    meta={"builtin": True, "_noStateFanout": True},
                    deliver_local=False,
                )
            if n.operatorClass is not None and bus._state_access_by_node_field.get((node_id, "operatorId")) is not None:
                await bus._publish_state(
                    node_id,
                    "operatorId",
                    str(n.nodeId or node_id),
                    origin=StateWriteOrigin.system,
                    source=StateWriteSource.system,
                    ts_ms=ts,
                    meta={"builtin": True, "_noStateFanout": True},
                    deliver_local=False,
                )
        except Exception:
            continue


async def validate_rungraph_or_raise(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Validate the rungraph before applying it.
    """
    # Global state-edge constraints (covers cross-service cycles too).
    validate_state_edges_or_raise(graph, forbid_cycles=True, forbid_multi_upstream=True)

    access_map: dict[tuple[str, str], F8StateAccess] = {}
    for n in list(graph.nodes or []):
        if str(n.serviceId) != bus.service_id:
            continue
        node_id = str(n.nodeId or "")
        if not node_id:
            continue
        for sf in list(n.stateFields or []):
            name = str(sf.name or "").strip()
            if not name:
                continue
            a = sf.access
            if isinstance(a, F8StateAccess):
                access_map[(node_id, name)] = a

    for n in list(graph.nodes or []):
        if str(n.serviceId) != bus.service_id:
            continue
        node_id = str(n.nodeId or "")
        if not node_id:
            raise ValueError("missing nodeId")
        access_by_name: dict[str, F8StateAccess] = {}
        for sf in list(n.stateFields or []):
            name = str(sf.name or "").strip()
            if not name:
                continue
            a = sf.access
            if isinstance(a, F8StateAccess):
                access_by_name[name] = a

        values = n.stateValues or {}
        if isinstance(values, dict):
            for k in list(values.keys()):
                key = str(k)
                a = access_by_name.get(key)
                if a is None:
                    raise ValueError(f"unknown state value: {node_id}.{key}")
                if a == F8StateAccess.ro:
                    raise ValueError(f"read-only state cannot be set by rungraph: {node_id}.{key}")

    for e in list(graph.edges or []):
        if e.kind != F8EdgeKindEnum.state:
            continue
        if str(e.toServiceId) != bus.service_id:
            continue
        to_node = str(e.toOperatorId or "")
        to_field = str(e.toPort or "")
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
