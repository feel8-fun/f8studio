from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any, TYPE_CHECKING

from ..generated import F8Edge, F8EdgeKindEnum, F8RuntimeGraph, F8RuntimeGraphMeta, F8StateAccess
from ..json_unwrap import unwrap_json_value
from ..nats_naming import data_subject
from .state_write import StateWriteOrigin, StateWriteSource
from ..time_utils import now_ms
from ..rungraph_validation import (
    validate_state_edge_targets_writable_or_raise,
    validate_state_edges_or_raise,
)

from .cross_state import (
    stop_unused_cross_state_watches,
    sync_cross_state_watches,
    update_cross_state_bindings,
)
from .error_utils import log_error_once
from .state_publish import publish_state
from .routing_data import precreate_input_buffers_for_cross_in, sync_subscriptions

if TYPE_CHECKING:
    from .bus import ServiceBus


log = logging.getLogger(__name__)


def _with_rungraph_ts(graph: F8RuntimeGraph, ts_ms: int) -> F8RuntimeGraph:
    meta = graph.meta if graph.meta is not None else F8RuntimeGraphMeta()
    meta2 = meta.model_copy(deep=True, update={"ts": int(ts_ms)})
    return graph.model_copy(deep=True, update={"meta": meta2})


def _encode_rungraph_bytes(graph: F8RuntimeGraph) -> bytes:
    payload = graph.model_dump(mode="json", by_alias=True)
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _log_rungraph_error_once(bus: "ServiceBus", key: str, message: str, exc: BaseException | None = None) -> None:
    """
    Log rungraph apply errors once per bus instance to avoid log spam.
    """
    if key in bus._rungraph_apply_error_once:
        return
    bus._rungraph_apply_error_once.add(key)
    if exc is None:
        log.warning("rungraph_apply[%s] %s", bus.service_id, message)
        return
    log.error("rungraph_apply[%s] %s", bus.service_id, message, exc_info=exc)


async def set_rungraph(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Apply and publish a full rungraph snapshot for this service.

    Invariant: the KV snapshot should represent a successfully-applied (running) rungraph.
    """
    graph2 = _with_rungraph_ts(graph, int(now_ms()))
    ok = await apply_rungraph(bus, graph2)
    if not ok:
        raise RuntimeError("set_rungraph: apply_rungraph failed")
    raw = _encode_rungraph_bytes(graph2)
    await bus._transport.kv_put(bus._rungraph_key, raw)


async def apply_rungraph(bus: "ServiceBus", graph: F8RuntimeGraph) -> bool:
    """
    Apply a decoded rungraph model (no JSON encode/decode).
    """
    try:
        await validate_rungraph_or_raise(bus, graph)
    except Exception as exc:
        _log_rungraph_error_once(
            bus,
            "rungraph_validate_failed",
            f"rungraph rejected by validation: {type(exc).__name__}: {exc}",
        )
        return False

    # Service/container nodes use `nodeId == serviceId`.
    for n in list(graph.nodes or []):
        if n.operatorClass is None and str(n.nodeId) != str(n.serviceId):
            _log_rungraph_error_once(bus, "rungraph_invalid_service_node", "service node requires nodeId == serviceId")
            return False

    # Build local node state-access map (used by validators, endpoints, and routing).
    state_access_by_node_field: dict[tuple[str, str], F8StateAccess] = {}
    for n in list(graph.nodes or []):
        if str(n.serviceId or "") != bus.service_id:
            continue
        node_id = str(n.nodeId or "").strip()
        if not node_id:
            continue
        for sf in list(n.stateFields or []):
            name = str(sf.name or "").strip()
            if not name:
                continue
            access = sf.access
            if isinstance(access, F8StateAccess):
                state_access_by_node_field[(node_id, name)] = access

    bus._graph = graph
    # Reset cross-state ordering on graph changes.
    bus._cross_state_last_ts.clear()
    # Cache local node state access for enforcement and filtering.
    bus._state_access_by_node_field = state_access_by_node_field
    if bus._debug_state:
        node_count = len(list(graph.nodes or []))
        edge_count = len(list(graph.edges or []))
        graph_id = str(graph.graphId or "")
        print("state_debug[%s] rungraph_applied graph=%s nodes=%s edges=%s" % (bus.service_id, graph_id, str(node_count), str(edge_count)))

    await rebuild_routes(bus)
    await apply_rungraph_state_values(bus, graph)
    await seed_builtin_identity_state(bus, graph)

    for hook in list(bus._rungraph_hooks):
        try:
            r = hook.on_rungraph(graph)
            if asyncio.iscoroutine(r):
                await r
        except Exception as exc:
            # This is a boundary for user hook code; don't crash the bus.
            _log_rungraph_error_once(
                bus,
                f"rungraph_hook_failed:{hook.__class__.__name__}",
                f"rungraph hook failed: {hook.__class__.__name__}.on_rungraph",
                exc,
            )
    # Cross-state watches and initial sync are intentionally installed AFTER
    # rungraph hooks so local runtime nodes are registered first. This avoids
    # dropping initial remote values due to missing nodes/fields.
    await sync_cross_state_watches(bus)
    # With strong cross-state sync, materialize remote values first (no fanout),
    # then run a single ordered intra-service init propagation.
    await initial_sync_intra_state_edges(bus, graph)
    return True


async def apply_rungraph_state_values(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Materialize per-node `stateValues` into KV (and dispatch locally).
    """
    try:
        rungraph_ts = int(graph.meta.ts or 0) if graph.meta is not None else 0
    except Exception:
        rungraph_ts = 0

    concurrency = max(1, int(bus._state_sync_concurrency))
    sem = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task[None]] = []

    async def _seed_one(node_id: str, field: str, value: Any) -> None:
        async with sem:
            access = bus._state_access_by_node_field.get((node_id, field))
            if access not in (F8StateAccess.rw, F8StateAccess.wo):
                return
            if (node_id, field) in bus._cross_state_targets:
                return
            unwrapped = unwrap_json_value(value)

            # Reconcile semantics: only seed rungraph snapshot values if KV doesn't
            # already have a newer/equal value (by timestamp). This prevents rungraph
            # deploys from clobbering user/runtime updates.
            if rungraph_ts > 0:
                try:
                    st = await bus.get_state(node_id, field)
                except Exception as exc:
                    log_error_once(
                        bus,
                        key=f"rungraph_state_reconcile_read_failed:{node_id}:{field}",
                        message=f"failed to read existing state during rungraph reconcile for {node_id}.{field}",
                        exc=exc,
                    )
                    st = None
                if st is not None and st.found:
                    try:
                        if st.value == unwrapped:
                            return
                    except (TypeError, ValueError):
                        pass
                    try:
                        if st.ts_ms is not None and int(st.ts_ms) > int(rungraph_ts):
                            return
                    except (TypeError, ValueError):
                        pass

            try:
                await publish_state(
                    bus,
                    node_id,
                    field,
                    unwrapped,
                    origin=StateWriteOrigin.rungraph,
                    source=StateWriteSource.rungraph,
                    ts_ms=(int(rungraph_ts) if rungraph_ts > 0 else None),
                    meta={"via": "rungraph", "rungraphReconcile": True, "_noStateFanout": True},
                )
            except Exception as exc:
                log_error_once(
                    bus,
                    key=f"rungraph_state_seed_failed:{node_id}:{field}",
                    message=f"failed to seed rungraph state for {node_id}.{field}",
                    exc=exc,
                )

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
            task = asyncio.create_task(
                _seed_one(node_id, field, v),
                name=f"service_bus:rungraph_seed:{node_id}:{field}",
            )
            tasks.append(task)

    if tasks:
        await asyncio.gather(*tasks)


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
            print("state_debug[%s] state_edge_init_no_roots (cycle?)" % (bus.service_id,))
        return

    ts0 = int(now_ms())

    # Propagate only from roots that have an actual value in KV/cache.
    visited: set[tuple[str, str]] = set()
    for root in list(roots):
        try:
            root_state = await bus.get_state(root[0], root[1])
        except Exception as exc:
            log_error_once(
                bus,
                key=f"state_edge_init_root_read_failed:{root[0]}:{root[1]}",
                message=f"state-edge init failed to read root {root[0]}.{root[1]}",
                exc=exc,
            )
            continue
        if not root_state.found:
            continue

        queue: deque[tuple[tuple[str, str], object]] = deque([(root, root_state.value)])
        seen_in_component: set[tuple[str, str]] = set()
        while queue:
            from_key, from_val = queue.popleft()
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
                except Exception as exc:
                    log_error_once(
                        bus,
                        key=f"state_edge_init_target_read_failed:{to_key[0]}:{to_key[1]}",
                        message=f"state-edge init failed to read target {to_key[0]}.{to_key[1]}",
                        exc=exc,
                    )
                    to_state = None

                if to_state is not None and to_state.found:
                    try:
                        if to_state.value == from_val:
                            queue.append((to_key, to_state.value))
                            continue
                    except (TypeError, ValueError):
                        pass

                try:
                    await publish_state(
                        bus,
                        to_key[0],
                        to_key[1],
                        from_val,
                        ts_ms=ts0,
                        origin=StateWriteOrigin.external,
                        source=StateWriteSource.state_edge_intra_init,
                        meta={"fromNodeId": from_key[0], "fromField": from_key[1]},
                    )
                except Exception as exc:
                    log_error_once(
                        bus,
                        key=f"state_edge_init_publish_failed:{to_key[0]}:{to_key[1]}",
                        message=f"state-edge init failed to publish {to_key[0]}.{to_key[1]}",
                        exc=exc,
                    )
                    continue

                # Continue propagation using the post-validation cached value if available.
                try:
                    cached = bus._state_cache.get(to_key)
                    next_val = cached[0] if cached is not None else from_val
                except (TypeError, ValueError):
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
                await publish_state(
                    bus,
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
                await publish_state(
                    bus,
                    node_id,
                    "operatorId",
                    str(n.nodeId or node_id),
                    origin=StateWriteOrigin.system,
                    source=StateWriteSource.system,
                    ts_ms=ts,
                    meta={"builtin": True, "_noStateFanout": True},
                    deliver_local=False,
                )
        except Exception as exc:
            log_error_once(
                bus,
                key=f"seed_builtin_identity_state_failed:{node_id}",
                message=f"failed to seed builtin identity state for node {node_id}",
                exc=exc,
            )
            continue


async def validate_rungraph_or_raise(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Validate the rungraph before applying it.
    """
    # Global state-edge constraints (covers cross-service cycles too).
    validate_state_edges_or_raise(graph, forbid_cycles=True, forbid_multi_upstream=True)
    validate_state_edge_targets_writable_or_raise(graph, local_service_id=bus.service_id)

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
    update_cross_state_bindings(bus, graph)
    await stop_unused_cross_state_watches(bus)
