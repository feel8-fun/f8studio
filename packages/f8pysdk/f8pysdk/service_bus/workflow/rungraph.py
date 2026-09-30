from __future__ import annotations

from f8pysdk.codec import copy_model, dump_json
import asyncio
import logging
from collections import deque
from typing import Any, TYPE_CHECKING

import msgspec

from ...generated import (
    F8DataPortSpec,
    F8Edge,
    F8EdgeKindEnum,
    F8RuntimeGraph,
    F8RuntimeGraphMeta,
    F8RuntimeNode,
    F8StateAccess,
)
from ..._specs.schema import data_port_payload_kind
from ...codec import unwrap_json_value
from ...f8_naming import data_key
from ...rungraph_fingerprint import build_rungraph_deploy_fingerprint
from ...state import StateRead, StateWriteOrigin, StateWriteSource
from ..internal.logging import log_error_once
from ..state.helpers import build_intra_state_route_meta
from ..state.options import StatePublishOptions
from ...time_utils import now_ms
from ...rungraph_validation import (
    validate_state_edge_targets_writable_or_raise,
    validate_state_edges_or_raise,
)

from .cross_state import (
    stop_unused_cross_state_watches,
    sync_cross_state_watches,
    update_cross_state_bindings,
)
from ..state.pipeline import publish_state
from ...codec import encode_obj
from .metadata import build_builtin_identity_state_meta, build_rungraph_reconcile_meta

if TYPE_CHECKING:
    from ..runtime import ServiceBus


log = logging.getLogger(__name__)

_STREAM_PAYLOAD_KINDS = {"bytes", "video_frame", "audio_chunk"}
_RUNGRAPH_VALIDATION_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError, msgspec.MsgspecError)
_RUNGRAPH_STATE_READ_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    msgspec.MsgspecError,
)
_RUNGRAPH_STATE_PUBLISH_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    msgspec.MsgspecError,
)
# Rungraph hooks are extension/component code. Keep graph deployment alive,
# but preserve a traceback with hook phase and class context.
_RUNGRAPH_HOOK_ERRORS = (Exception,)


def _is_unset(value: object) -> bool:
    return isinstance(value, msgspec.UnsetType)


def _is_service_node_operator(operator_class: object) -> bool:
    return operator_class is None or _is_unset(operator_class)


def _port_payload_kind_text(port: F8DataPortSpec) -> str:
    return data_port_payload_kind(port).value


def _edge_from_node_id(edge: F8Edge) -> str:
    node_id = "" if _is_unset(edge.fromOperatorId) else str(edge.fromOperatorId or "").strip()
    if node_id:
        return node_id
    return str(edge.fromServiceId or "").strip()


def _edge_to_node_id(edge: F8Edge) -> str:
    node_id = "" if _is_unset(edge.toOperatorId) else str(edge.toOperatorId or "").strip()
    if node_id:
        return node_id
    return str(edge.toServiceId or "").strip()


def _find_runtime_node(graph: F8RuntimeGraph, node_id: str) -> F8RuntimeNode | None:
    node_id_s = str(node_id or "").strip()
    if not node_id_s:
        return None
    for node in list(graph.nodes or []):
        if str(node.nodeId or "").strip() == node_id_s:
            return node
    return None


def _find_data_port(ports: list[F8DataPortSpec] | msgspec.UnsetType, port_name: str) -> F8DataPortSpec | None:
    if _is_unset(ports):
        return None
    port_name_s = str(port_name or "").strip()
    if not port_name_s:
        return None
    for port in list(ports or []):
        if not isinstance(port, F8DataPortSpec):
            continue
        if str(port.name or "").strip() == port_name_s:
            return port
    return None


def _edge_uses_stream_payload(graph: F8RuntimeGraph, edge: F8Edge) -> bool:
    from_node_id = _edge_from_node_id(edge)
    to_node_id = _edge_to_node_id(edge)
    from_node = _find_runtime_node(graph, from_node_id) if from_node_id else None
    to_node = _find_runtime_node(graph, to_node_id) if to_node_id else None
    from_port = _find_data_port(from_node.dataOutPorts, str(edge.fromPort)) if from_node is not None else None
    if from_port is not None and _port_payload_kind_text(from_port) in _STREAM_PAYLOAD_KINDS:
        return True
    to_port = _find_data_port(to_node.dataInPorts, str(edge.toPort)) if to_node is not None else None
    if to_port is not None and _port_payload_kind_text(to_port) in _STREAM_PAYLOAD_KINDS:
        return True
    return False


def _with_rungraph_ts(graph: F8RuntimeGraph, ts_ms: int) -> F8RuntimeGraph:
    meta = graph.meta
    if meta is None or isinstance(meta, msgspec.UnsetType):
        meta = F8RuntimeGraphMeta()
    meta2 = copy_model(meta, deep=True, update={"ts": int(ts_ms)})
    return copy_model(graph, deep=True, update={"meta": meta2})


def _rungraph_ts_ms(graph: F8RuntimeGraph) -> int:
    meta = graph.meta
    if meta is None or isinstance(meta, msgspec.UnsetType):
        return 0
    ts = meta.ts
    if ts is None or isinstance(ts, msgspec.UnsetType):
        return 0
    try:
        return int(ts)
    except (TypeError, ValueError) as exc:
        log.warning("rungraph has invalid meta.ts value=%r: %s", ts, exc)
        return 0


def _encode_rungraph_bytes(graph: F8RuntimeGraph) -> bytes:
    payload = dump_json(graph)
    return encode_obj(payload)


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


async def _read_state_for_rungraph_or_none(
    bus: "ServiceBus",
    node_id: str,
    field: str,
    *,
    error_key: str,
    error_message: str,
) -> StateRead | None:
    try:
        return await bus.get_state(node_id, field)
    except _RUNGRAPH_STATE_READ_ERRORS as exc:
        log_error_once(bus, key=error_key, message=error_message, exc=exc)
        return None


async def _publish_state_for_rungraph_or_false(
    bus: "ServiceBus",
    node_id: str,
    field: str,
    value: Any,
    *,
    error_key: str,
    error_message: str,
    origin: StateWriteOrigin,
    source: StateWriteSource,
    ts_ms: int | None = None,
    meta: dict[str, Any] | None = None,
    deliver_local: bool = True,
    options: StatePublishOptions | None = None,
) -> bool:
    try:
        await publish_state(
            bus,
            node_id,
            field,
            value,
            origin=origin,
            source=source,
            ts_ms=ts_ms,
            meta=meta,
            deliver_local=deliver_local,
            options=options,
        )
    except _RUNGRAPH_STATE_PUBLISH_ERRORS as exc:
        log_error_once(bus, key=error_key, message=error_message, exc=exc)
        return False
    return True


def _should_apply_rungraph_state_value(
    existing: StateRead | None,
    value: Any,
    rungraph_ts: int,
    *,
    service_id: str,
    node_id: str,
    field: str,
) -> bool:
    if rungraph_ts <= 0:
        return True
    if existing is None or not existing.found:
        return True
    try:
        if existing.value == value:
            return False
    except (TypeError, ValueError) as exc:
        log.warning(
            "rungraph state reconcile compare failed service_id=%s node_id=%s field=%s: %s",
            service_id,
            node_id,
            field,
            exc,
        )
    try:
        return existing.ts_ms is None or int(existing.ts_ms) < int(rungraph_ts)
    except (TypeError, ValueError) as exc:
        log.warning(
            "rungraph state reconcile timestamp compare failed service_id=%s node_id=%s field=%s ts_ms=%r rungraph_ts=%s: %s",
            service_id,
            node_id,
            field,
            existing.ts_ms,
            rungraph_ts,
            exc,
        )
        return True


async def set_rungraph(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Apply and publish a full rungraph snapshot for this service.

    Invariant: the KV snapshot should represent a successfully-applied (running) rungraph.
    """
    graph2 = _with_rungraph_ts(graph, int(now_ms()))
    ok = await apply_rungraph(bus, graph2)
    if not ok:
        raise RuntimeError("set_rungraph: apply_rungraph failed")
    bus._rungraph_fingerprint = build_rungraph_deploy_fingerprint(graph2)
    raw = _encode_rungraph_bytes(graph2)
    try:
        await asyncio.wait_for(bus._transport.retained_put(bus._rungraph_key, raw), timeout=1.0)
    except (asyncio.TimeoutError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        log.error("rungraph config retained publish failed service_id=%s", bus.service_id, exc_info=exc)


async def apply_rungraph(bus: "ServiceBus", graph: F8RuntimeGraph) -> bool:
    """
    Apply a decoded rungraph model (no JSON encode/decode).
    """
    try:
        await validate_rungraph_or_raise(bus, graph)
    except _RUNGRAPH_VALIDATION_ERRORS as exc:
        _log_rungraph_error_once(
            bus,
            "rungraph_validate_failed",
            f"rungraph rejected by validation: {type(exc).__name__}: {exc}",
            exc,
        )
        return False

    # Service/container nodes use `nodeId == serviceId`.
    for n in list(graph.nodes or []):
        if _is_service_node_operator(n.operatorClass) and str(n.nodeId) != str(n.serviceId):
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
    bus.state_router.reset_remote_state_ordering()
    bus.state_store.set_access_map(state_access_by_node_field)
    bus.command_gateway.refresh_bindings()
    if bus._debug_state:
        node_count = len(list(graph.nodes or []))
        edge_count = len(list(graph.edges or []))
        graph_id = str(graph.graphId or "")
        print("state_debug[%s] rungraph_applied graph=%s nodes=%s edges=%s" % (bus.service_id, graph_id, str(node_count), str(edge_count)))

    await rebuild_routes(bus)
    await apply_rungraph_state_values(bus, graph)
    await seed_builtin_identity_state(bus, graph)

    for hook in list(bus._rungraph_hooks):
        if hook not in bus._rungraph_hooks:
            continue
        try:
            r = hook.on_rungraph(graph)
            if asyncio.iscoroutine(r):
                await r
        except _RUNGRAPH_HOOK_ERRORS as exc:
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
    rungraph_ts = _rungraph_ts_ms(graph)

    concurrency = max(1, int(bus._state_sync_concurrency))
    sem = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task[None]] = []

    async def _seed_one(node_id: str, field: str, value: Any) -> None:
        async with sem:
            access = bus.state_store.access_for(node_id=node_id, field=field)
            if access not in (F8StateAccess.rw, F8StateAccess.wo):
                return
            if bus.state_router.is_cross_state_target(node_id=node_id, field=field):
                return
            unwrapped = unwrap_json_value(value)

            # Reconcile semantics: only seed rungraph snapshot values if KV doesn't
            # already have a newer/equal value (by timestamp). This prevents rungraph
            # deploys from clobbering user/runtime updates.
            if rungraph_ts > 0:
                st = await _read_state_for_rungraph_or_none(
                    bus,
                    node_id,
                    field,
                    error_key=f"rungraph_state_reconcile_read_failed:{node_id}:{field}",
                    error_message=f"failed to read existing state during rungraph reconcile for {node_id}.{field}",
                )
                if not _should_apply_rungraph_state_value(
                    st,
                    unwrapped,
                    rungraph_ts,
                    service_id=bus.service_id,
                    node_id=node_id,
                    field=field,
                ):
                    return

            await _publish_state_for_rungraph_or_false(
                bus,
                node_id,
                field,
                unwrapped,
                origin=StateWriteOrigin.rungraph,
                source=StateWriteSource.rungraph,
                ts_ms=(int(rungraph_ts) if rungraph_ts > 0 else None),
                meta=build_rungraph_reconcile_meta(),
                options=StatePublishOptions(fanout_intra_state_edges=False),
                error_key=f"rungraph_state_seed_failed:{node_id}:{field}",
                error_message=f"failed to seed rungraph state for {node_id}.{field}",
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
        access = bus.state_store.access_for(node_id=to_key[0], field=to_key[1])
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
        root_state = await _read_state_for_rungraph_or_none(
            bus,
            root[0],
            root[1],
            error_key=f"state_edge_init_root_read_failed:{root[0]}:{root[1]}",
            error_message=f"state-edge init failed to read root {root[0]}.{root[1]}",
        )
        if root_state is None:
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

                to_state = await _read_state_for_rungraph_or_none(
                    bus,
                    to_key[0],
                    to_key[1],
                    error_key=f"state_edge_init_target_read_failed:{to_key[0]}:{to_key[1]}",
                    error_message=f"state-edge init failed to read target {to_key[0]}.{to_key[1]}",
                )

                if to_state is not None and to_state.found:
                    try:
                        if to_state.value == from_val:
                            queue.append((to_key, to_state.value))
                            continue
                    except (TypeError, ValueError) as exc:
                        log.debug(
                            "state-edge init target compare failed service_id=%s node_id=%s field=%s",
                            bus.service_id,
                            to_key[0],
                            to_key[1],
                            exc_info=exc,
                        )

                published = await _publish_state_for_rungraph_or_false(
                    bus,
                    to_key[0],
                    to_key[1],
                    from_val,
                    ts_ms=ts0,
                    origin=StateWriteOrigin.external,
                    source=StateWriteSource.state_edge_intra_init,
                    meta=build_intra_state_route_meta(from_node_id=from_key[0], from_field=from_key[1]),
                    error_key=f"state_edge_init_publish_failed:{to_key[0]}:{to_key[1]}",
                    error_message=f"state-edge init failed to publish {to_key[0]}.{to_key[1]}",
                )
                if not published:
                    continue

                # Continue propagation using the post-validation cached value if available.
                try:
                    cached = bus.state_store.cache_entry(node_id=to_key[0], field=to_key[1])
                    next_val = cached[0] if cached is not None else from_val
                except (TypeError, ValueError) as exc:
                    log.debug(
                        "state-edge init cache read failed service_id=%s node_id=%s field=%s",
                        bus.service_id,
                        to_key[0],
                        to_key[1],
                        exc_info=exc,
                    )
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
        if bus.state_store.access_for(node_id=node_id, field="svcId") is not None:
            published_svc_id = await _publish_state_for_rungraph_or_false(
                bus,
                node_id,
                "svcId",
                str(n.serviceId or bus.service_id),
                origin=StateWriteOrigin.system,
                source=StateWriteSource.system,
                ts_ms=ts,
                meta=build_builtin_identity_state_meta(),
                deliver_local=False,
                error_key=f"seed_builtin_identity_state_failed:{node_id}",
                error_message=f"failed to seed builtin identity state for node {node_id}",
            )
            if not published_svc_id:
                continue
        if (
            not _is_service_node_operator(n.operatorClass)
            and bus.state_store.access_for(node_id=node_id, field="operatorId") is not None
        ):
            await _publish_state_for_rungraph_or_false(
                bus,
                node_id,
                "operatorId",
                str(n.nodeId or node_id),
                origin=StateWriteOrigin.system,
                source=StateWriteSource.system,
                ts_ms=ts,
                meta=build_builtin_identity_state_meta(),
                deliver_local=False,
                error_key=f"seed_builtin_identity_state_failed:{node_id}",
                error_message=f"failed to seed builtin identity state for node {node_id}",
            )


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
        except _RUNGRAPH_HOOK_ERRORS as exc:
            raise ValueError(
                f"rungraph hook validation failed: {hook.__class__.__name__}.validate_rungraph: {exc}"
            ) from exc


async def rebuild_routes(bus: "ServiceBus") -> None:
    graph = bus._graph
    if graph is None:
        return

    data_router = bus.data_router
    state_router = bus.state_router
    state_router.clear_intra_state_routes()

    # Intra (in-process) routing: local service -> local service.
    intra: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
    intra_in: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
    input_stream_keys: dict[tuple[str, str], str] = {}
    for edge in graph.edges if isinstance(graph.edges, list) else []:
        if edge.kind != F8EdgeKindEnum.data:
            continue
        if str(edge.fromServiceId) != bus.service_id or str(edge.toServiceId) != bus.service_id:
            continue
        from_node = _edge_from_node_id(edge)
        to_node = _edge_to_node_id(edge)
        if not from_node or not to_node:
            continue
        if _edge_uses_stream_payload(graph, edge):
            key = data_key(str(edge.fromServiceId), from_node_id=from_node, port_id=str(edge.fromPort))
            input_stream_keys[(to_node, str(edge.toPort))] = key
            continue
        intra.setdefault((from_node, str(edge.fromPort)), []).append((to_node, str(edge.toPort), edge))
        intra_in.setdefault((to_node, str(edge.toPort)), []).append((from_node, str(edge.fromPort), edge))
    intra_data_out = {k: tuple(v) for k, v in intra.items()}
    intra_data_in = {k: tuple(v) for k, v in intra_in.items()}

    # Intra-service state fanout: local state edges.
    intra_state_out: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
    for edge in graph.edges if isinstance(graph.edges, list) else []:
        if edge.kind != F8EdgeKindEnum.state:
            continue
        if str(edge.fromServiceId) != bus.service_id or str(edge.toServiceId) != bus.service_id:
            continue
        if not edge.fromOperatorId or not edge.toOperatorId:
            continue
        intra_state_out.setdefault((str(edge.fromOperatorId), str(edge.fromPort)), []).append((str(edge.toOperatorId), str(edge.toPort), edge))
    state_router.replace_intra_state_routes({k: tuple(v) for k, v in intra_state_out.items()})

    # Cross routing.
    cross_in: dict[str, list[tuple[str, str, F8Edge]]] = {}
    cross_out: dict[tuple[str, str], str] = {}
    for edge in graph.edges if isinstance(graph.edges, list) else []:
        if edge.kind != F8EdgeKindEnum.data:
            continue
        if str(edge.fromServiceId) == str(edge.toServiceId):
            continue
        from_node = _edge_from_node_id(edge)
        to_node = _edge_to_node_id(edge)
        if not from_node or not to_node:
            continue

        key = data_key(str(edge.fromServiceId), from_node_id=from_node, port_id=str(edge.fromPort))

        if str(edge.toServiceId) == bus.service_id:
            if _edge_uses_stream_payload(graph, edge):
                input_stream_keys[(to_node, str(edge.toPort))] = key
                continue
            cross_in.setdefault(key, []).append((to_node, str(edge.toPort), edge))
            continue

        if str(edge.fromServiceId) == bus.service_id:
            if _edge_uses_stream_payload(graph, edge):
                continue
            cross_out[(from_node, str(edge.fromPort))] = key

    await data_router.replace_routes(
        intra_data_out=intra_data_out,
        intra_data_in=intra_data_in,
        cross_in_by_key={k: tuple(v) for k, v in cross_in.items()},
        cross_out_keys=cross_out,
        input_stream_keys=input_stream_keys,
    )
    update_cross_state_bindings(bus, graph)
    await stop_unused_cross_state_watches(bus)
