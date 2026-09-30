from __future__ import annotations

from f8pysdk.codec import dump_json
import asyncio
import enum
import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from ...generated import F8StateAccess
from ...codec import parse_bool, unwrap_json_value
from ...f8_naming import ensure_token
from ...state import StateWriteContext, StateWriteError, StateWriteOrigin, StateWriteSource
from ..internal.logging import log_error_once
from .helpers import build_intra_state_route_meta
from .options import StatePublishOptions
from ...time_utils import now_ms
from ...codec import encode_obj
from ...zenoh_naming import zenoh_state_key
from ..internal.command import dispatch_command_input
from ._errors import STATE_LOCAL_DELIVERY_ERRORS, STATE_NODE_CALLBACK_ERRORS, STATE_ROUTE_PROPAGATION_ERRORS

if TYPE_CHECKING:
    from ..runtime import ServiceBus


log = logging.getLogger(__name__)


@dataclass
class _StateUpdate:
    node_id: str
    field: str
    value: Any
    ts_ms: int
    origin: StateWriteOrigin
    source: str
    actor: str
    meta: dict[str, Any]
    seq: int


def origin_allows_access(origin: StateWriteOrigin, access: F8StateAccess) -> bool:
    if origin == StateWriteOrigin.system:
        return True
    if origin == StateWriteOrigin.runtime:
        return access in (F8StateAccess.rw, F8StateAccess.ro)
    if origin == StateWriteOrigin.rungraph:
        return access in (F8StateAccess.rw, F8StateAccess.wo)
    if origin == StateWriteOrigin.external:
        return access in (F8StateAccess.rw, F8StateAccess.wo)
    return False


def coerce_state_value(value: Any) -> Any:
    """
    Best-effort conversion of state values into JSON-friendly primitives.

    This prevents accidental persistence of wrapped schema/value objects
    as strings, which then breaks runtime numeric coercion.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [coerce_state_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): coerce_state_value(v) for k, v in value.items()}

    # Enum-like objects.
    if isinstance(value, enum.Enum):
        return coerce_state_value(value.value)

    # msgspec structs and other dump-able model objects.
    try:
        dumped = dump_json(value)  # type: ignore[attr-defined]
        return coerce_state_value(dumped)
    except (AttributeError, TypeError, ValueError):
        pass

    unwrapped = unwrap_json_value(value)
    if unwrapped is not value:
        return coerce_state_value(unwrapped)

    return value


def _is_runtime_builtin_service_state(bus: "ServiceBus", *, node_id: str, field: str, ctx: StateWriteContext) -> bool:
    if str(node_id) != str(bus.service_id):
        return False
    if str(field) not in ("active", "svcId"):
        return False
    return ctx.origin in (StateWriteOrigin.runtime, StateWriteOrigin.system)


def _build_state_payload(update: _StateUpdate) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "value": update.value,
        "actor": update.actor,
        "tsMs": int(update.ts_ms),
        "source": update.source,
        "origin": update.origin.value,
        "seq": int(update.seq),
    }
    for k, v in dict(update.meta or {}).items():
        if k in ("value", "actor", "tsMs", "source", "origin", "seq"):
            continue
        payload[k] = v
    return payload


async def _route_intra_state_edges(
    bus: "ServiceBus",
    *,
    node_id: str,
    field: str,
    value: Any,
    ts_ms: int,
    options: StatePublishOptions,
) -> None:
    if not options.fanout_intra_state_edges:
        return
    node_id_s = str(node_id)
    field_s = str(field)
    targets = bus.state_router.intra_targets(node_id=node_id_s, field=field_s)
    if not targets:
        return
    for to_node, to_field, _edge in list(targets):
        to_node_s = str(to_node)
        to_field_s = str(to_field)
        if to_node_s == node_id_s and to_field_s == field_s:
            continue
        access = bus.state_store.access_for(node_id=to_node_s, field=to_field_s)
        if access not in (F8StateAccess.rw, F8StateAccess.wo):
            continue
        try:
            await publish_state(
                bus,
                to_node_s,
                to_field_s,
                value,
                ts_ms=int(ts_ms),
                origin=StateWriteOrigin.external,
                source=StateWriteSource.state_edge_intra,
                meta=build_intra_state_route_meta(from_node_id=node_id_s, from_field=field_s),
            )
        except StateWriteError as exc:
            log_error_once(
                bus,
                key=f"intra_state_route_write_error:{to_node_s}:{to_field_s}",
                message=f"intra-state propagation rejected for {to_node_s}.{to_field_s}",
                exc=exc,
            )
            continue
        except STATE_ROUTE_PROPAGATION_ERRORS as exc:
            log_error_once(
                bus,
                key=f"intra_state_route_unexpected_error:{to_node_s}:{to_field_s}",
                message=f"intra-state propagation failed for {to_node_s}.{to_field_s}",
                exc=exc,
            )
            continue


async def _deliver_state_local(
    bus: "ServiceBus",
    node_id: str,
    field: str,
    value: Any,
    ts_ms: int,
    meta_dict: dict[str, Any],
    options: StatePublishOptions,
) -> None:
    node_id_s = str(node_id)
    field_s = str(field)
    command_gateway = bus.command_gateway
    if command_gateway.is_hidden_field(node_id=node_id_s, field=field_s):
        binding = command_gateway.input_binding(node_id=node_id_s, field=field_s)
        if binding is not None:
            await dispatch_command_input(
                bus,
                node_id=node_id_s,
                field=field_s,
                value=value,
                ts_ms=int(ts_ms),
                meta=dict(meta_dict),
            )
            return
        await _route_intra_state_edges(
            bus,
            node_id=node_id_s,
            field=field_s,
            value=value,
            ts_ms=int(ts_ms),
            options=options,
        )
        return
    node = bus._nodes.get(node_id_s)
    if node is None:
        return
    try:
        await node.on_state(field_s, value, ts_ms=int(ts_ms))
    except STATE_NODE_CALLBACK_ERRORS as exc:
        log_error_once(
            bus,
            key=f"node_on_state_failed:{node_id_s}:{field_s}",
            message=f"node.on_state failed for {node_id_s}.{field_s}",
            exc=exc,
        )
    await _route_intra_state_edges(
        bus,
        node_id=node_id_s,
        field=field_s,
        value=value,
        ts_ms=int(ts_ms),
        options=options,
    )


def _resolve_publish_options(
    *, meta: dict[str, Any] | None, options: StatePublishOptions | None
) -> tuple[StatePublishOptions, dict[str, Any]]:
    meta_dict = dict(meta or {})
    if options is not None:
        return options, meta_dict
    return StatePublishOptions(), meta_dict


async def validate_state_update(
    bus: "ServiceBus",
    *,
    node_id: str,
    field: str,
    value: Any,
    ts_ms: int,
    meta: dict[str, Any] | None,
    ctx: StateWriteContext,
) -> Any:
    """
    Centralized state validation hook.

    If a node implements `validate_state(field, value, ts_ms=..., meta=...)`, it may:
    - return a (possibly transformed) value to accept
    - raise StateWriteError/ValueError to reject
    """
    node_id_s = str(node_id)
    field_s = str(field)
    node = bus._nodes.get(node_id_s)

    if _is_runtime_builtin_service_state(bus, node_id=node_id_s, field=field_s, ctx=ctx):
        return value

    access = bus.state_store.access_for(node_id=node_id_s, field=field_s)
    # If we have an applied graph, unknown fields are rejected.
    if bus._graph is not None and access is None:
        raise StateWriteError(
            "UNKNOWN_FIELD",
            f"unknown state field: {node_id_s}.{field_s}",
            details={"nodeId": node_id_s, "field": field_s},
        )

    # Enforce write access when known.
    if access is not None and not origin_allows_access(ctx.origin, access):
        raise StateWriteError(
            "FORBIDDEN",
            f"state field not writable: {node_id_s}.{field_s} ({access.value})",
            details={
                "nodeId": node_id_s,
                "field": field_s,
                "access": access.value,
                "origin": ctx.origin.value,
            },
        )

    if bus.command_gateway.is_hidden_field(node_id=node_id_s, field=field_s):
        return value

    if node is None:
        return value
    try:
        meta_dict = dict(meta or {})
        meta_dict.setdefault("origin", ctx.origin.value)
        meta_dict.setdefault("source", ctx.resolved_source)
        r = node.validate_state(field_s, value, ts_ms=int(ts_ms), meta=meta_dict)
        if asyncio.iscoroutine(r):
            return await r
        return r
    except StateWriteError:
        raise
    except ValueError as exc:
        raise StateWriteError("INVALID_VALUE", str(exc)) from exc
    except STATE_NODE_CALLBACK_ERRORS as exc:
        raise StateWriteError("INVALID_VALUE", str(exc)) from exc


async def publish_state(
    bus: "ServiceBus",
    node_id: str,
    field: str,
    value: Any,
    *,
    origin: StateWriteOrigin,
    ts_ms: int | None = None,
    source: StateWriteSource | str | None = None,
    meta: dict[str, Any] | None = None,
    deliver_local: bool = True,
    options: StatePublishOptions | None = None,
) -> None:
    """
    Persist and locally apply a state update.

    `meta` is payload/validation metadata only. Runtime propagation controls
    should be expressed via `StatePublishOptions`.
    """
    node_id = ensure_token(node_id, label="node_id")
    field = str(field)
    if node_id == bus.service_id and field == "active" and origin != StateWriteOrigin.runtime:
        active = parse_bool(value)
        if active is None:
            raise StateWriteError(
                "INVALID_VALUE",
                "active must be a boolean",
                details={"nodeId": node_id, "field": field},
            )
        await bus.set_active(bool(active), source=source or StateWriteSource.endpoint, meta=dict(meta or {}))
        return
    ts = int(ts_ms or now_ms())
    ctx = StateWriteContext(origin=origin, source=source)
    publish_options, payload_meta = _resolve_publish_options(meta=meta, options=options)
    update = _StateUpdate(
        node_id=node_id,
        field=field,
        value=value,
        ts_ms=ts,
        origin=ctx.origin,
        source=ctx.resolved_source,
        actor=bus.service_id,
        meta=payload_meta,
        seq=bus._next_state_publish_seq(),
    )
    payload = _build_state_payload(update)
    update.value = await validate_state_update(
        bus,
        node_id=node_id,
        field=field,
        value=payload.get("value"),
        ts_ms=int(payload.get("tsMs") or now_ms()),
        meta=dict(payload),
        ctx=ctx,
    )
    update.value = coerce_state_value(update.value)
    payload = _build_state_payload(update)

    key = zenoh_state_key(bus.service_id, node_id=node_id, field=field)
    is_hidden_command_state = bus.command_gateway.is_hidden_field(node_id=node_id, field=field)
    # Value-dedupe: avoid republishing identical values.
    #
    # This is important for intra-service state edges: a graph may contain
    # cycles, and without dedupe a value can loop until hop-limit cutoff.
    existing = bus.state_store.cache_entry(node_id=node_id, field=field)
    if existing is not None and not is_hidden_command_state and not publish_options.force_publish:
        try:
            if existing[0] == update.value:
                if bus._debug_state:
                    print(
                        "state_debug[%s] publish_state dedupe node=%s field=%s"
                        % (bus.service_id, node_id, field)
                    )
                return
        except (TypeError, ValueError):
            pass
    if bus._debug_state:
        print(
            "state_debug[%s] publish_state node=%s field=%s ts=%s origin=%s source=%s"
            % (bus.service_id, node_id, field, str(payload.get("tsMs")), ctx.origin.value, update.source)
        )
    await bus._transport.retained_put(key, encode_obj(payload))
    bus.state_store.cache_value(node_id=node_id, field=field, value=update.value, ts_ms=int(payload["tsMs"]))
    if deliver_local:
        # Local writes (actor == self.service_id) do not round-trip through the KV watcher.
        # Apply to listeners and the node callback immediately.
        try:
            await _deliver_state_local(
                bus,
                node_id,
                field,
                update.value,
                int(payload["tsMs"]),
                dict(payload),
                publish_options,
            )
        except STATE_LOCAL_DELIVERY_ERRORS as exc:
            log.error(
                "state persisted but local delivery failed service_id=%s node_id=%s field=%s",
                bus.service_id,
                node_id,
                field,
                exc_info=exc,
            )
