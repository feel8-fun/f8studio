from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from ..generated import F8StateAccess
from ..nats_naming import ensure_token, kv_key_node_state
from .state_write import StateWriteContext, StateWriteError, StateWriteOrigin
from ..time_utils import now_ms

if TYPE_CHECKING:
    from .bus import ServiceBus


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

    This prevents accidental persistence of pydantic RootModel/BaseModel objects
    as strings like "root=0.5", which then breaks runtime numeric coercion.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [coerce_state_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): coerce_state_value(v) for k, v in value.items()}

    # Enum-like objects.
    try:
        import enum

        if isinstance(value, enum.Enum):
            return coerce_state_value(value.value)
    except Exception:
        pass

    # Pydantic v2 models/root models.
    try:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json")
            return coerce_state_value(dumped)
    except Exception:
        pass

    # Generic RootModel-like `root` attribute.
    try:
        if hasattr(value, "root"):
            return coerce_state_value(getattr(value, "root"))
    except Exception:
        pass

    return value


def _build_state_payload(update: _StateUpdate) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "value": update.value,
        "actor": update.actor,
        "ts": int(update.ts_ms),
        "source": update.source,
        "origin": update.origin.value,
    }
    for k, v in dict(update.meta or {}).items():
        if k in ("value", "actor", "ts", "source", "origin"):
            continue
        payload[k] = v
    return payload


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
    node = bus._nodes.get(str(node_id))
    allow_unknown = bool(node.allow_unknown_state_fields) if node is not None else False

    access = bus._state_access_by_node_field.get((str(node_id), str(field)))
    # If we have an applied graph, unknown fields are rejected by default.
    # Nodes may opt into dynamic fields (eg. fan-in aggregators) via `allow_unknown_state_fields=True`.
    if bus._graph is not None and access is None and not allow_unknown:
        raise StateWriteError(
            "UNKNOWN_FIELD",
            f"unknown state field: {node_id}.{field}",
            details={"nodeId": str(node_id), "field": str(field)},
        )

    # Enforce write access when known.
    if access is not None and not origin_allows_access(ctx.origin, access):
        raise StateWriteError(
            "FORBIDDEN",
            f"state field not writable: {node_id}.{field} ({access.value})",
            details={
                "nodeId": str(node_id),
                "field": str(field),
                "access": access.value,
                "origin": ctx.origin.value,
            },
        )

    if node is None:
        return value
    try:
        meta_dict = dict(meta or {})
        meta_dict.setdefault("origin", ctx.origin.value)
        meta_dict.setdefault("source", ctx.resolved_source)
        r = node.validate_state(str(field), value, ts_ms=int(ts_ms), meta=meta_dict)
        if asyncio.iscoroutine(r):
            return await r
        return r
    except StateWriteError:
        raise
    except ValueError as exc:
        raise StateWriteError("INVALID_VALUE", str(exc)) from exc
    except Exception as exc:
        raise StateWriteError("INVALID_VALUE", str(exc)) from exc


async def publish_state(
    bus: "ServiceBus",
    node_id: str,
    field: str,
    value: Any,
    *,
    origin: StateWriteOrigin,
    ts_ms: int | None = None,
    source: str | None = None,
    meta: dict[str, Any] | None = None,
    deliver_local: bool = True,
) -> None:
    node_id = ensure_token(node_id, label="node_id")
    field = str(field)
    ts = int(ts_ms or now_ms())
    ctx = StateWriteContext(origin=origin, source=source)
    update = _StateUpdate(
        node_id=node_id,
        field=field,
        value=value,
        ts_ms=ts,
        origin=ctx.origin,
        source=ctx.resolved_source,
        actor=bus.service_id,
        meta=dict(meta or {}),
    )
    payload = _build_state_payload(update)
    update.value = await validate_state_update(
        bus,
        node_id=node_id,
        field=field,
        value=payload.get("value"),
        ts_ms=int(payload.get("ts") or now_ms()),
        meta=dict(payload),
        ctx=ctx,
    )
    update.value = coerce_state_value(update.value)
    payload = _build_state_payload(update)

    key = kv_key_node_state(node_id=node_id, field=field)
    if bus._debug_state:
        print(
            "state_debug[%s] publish_state node=%s field=%s ts=%s origin=%s source=%s"
            % (bus.service_id, node_id, field, str(payload.get("ts")), ctx.origin.value, update.source)
        )
    await bus._transport.kv_put(key, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
    bus._state_cache[(node_id, field)] = (update.value, int(payload["ts"]))
    if deliver_local:
        # Local writes (actor == self.service_id) do not round-trip through the KV watcher.
        # Apply to listeners and the node callback immediately.
        await bus._deliver_state_local(node_id, field, update.value, int(payload["ts"]), dict(payload))


async def publish_state_runtime(
    bus: "ServiceBus", node_id: str, field: str, value: Any, *, ts_ms: int | None = None
) -> None:
    await publish_state(
        bus,
        node_id,
        field,
        value,
        origin=StateWriteOrigin.runtime,
        source="runtime",
        ts_ms=ts_ms,
    )


async def apply_state_local(
    bus: "ServiceBus",
    node_id: str,
    field: str,
    value: Any,
    *,
    ts_ms: int | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """
    Apply a state update locally (cache + listeners + node callback) without writing to KV.

    This is useful for endpoint-only / fan-in paths where we want UI updates but do not
    want to persist synthetic state fields (eg. aggregated/fan-in keys) to the local bucket.
    """
    node_id = ensure_token(node_id, label="node_id")
    field = str(field)
    ts = int(ts_ms or now_ms())
    bus._state_cache[(node_id, field)] = (value, ts)
    meta_dict = dict(meta or {})
    meta_dict.setdefault("actor", bus.service_id)
    meta_dict.setdefault("ts", ts)
    meta_dict.setdefault("value", value)
    meta_dict.setdefault("source", str(meta_dict.get("source") or "local"))
    meta_dict.setdefault("origin", str(meta_dict.get("origin") or meta_dict.get("source") or "local"))
    await bus._deliver_state_local(node_id, field, value, ts, meta_dict)

