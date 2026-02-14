from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

from ..capabilities import LifecycleNode
from ..time_utils import now_ms
from .micro import _ServiceBusMicroEndpoints
from .state_publish import publish_state
from .state_write import StateWriteOrigin
from .state_write import StateWriteSource

if TYPE_CHECKING:
    from .bus import ServiceBus


async def _ensure_micro_endpoints_started(bus: "ServiceBus") -> None:
    if bus._micro_endpoints is not None:
        return
    endpoints = _ServiceBusMicroEndpoints(bus)
    bus._micro_endpoints = endpoints
    await endpoints.start()


async def _stop_micro_endpoints(bus: "ServiceBus") -> None:
    endpoints = bus._micro_endpoints
    if endpoints is not None:
        await endpoints.stop()
    bus._micro_endpoints = None


async def set_active(
    bus: "ServiceBus",
    active: bool,
    *,
    source: StateWriteSource | str | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """
    Set service active state.

    - Persists `active` into KV under `nodes.<service_id>.state.active`
    - Notifies lifecycle nodes + service hooks (engine/executor can pause/resume)
    """
    await apply_active(bus, active, persist=True, source=source, meta=meta)


async def start(bus: "ServiceBus") -> None:
    # Reset termination latch for a fresh run.
    bus._terminate_event = asyncio.Event()
    await bus._transport.connect()
    # Clear any stale ready flag from a previous run as early as possible.
    await announce_ready(bus, False, reason="starting")
    if bus._micro_endpoints is None:
        await _ensure_micro_endpoints_started(bus)
    # Ensure lifecycle state always exists in KV, even when no service code writes it explicitly.
    await apply_active(
        bus,
        bool(bus._active),
        persist=True,
        source=StateWriteSource.system,
        meta={"bootstrap": True},
    )
    await notify_before_ready(bus)
    await announce_ready(bus, True, reason="start")
    await notify_after_ready(bus)


async def stop(bus: "ServiceBus") -> None:
    await notify_before_stop(bus)
    await announce_ready(bus, False, reason="stop")

    await _stop_micro_endpoints(bus)

    for sub in list(bus._custom_subs):
        await sub.unsubscribe()
    bus._custom_subs.clear()

    for sub in list(bus._data_route_subs.values()):
        await sub.unsubscribe()
    bus._data_route_subs.clear()

    bus._cross_in_by_subject.clear()
    bus._intra_data_out.clear()
    bus._intra_data_in.clear()
    bus._cross_out_subjects.clear()
    bus._data_inputs.clear()

    bus._cross_state_in_by_key.clear()
    for (_sid, _key), watch in list(bus._remote_state_watches.items()):
        watcher, task = watch
        task.cancel()
        await watcher.stop()
    bus._remote_state_watches.clear()

    bus._state_cache.clear()

    await bus._transport.close()
    await notify_after_stop(bus)
    bus._rungraph_hooks.clear()
    bus._service_hooks.clear()


async def announce_ready(bus: "ServiceBus", ready: bool, *, reason: str) -> None:
    payload = {
        "serviceId": bus.service_id,
        "ready": bool(ready),
        "reason": str(reason or ""),
        "ts": int(now_ms()),
    }
    raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    await bus._transport.kv_put(bus._ready_key, raw)


async def notify_before_ready(bus: "ServiceBus") -> None:
    for hook in list(bus._service_hooks):
        try:
            r = hook.on_before_ready(bus)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            continue


async def notify_after_ready(bus: "ServiceBus") -> None:
    for hook in list(bus._service_hooks):
        try:
            r = hook.on_after_ready(bus)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            continue


async def notify_before_stop(bus: "ServiceBus") -> None:
    for hook in list(bus._service_hooks):
        try:
            r = hook.on_before_stop(bus)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            continue


async def notify_after_stop(bus: "ServiceBus") -> None:
    for hook in list(bus._service_hooks):
        try:
            r = hook.on_after_stop(bus)
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            continue


async def apply_active(
    bus: "ServiceBus", active: bool, *, persist: bool, source: StateWriteSource | str | None, meta: dict[str, Any] | None
) -> None:
    active = bool(active)

    changed = active != bus._active
    bus._active = active

    if persist:
        await publish_state(
            bus,
            bus.service_id,
            "active",
            bool(active),
            origin=StateWriteOrigin.runtime,
            source=source or StateWriteSource.runtime,
            meta={"lifecycle": True, **(dict(meta or {}))},
        )

    if not changed:
        return

    payload = {"source": str(source or "runtime"), **(dict(meta or {}))}

    for node in list(bus._nodes.values()):
        if not isinstance(node, LifecycleNode):
            continue
        r = node.on_lifecycle(bool(active), dict(payload))
        if asyncio.iscoroutine(r):
            await r

    for hook in list(bus._service_hooks):
        try:
            if bool(active):
                r = hook.on_activate(bus, dict(payload))
            else:
                r = hook.on_deactivate(bus, dict(payload))
            if asyncio.iscoroutine(r):
                await r
        except Exception:
            continue
