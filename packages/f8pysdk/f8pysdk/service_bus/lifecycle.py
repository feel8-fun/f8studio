from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .state_write import StateWriteSource
from .workflow import lifecycle as _impl

if TYPE_CHECKING:
    from .api.bus import ServiceBus

_ensure_micro_endpoints_started_impl = _impl._ensure_micro_endpoints_started
_stop_micro_endpoints_impl = _impl._stop_micro_endpoints


async def _ensure_micro_endpoints_started(bus: "ServiceBus") -> None:
    await _ensure_micro_endpoints_started_impl(bus)


async def _stop_micro_endpoints(bus: "ServiceBus") -> None:
    await _stop_micro_endpoints_impl(bus)


async def set_active(
    bus: "ServiceBus",
    active: bool,
    *,
    source: StateWriteSource | str | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    await _impl.set_active(bus, active, source=source, meta=meta)


async def start(bus: "ServiceBus") -> None:
    # Compatibility hook: allow patching `f8pysdk.service_bus.lifecycle._ensure_micro_endpoints_started`.
    _impl._ensure_micro_endpoints_started = _ensure_micro_endpoints_started
    _impl._stop_micro_endpoints = _stop_micro_endpoints
    await _impl.start(bus)


async def stop(bus: "ServiceBus") -> None:
    _impl._ensure_micro_endpoints_started = _ensure_micro_endpoints_started
    _impl._stop_micro_endpoints = _stop_micro_endpoints
    await _impl.stop(bus)


async def announce_ready(bus: "ServiceBus", ready: bool, *, reason: str) -> None:
    await _impl.announce_ready(bus, ready, reason=reason)


async def notify_before_ready(bus: "ServiceBus") -> None:
    await _impl.notify_before_ready(bus)


async def notify_after_ready(bus: "ServiceBus") -> None:
    await _impl.notify_after_ready(bus)


async def notify_before_stop(bus: "ServiceBus") -> None:
    await _impl.notify_before_stop(bus)


async def notify_after_stop(bus: "ServiceBus") -> None:
    await _impl.notify_after_stop(bus)


async def apply_active(
    bus: "ServiceBus",
    active: bool,
    *,
    persist: bool,
    source: StateWriteSource | str | None,
    meta: dict[str, Any] | None,
) -> None:
    await _impl.apply_active(bus, active, persist=persist, source=source, meta=meta)


__all__ = [
    "_ensure_micro_endpoints_started",
    "_stop_micro_endpoints",
    "announce_ready",
    "apply_active",
    "notify_after_ready",
    "notify_after_stop",
    "notify_before_ready",
    "notify_before_stop",
    "set_active",
    "start",
    "stop",
]
