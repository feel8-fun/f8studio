from __future__ import annotations

import asyncio
import logging

import msgspec

from f8pysdk.codec import decode_as
from f8pysdk.specs import F8MonitorSnapshot

from .events import EventJournal


logger = logging.getLogger(__name__)


class MonitorEnvelope(msgspec.Struct, frozen=True, kw_only=True):
    value: F8MonitorSnapshot
    ts: int


class RuntimeMonitorStore:
    def __init__(self, events: EventJournal, *, studio_service_id: str | None = None) -> None:
        self._events = events
        self._studio_service_id = studio_service_id
        self._latest: dict[tuple[str, str], F8MonitorSnapshot] = {}
        self._lock = asyncio.Lock()
        self._reported_decode_errors: set[str] = set()

    async def ingest(self, key: str, payload: bytes) -> None:
        try:
            envelope = decode_as(payload, MonitorEnvelope)
        except ValueError as exc:
            signature = f"{type(exc).__name__}:{exc}"
            if signature not in self._reported_decode_errors:
                self._reported_decode_errors.add(signature)
                logger.warning("invalid runtime monitor payload key=%s", key, exc_info=exc)
            return
        snapshot = envelope.value
        service_id = str(snapshot.serviceId)
        if service_id == "studio" and self._studio_service_id is not None:
            return
        if service_id.startswith("studio_"):
            if service_id != self._studio_service_id:
                return
            snapshot = msgspec.structs.replace(
                snapshot,
                serviceId="studio",
                nodeId="studio" if str(snapshot.nodeId) == service_id else snapshot.nodeId,
            )
        identity = (str(snapshot.serviceId), str(snapshot.nodeId))
        async with self._lock:
            self._latest[identity] = snapshot
        await self._events.publish(
            event_type="runtime.monitor",
            scope=f"service:{snapshot.serviceId}",
            payload=msgspec.to_builtins(snapshot, str_keys=True),
            reliable=False,
        )

    async def snapshot(self) -> tuple[F8MonitorSnapshot, ...]:
        async with self._lock:
            return tuple(self._latest[key] for key in sorted(self._latest))


__all__ = ["MonitorEnvelope", "RuntimeMonitorStore"]
