from __future__ import annotations

import asyncio
import logging

import msgspec

from f8pysdk.codec import decode_as
from f8pysdk.specs import F8MonitorSnapshot

from .events import EventJournal
from .runtime_identity import StudioRuntimeIdentity


logger = logging.getLogger(__name__)


class MonitorEnvelope(msgspec.Struct, frozen=True, kw_only=True):
    value: F8MonitorSnapshot
    ts: int


class RuntimeMonitorStore:
    def __init__(self, events: EventJournal, *, studio_service_id: str | None = None) -> None:
        self._events = events
        self._identity = StudioRuntimeIdentity(studio_service_id)
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
        identity = self._identity.to_public(str(snapshot.serviceId), str(snapshot.nodeId))
        if identity is None:
            return
        snapshot = msgspec.structs.replace(snapshot, serviceId=identity[0], nodeId=identity[1])
        async with self._lock:
            self._latest[identity] = snapshot
        self._events.live.set(
            f"monitor/{snapshot.serviceId}/{snapshot.nodeId}",
            msgspec.to_builtins(snapshot, str_keys=True),
        )

    async def snapshot(self) -> tuple[F8MonitorSnapshot, ...]:
        async with self._lock:
            return tuple(self._latest[key] for key in sorted(self._latest))


__all__ = ["MonitorEnvelope", "RuntimeMonitorStore"]
