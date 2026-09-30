from __future__ import annotations

from f8studio_server.errors import InvalidRequestError

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4
from typing import Literal

import msgspec

from f8pysdk.specs import F8JsonValue
from .live import LiveValueHub


StudioEventType = Literal[
    "agent.session.deleted", "agent.session.updated", "asset.created", "asset.updated", "asset.deleted",
    "project.created", "project.updated", "project.deleted", "graph.committed",
    "deploy.queued", "deploy.running", "deploy.cancelled", "deploy.finished",
    "service.process_started", "service.process_stopped", "service.log",
    "runtime.error", "media.error", "server.error", "presentation.command",
]


class EventEnvelope(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    event_id: str
    server_epoch: str
    sequence: int
    type: StudioEventType
    scope: str
    timestamp: str
    payload: F8JsonValue


@dataclass(frozen=True)
class OpenEventStream:
    subscription_id: str
    queue: asyncio.Queue[EventEnvelope | None]
    replay: tuple[EventEnvelope, ...]
    snapshot_required: bool
    current_sequence: int
    oldest_sequence: int


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class EventJournal:
    def __init__(
        self,
        *,
        server_epoch: str,
        retention: int = 2048,
        log_retention: int = 1000,
        subscriber_queue_size: int = 256,
    ) -> None:
        if retention < 1:
            raise InvalidRequestError("event retention must be positive")
        if log_retention < 1:
            raise InvalidRequestError("log retention must be positive")
        if subscriber_queue_size < 1:
            raise InvalidRequestError("subscriber queue size must be positive")
        self.live = LiveValueHub()
        self._server_epoch = server_epoch
        self._events: deque[EventEnvelope] = deque(maxlen=retention)
        self._logs: deque[EventEnvelope] = deque(maxlen=log_retention)
        self._subscriber_queue_size = subscriber_queue_size
        self._subscribers: dict[str, asyncio.Queue[EventEnvelope | None]] = {}
        self._sequence = 0
        self._lock = asyncio.Lock()

    @property
    def server_epoch(self) -> str:
        return self._server_epoch

    async def publish(
        self,
        *,
        event_type: StudioEventType,
        scope: str,
        payload: F8JsonValue,
    ) -> EventEnvelope:
        async with self._lock:
            self._sequence += 1
            event = EventEnvelope(
                event_id=uuid4().hex,
                server_epoch=self._server_epoch,
                sequence=self._sequence,
                type=event_type,
                scope=scope,
                timestamp=_timestamp(),
                payload=payload,
            )
            self._events.append(event)
            if (
                event_type == "service.log"
                or event_type.startswith("deploy.")
                or event_type.startswith("service.process_")
                or event_type in {"runtime.error", "media.error", "server.error"}
            ):
                self._logs.append(event)
            for subscription_id, queue in tuple(self._subscribers.items()):
                if queue.full():
                    self._overflow(subscription_id, queue)
                    continue
                queue.put_nowait(event)
            return event

    async def recent_logs(self, *, limit: int = 500, before_sequence: int | None = None) -> tuple[EventEnvelope, ...]:
        if limit < 1:
            raise InvalidRequestError("log limit must be positive")
        if before_sequence is not None and before_sequence < 1:
            raise InvalidRequestError("before_sequence must be positive")
        async with self._lock:
            logs = tuple(self._logs)
            if before_sequence is not None:
                logs = tuple(event for event in logs if event.sequence < before_sequence)
            return logs[-limit:]

    async def open_stream(
        self,
        *,
        client_epoch: str | None,
        after_sequence: int | None,
    ) -> OpenEventStream:
        async with self._lock:
            oldest_sequence = self._events[0].sequence if self._events else self._sequence + 1
            valid_cursor = (
                client_epoch == self._server_epoch
                and after_sequence is not None
                and after_sequence >= oldest_sequence - 1
                and after_sequence <= self._sequence
            )
            replay = (
                tuple(event for event in self._events if event.sequence > after_sequence)
                if valid_cursor and after_sequence is not None
                else ()
            )
            subscription_id = uuid4().hex
            queue: asyncio.Queue[EventEnvelope | None] = asyncio.Queue(maxsize=self._subscriber_queue_size)
            self._subscribers[subscription_id] = queue
            return OpenEventStream(
                subscription_id=subscription_id,
                queue=queue,
                replay=replay,
                snapshot_required=not valid_cursor,
                current_sequence=self._sequence,
                oldest_sequence=oldest_sequence,
            )

    async def close_stream(self, subscription_id: str) -> None:
        async with self._lock:
            self._subscribers.pop(subscription_id, None)

    def _overflow(self, subscription_id: str, queue: asyncio.Queue[EventEnvelope | None]) -> None:
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(None)
        self._subscribers.pop(subscription_id, None)


__all__ = ["EventEnvelope", "EventJournal", "OpenEventStream"]
