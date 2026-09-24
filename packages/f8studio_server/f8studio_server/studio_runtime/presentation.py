from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, cast

import msgspec

from f8pysdk.specs import F8JsonValue

from ..events import EventJournal
from ..models import PresentationCommand


logger = logging.getLogger(__name__)


class PresentationOutlet(Protocol):
    def emit(
        self,
        node_id: str,
        command: str,
        payload: dict[str, Any],
        *,
        ts_ms: int | None = None,
    ) -> None: ...


class EventPresentationOutlet:
    def __init__(self, events: EventJournal) -> None:
        self._events = events
        self._tasks: set[asyncio.Task[object]] = set()
        self._latest: dict[tuple[str, str], PresentationCommand] = {}
        self._closed = False

    def emit(
        self,
        node_id: str,
        command: str,
        payload: dict[str, Any],
        *,
        ts_ms: int | None = None,
    ) -> None:
        if self._closed:
            return
        normalized_payload = cast(F8JsonValue, msgspec.to_builtins(payload, str_keys=True))
        if not isinstance(normalized_payload, dict):
            raise TypeError("presentation payload must encode to an object")
        if command.endswith(".detach"):
            for key in tuple(self._latest):
                if key[0] == node_id:
                    del self._latest[key]
        else:
            self._latest[(node_id, command)] = PresentationCommand(
                node_id=node_id,
                command=command,
                payload=normalized_payload,
                ts_ms=ts_ms,
            )
        task = asyncio.create_task(
            self._events.publish(
                event_type="presentation.command",
                scope=f"node:{node_id}",
                payload={
                    "nodeId": node_id,
                    "command": command,
                    "payload": normalized_payload,
                    "tsMs": ts_ms,
                },
                reliable=False,
            ),
            name=f"presentation:{node_id}:{command}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def snapshot(self) -> tuple[PresentationCommand, ...]:
        return tuple(
            sorted(
                self._latest.values(),
                key=lambda item: (
                    item.ts_ms if item.ts_ms is not None else -1,
                    item.node_id,
                    item.command,
                ),
            )
        )

    def _task_done(self, task: asyncio.Task[object]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("presentation event publication failed", exc_info=error)

    async def close(self) -> None:
        self._closed = True
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._latest.clear()


__all__ = ["EventPresentationOutlet", "PresentationOutlet"]
