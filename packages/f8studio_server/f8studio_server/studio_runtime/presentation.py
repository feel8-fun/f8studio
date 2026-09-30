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
        self._extension_latest: dict[tuple[str, str], PresentationCommand] = {}
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
            for key in tuple(self._extension_latest):
                if key[0] == node_id:
                    del self._extension_latest[key]
        prefix = f"presentation/{node_id}/"
        if command.endswith(".detach"):
            self._events.live.delete_prefix(prefix)
        elif command in {"viz.text.update", "viz.wave.set", "viz.track.set", "viz.video.set", "viz.audio.set",
                          "viz.three_d.set", "viz.three_d.world_up", "viz.tcode.snapshot"}:
            self._events.live.set(prefix + command, {
                "nodeId": node_id, "command": command, "payload": normalized_payload,
                "tsMs": ts_ms, "seq": self._events.live.sequence + 1,
            })
            return
        if not command.endswith(".detach"):
            self._extension_latest[(node_id, command)] = PresentationCommand(
                node_id=node_id, command=command, payload=normalized_payload, ts_ms=ts_ms,
            )
        # Unknown/extension commands may be incremental and must not coalesce.
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
                ),
            name=f"presentation:{node_id}:{command}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def snapshot(self) -> tuple[PresentationCommand, ...]:
        commands = list(self._extension_latest.values())
        for value in self._events.live.values_with_prefix("presentation/"):
            if not isinstance(value, dict):
                raise TypeError("presentation live value must be an object")
            commands.append(msgspec.convert({
                "nodeId": value["nodeId"], "command": value["command"],
                "payload": value["payload"], "tsMs": value["tsMs"],
            }, type=PresentationCommand))
        return tuple(
            sorted(
                commands,
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
        self._extension_latest.clear()
        self._events.live.delete_prefix("presentation/")


__all__ = ["EventPresentationOutlet", "PresentationOutlet"]
