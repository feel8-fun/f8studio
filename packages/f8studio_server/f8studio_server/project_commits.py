from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast
from weakref import WeakValueDictionary

import msgspec
from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import PatchResult

from .events import EventJournal


class ProjectCommits:
    """Order application mutations through runtime sync and event publication."""

    def __init__(self, events: EventJournal, refresh_hotkeys: Callable[[], None]) -> None:
        self._events = events
        self._refresh_hotkeys = refresh_hotkeys
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def lock(self, project_id: str) -> asyncio.Lock:
        # Called only on the application event loop. Idle projects retain no lock.
        lock = self._locks.get(project_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[project_id] = lock
        return lock

    async def publish(self, project_id: str, result: PatchResult) -> PatchResult:
        await asyncio.to_thread(self._refresh_hotkeys)
        await self._events.publish(
            event_type="graph.committed", scope=f"project:{project_id}",
            payload=cast(F8JsonValue, msgspec.to_builtins({
                "requestId": result.request_id, "graphChanged": result.graph_changed,
                "layoutChanged": result.layout_changed, "runtimeErrors": result.runtime_errors,
                "document": result.document,
            }, str_keys=True)),
        )
        return result
