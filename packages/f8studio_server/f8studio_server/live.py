from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from f8pysdk.specs import F8JsonValue


@dataclass(eq=False)
class LiveSubscription:
    pending: dict[str, F8JsonValue] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    snapshot: bool = False

    async def next_patch(self) -> dict[str, F8JsonValue]:
        await self.ready.wait()
        self.ready.clear()
        updates, removed = self.pending, self.deleted
        self.pending, self.deleted = {}, set()
        if self.snapshot:
            self.snapshot = False
            return {"type": "live.snapshot", "values": updates}
        return {"type": "live.patch", "set": updates, "delete": sorted(removed)}


class LiveValueHub:
    """Latest-value mirror owned by one event loop, bounded by the key budget.

    Subscribe and copy the snapshot without awaiting, so updates cannot fall
    between snapshot capture and subscription registration.
    """

    def __init__(self, *, max_keys: int = 16384) -> None:
        self._values: dict[str, F8JsonValue] = {}
        self._subscribers: set[LiveSubscription] = set()
        self._max_keys = max_keys
        self._sequence = 0

    @property
    def sequence(self) -> int:
        return self._sequence

    def set(self, key: str, value: F8JsonValue) -> None:
        if key not in self._values and len(self._values) >= self._max_keys:
            self.delete(next(iter(self._values)))
        self._sequence += 1
        self._values[key] = value
        for subscriber in self._subscribers:
            subscriber.deleted.discard(key)
            subscriber.pending[key] = value
            self._bound_pending(subscriber)
            subscriber.ready.set()

    def delete(self, key: str) -> None:
        if key not in self._values:
            return
        del self._values[key]
        for subscriber in self._subscribers:
            subscriber.pending.pop(key, None)
            if not subscriber.snapshot:
                subscriber.deleted.add(key)
            self._bound_pending(subscriber)
            subscriber.ready.set()

    def _bound_pending(self, subscriber: LiveSubscription) -> None:
        if len(subscriber.pending) + len(subscriber.deleted) > self._max_keys:
            subscriber.pending = dict(self._values)
            subscriber.deleted.clear()
            subscriber.snapshot = True

    def delete_prefix(self, prefix: str) -> None:
        for key in tuple(self._values):
            if key.startswith(prefix):
                self.delete(key)

    def subscribe(self) -> tuple[LiveSubscription, dict[str, F8JsonValue]]:
        subscription = LiveSubscription()
        self._subscribers.add(subscription)
        return subscription, dict(self._values)

    def unsubscribe(self, subscription: LiveSubscription) -> None:
        self._subscribers.discard(subscription)
