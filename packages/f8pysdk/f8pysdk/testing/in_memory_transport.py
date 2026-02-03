from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


def _match_pattern(pattern: str, key: str) -> bool:
    if pattern == key:
        return True
    if pattern.endswith(">"):
        prefix = pattern[:-1]
        return key.startswith(prefix)
    return False


@dataclass
class InMemoryCluster:
    kv: dict[str, dict[str, bytes]] = field(default_factory=dict)
    kv_watchers: dict[str, list[tuple[str, Callable[[str, bytes], Awaitable[None]]]]] = field(default_factory=dict)
    subs: dict[str, list[Callable[[str, bytes], Awaitable[None]]]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def kv_put(self, bucket: str, key: str, value: bytes) -> None:
        callbacks: list[Callable[[str, bytes], Awaitable[None]]] = []
        async with self.lock:
            self.kv.setdefault(bucket, {})[key] = bytes(value)
            for pattern, cb in list(self.kv_watchers.get(bucket, [])):
                if _match_pattern(pattern, key):
                    callbacks.append(cb)
        for cb in callbacks:
            await cb(key, bytes(value))

    async def kv_get(self, bucket: str, key: str) -> bytes | None:
        async with self.lock:
            return self.kv.get(bucket, {}).get(key)

    def add_kv_watch(self, bucket: str, pattern: str, cb: Callable[[str, bytes], Awaitable[None]]) -> None:
        self.kv_watchers.setdefault(bucket, []).append((pattern, cb))

    def remove_kv_watch(self, bucket: str, pattern: str, cb: Callable[[str, bytes], Awaitable[None]]) -> None:
        watchers = self.kv_watchers.get(bucket)
        if not watchers:
            return
        try:
            watchers.remove((pattern, cb))
        except ValueError:
            return

    async def publish(self, subject: str, payload: bytes) -> None:
        for cb in list(self.subs.get(subject, [])):
            await cb(str(subject), bytes(payload))

    def subscribe(self, subject: str, cb: Callable[[str, bytes], Awaitable[None]]) -> None:
        self.subs.setdefault(subject, []).append(cb)

    def unsubscribe(self, subject: str, cb: Callable[[str, bytes], Awaitable[None]]) -> None:
        subs = self.subs.get(subject)
        if not subs:
            return
        try:
            subs.remove(cb)
        except ValueError:
            return


class _WatchHandle:
    def __init__(self, cluster: InMemoryCluster, bucket: str, pattern: str, cb: Callable[[str, bytes], Awaitable[None]]):
        self._cluster = cluster
        self._bucket = bucket
        self._pattern = pattern
        self._cb = cb

    async def stop(self) -> None:
        self._cluster.remove_kv_watch(self._bucket, self._pattern, self._cb)


class InMemoryTransport:
    """
    Minimal in-memory transport for async tests (KV + pub/sub).

    This mirrors only the subset of methods used by ServiceBus.
    """

    def __init__(self, *, cluster: InMemoryCluster, kv_bucket: str) -> None:
        self._cluster = cluster
        self._kv_bucket = str(kv_bucket)

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def require_client(self) -> Any:
        return self

    async def publish(self, subject: str, payload: bytes) -> None:
        await self._cluster.publish(str(subject), bytes(payload))

    async def subscribe(
        self,
        subject: str,
        *,
        queue: str | None = None,
        cb: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> Any:
        if cb is None:
            return None
        self._cluster.subscribe(str(subject), cb)

        class _Sub:
            async def unsubscribe(self_inner) -> None:
                self._cluster.unsubscribe(str(subject), cb)

        return _Sub()

    async def kv_put(self, key: str, value: bytes) -> None:
        await self._cluster.kv_put(self._kv_bucket, str(key), bytes(value))

    async def kv_get(self, key: str) -> bytes | None:
        return await self._cluster.kv_get(self._kv_bucket, str(key))

    async def kv_watch_in_bucket(
        self, bucket: str, key_pattern: str, *, cb: Callable[[str, bytes], Awaitable[None]]
    ) -> Any:
        self._cluster.add_kv_watch(str(bucket), str(key_pattern), cb)
        handle = _WatchHandle(self._cluster, str(bucket), str(key_pattern), cb)
        task = asyncio.create_task(asyncio.sleep(0), name=f"mem_kv_watch:{bucket}:{key_pattern}")
        return (handle, task)

    async def kv_get_in_bucket(self, bucket: str, key: str) -> bytes | None:
        return await self._cluster.kv_get(str(bucket), str(key))
