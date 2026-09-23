from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


def _match_pattern(pattern: str, key: str) -> bool:
    if pattern == key:
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return key == prefix or key.startswith(f"{prefix}/")
    if pattern.endswith(">"):
        prefix = pattern[:-1]
        return key.startswith(prefix)
    return False


@dataclass
class InMemoryCluster:
    retained: dict[str, bytes] = field(default_factory=dict)
    retained_watchers: list[tuple[str, Callable[[str, bytes], Awaitable[None]]]] = field(default_factory=list)
    subs: dict[str, list[Callable[[str, bytes], Awaitable[None]]]] = field(default_factory=dict)
    request_handlers: dict[str, Callable[[bytes], Awaitable[bytes | None]]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def retained_put(self, key: str, value: bytes) -> None:
        key_s = str(key).strip("/")
        callbacks: list[Callable[[str, bytes], Awaitable[None]]] = []
        async with self.lock:
            self.retained[key_s] = bytes(value)
            for pattern, cb in list(self.retained_watchers):
                if _match_pattern(pattern, key_s):
                    callbacks.append(cb)
        for cb in callbacks:
            await cb(key_s, bytes(value))

    async def retained_get(self, key: str) -> bytes | None:
        async with self.lock:
            return self.retained.get(str(key).strip("/"))

    def add_retained_watch(self, pattern: str, cb: Callable[[str, bytes], Awaitable[None]]) -> None:
        self.retained_watchers.append((str(pattern).strip("/"), cb))

    def remove_retained_watch(self, pattern: str, cb: Callable[[str, bytes], Awaitable[None]]) -> None:
        try:
            self.retained_watchers.remove((str(pattern).strip("/"), cb))
        except ValueError:
            return

    async def publish(self, key: str, payload: bytes) -> None:
        for cb in list(self.subs.get(key, [])):
            await cb(str(key), bytes(payload))

    def subscribe(self, key: str, cb: Callable[[str, bytes], Awaitable[None]]) -> None:
        self.subs.setdefault(key, []).append(cb)

    def unsubscribe(self, key: str, cb: Callable[[str, bytes], Awaitable[None]]) -> None:
        subs = self.subs.get(key)
        if not subs:
            return
        try:
            subs.remove(cb)
        except ValueError:
            return

    async def request(self, key: str, payload: bytes) -> bytes | None:
        handler = self.request_handlers.get(str(key))
        if handler is None:
            return None
        result = await handler(bytes(payload))
        if result is None:
            return None
        return bytes(result)

    def serve(self, key: str, handler: Callable[[bytes], Awaitable[bytes | None]]) -> None:
        self.request_handlers[str(key)] = handler

    def unserve(self, key: str, handler: Callable[[bytes], Awaitable[bytes | None]]) -> None:
        existing = self.request_handlers.get(str(key))
        if existing is handler:
            self.request_handlers.pop(str(key), None)


class _WatchHandle:
    def __init__(self, cluster: InMemoryCluster, pattern: str, cb: Callable[[str, bytes], Awaitable[None]]):
        self._cluster = cluster
        self._pattern = pattern
        self._cb = cb

    async def stop(self) -> None:
        self._cluster.remove_retained_watch(self._pattern, self._cb)

    async def unsubscribe(self) -> None:
        await self.stop()


class _ServeHandle:
    def __init__(
        self,
        cluster: InMemoryCluster,
        key: str,
        handler: Callable[[bytes], Awaitable[bytes | None]],
    ) -> None:
        self._cluster = cluster
        self._key = key
        self._handler = handler

    async def unsubscribe(self) -> None:
        self._cluster.unserve(self._key, self._handler)

    async def stop(self) -> None:
        await self.unsubscribe()


class InMemoryTransport:
    """
    Minimal in-memory transport for async tests (retained keys + pub/sub).

    This mirrors only the subset of methods used by ServiceBus.
    """

    def __init__(self, *, cluster: InMemoryCluster) -> None:
        self._cluster = cluster

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def require_client(self) -> Any:
        return self

    async def publish(self, key: str, payload: bytes) -> None:
        await self._cluster.publish(str(key).strip("/"), bytes(payload))

    async def subscribe(
        self,
        key_expr: str,
        *,
        queue: str | None = None,
        cb: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> Any:
        if cb is None:
            return None
        cluster = self._cluster
        key_name = str(key_expr).strip("/")
        cluster.subscribe(key_name, cb)

        class _Sub:
            async def unsubscribe(self) -> None:
                cluster.unsubscribe(key_name, cb)

        return _Sub()

    async def request(
        self,
        key: str,
        payload: bytes,
        *,
        timeout: float = 1.0,
        raise_on_error: bool = False,
    ) -> bytes | None:
        try:
            return await asyncio.wait_for(self._cluster.request(str(key).strip("/"), bytes(payload)), timeout=float(timeout))
        except asyncio.TimeoutError:
            if raise_on_error:
                raise
            return None

    async def serve(self, key: str, handler: Callable[[bytes], Awaitable[bytes | None]]) -> _ServeHandle:
        key_name = str(key).strip("/")
        self._cluster.serve(key_name, handler)
        return _ServeHandle(self._cluster, key_name, handler)

    async def retained_put(self, key: str, value: bytes) -> None:
        await self._cluster.retained_put(str(key), bytes(value))

    async def retained_get(self, key: str) -> bytes | None:
        return await self._cluster.retained_get(str(key))

    async def retained_watch(
        self,
        key_expr: str,
        *,
        cb: Callable[[str, bytes], Awaitable[None]],
        with_initial: bool = True,
    ) -> Any:
        pattern = str(key_expr).strip("/")
        self._cluster.add_retained_watch(pattern, cb)
        if with_initial:
            for key, value in list(self._cluster.retained.items()):
                if _match_pattern(pattern, key):
                    await cb(key, bytes(value))
        return _WatchHandle(self._cluster, pattern, cb)
