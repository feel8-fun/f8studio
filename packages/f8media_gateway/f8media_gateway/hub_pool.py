from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar


class SourceHub(Protocol):
    source: str

    @property
    def closed(self) -> bool: ...

    def start(self) -> None: ...

    async def close(self) -> None: ...


H = TypeVar("H", bound=SourceHub)


@dataclass
class _Lease(Generic[H]):
    hub: H
    references: int = 0


@dataclass
class _Gate:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class HubPool(Generic[H]):
    """Event-loop-owned leases; blocking source construction runs in a worker.

    Releases identify an instance, so replacing a failed source cannot transfer
    old consumers' reference counts to the replacement.
    """

    def __init__(self, factory: Callable[[str], H]) -> None:
        self._factory = factory
        self._current: dict[str, _Lease[H]] = {}
        self._leases: dict[int, _Lease[H]] = {}
        self._gates: dict[str, _Gate] = {}
        self._closed = False

    @property
    def source_count(self) -> int:
        return len(self._current)

    async def acquire(self, source: str) -> H:
        if self._closed:
            raise RuntimeError("media source pool is closed")
        gate = self._gates.setdefault(source, _Gate())
        gate.users += 1
        try:
            async with gate.lock:
                if self._closed:
                    raise RuntimeError("media source pool is closed")
                lease = self._current.get(source)
                if lease is None or lease.hub.closed:
                    opening = asyncio.create_task(asyncio.to_thread(self._factory, source))
                    try:
                        hub = await asyncio.shield(opening)
                    except asyncio.CancelledError:
                        # to_thread cannot stop a constructor. Reclaim its result.
                        hub = await opening
                        await hub.close()
                        raise
                    if self._closed:
                        await hub.close()
                        raise RuntimeError("media source pool closed during source startup")
                    lease = _Lease(hub)
                    self._current[source] = lease
                    self._leases[id(hub)] = lease
                    hub.start()
                lease.references += 1
                return lease.hub
        finally:
            gate.users -= 1
            if gate.users == 0:
                del self._gates[source]

    async def release(self, hub: H) -> None:
        lease = self._leases.get(id(hub))
        if lease is None:
            return
        lease.references -= 1
        if lease.references > 0:
            return
        del self._leases[id(hub)]
        if self._current.get(hub.source) is lease:
            del self._current[hub.source]
        await hub.close()

    async def close(self) -> None:
        self._closed = True
        leases = tuple(self._leases.values())
        self._current.clear()
        self._leases.clear()
        for lease in leases:
            await lease.hub.close()
