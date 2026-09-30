"""One owned pending refresh per visualization; closing prevents late emissions."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class ThrottledFlusher:
    def __init__(self, flush: Callable[[int], Awaitable[None]], *, name: str) -> None:
        self._flush = flush
        self._name = name
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._last_ms = 0
        self._lock = asyncio.Lock()
        self._last_error: tuple[type[BaseException], str] | None = None

    async def schedule(self, *, now_ms: int, throttle_ms: int) -> None:
        if self._closed:
            return
        target = self._last_ms + max(0, throttle_ms)
        if self._last_ms <= 0 or now_ms >= target:
            await self._cancel_pending()
            await self._flush_now(now_ms)
        elif self._task is None or self._task.done():
            self._task = asyncio.create_task(self._flush_after(target - now_ms), name=self._name)
            self._task.add_done_callback(self._finished)

    async def _flush_now(self, now_ms: int) -> None:
        async with self._lock:
            if self._closed:
                return
            await self._flush(now_ms)
            self._last_ms = now_ms
            self._last_error = None

    async def _flush_after(self, delay_ms: int) -> None:
        await asyncio.sleep(delay_ms / 1000)
        await self._flush_now(int(time.time() * 1000))

    def _finished(self, task: asyncio.Task[None]) -> None:
        if self._task is task:
            self._task = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            key = (type(error), str(error))
            if key != self._last_error:
                logger.error("visualization refresh failed task=%s", self._name, exc_info=error)
            self._last_error = key

    async def _cancel_pending(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def close(self) -> None:
        self._closed = True
        await self._cancel_pending()
        # Also wait for an immediate flush owned by a caller before detach.
        async with self._lock:
            return
