from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any


_TICK_LOOP_ERRORS = (Exception,)


class PyScriptTickScheduler:
    def __init__(
        self,
        *,
        node_id: str,
        tick_enabled: bool,
        tick_ms: int,
        now_ms: Callable[[], int],
        is_closing: Callable[[], bool],
        is_tick_allowed: Callable[[], bool],
        run_tick: Callable[[dict[str, int]], Awaitable[None]],
        log_error: Callable[[str, str, BaseException], None],
    ) -> None:
        self._node_id = str(node_id)
        self._tick_enabled = bool(tick_enabled)
        self._tick_ms = self.coerce_tick_ms(tick_ms, default=100)
        self._now_ms = now_ms
        self._is_closing = is_closing
        self._is_tick_allowed = is_tick_allowed
        self._run_tick = run_tick
        self._log_error = log_error
        self._task: asyncio.Task[object] | None = None
        self._seq = 0

    @property
    def tick_ms(self) -> int:
        return int(self._tick_ms)

    @staticmethod
    def coerce_tick_ms(value: Any, *, default: int) -> int:
        try:
            out = int(value)
        except (TypeError, ValueError):
            out = int(default)
        return max(1, out)

    def set_enabled(self, enabled: bool) -> None:
        self._tick_enabled = bool(enabled)
        self.ensure_task()

    def set_tick_ms(self, value: Any) -> None:
        self._tick_ms = self.coerce_tick_ms(value, default=self._tick_ms)

    def ensure_task(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self._tick_loop(), name=f"pyscript:tick:{self._node_id}")

    async def shutdown(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _can_tick(self) -> bool:
        return bool(self._tick_enabled and self._is_tick_allowed())

    async def _tick_loop(self) -> None:
        last_tick_ts = self._now_ms()
        next_deadline = time.monotonic()
        while not self._is_closing():
            try:
                if not self._can_tick():
                    next_deadline = time.monotonic()
                    await asyncio.sleep(0.05)
                    continue

                now_mono = time.monotonic()
                wait_s = next_deadline - now_mono
                if wait_s > 0:
                    await asyncio.sleep(wait_s)

                if not self._can_tick():
                    next_deadline = time.monotonic()
                    continue

                current_ts_ms = self._now_ms()
                delta_ms = max(0, current_ts_ms - last_tick_ts)
                last_tick_ts = current_ts_ms
                self._seq += 1
                await self._run_tick(
                    {
                        "seq": int(self._seq),
                        "tsMs": int(current_ts_ms),
                        "deltaMs": int(delta_ms),
                    }
                )

                interval_s = max(0.001, float(self._tick_ms) / 1000.0)
                now_after = time.monotonic()
                if next_deadline <= now_after:
                    next_deadline = now_after + interval_s
                else:
                    next_deadline += interval_s
            except asyncio.CancelledError:
                raise
            except _TICK_LOOP_ERRORS as exc:
                self._log_error("tick_loop", "tick loop failed", exc)
                await asyncio.sleep(0.05)


__all__ = ["PyScriptTickScheduler"]
