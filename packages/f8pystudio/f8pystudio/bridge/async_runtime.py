from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any


logger = logging.getLogger(__name__)


class AsyncRuntimeThread:
    """
    Dedicated asyncio event loop running in a background thread.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run, name="pystudio-async", daemon=True)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._closed = threading.Event()

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_until_complete(self._main())
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as exc:
                logger.exception("failed to cancel async tasks on shutdown", exc_info=exc)
            try:
                loop.close()
            except Exception as exc:
                logger.exception("failed to close async loop", exc_info=exc)
            finally:
                self._loop = None
                self._closed.set()

    async def _main(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.05)

    def is_accepting_submissions(self) -> bool:
        loop = self._loop
        if loop is None:
            return False
        if self._stop.is_set() or self._closed.is_set():
            return False
        return not loop.is_closed()

    def submit(self, coro: Any) -> concurrent.futures.Future[Any]:
        coro_obj = coro() if asyncio.iscoroutinefunction(coro) else coro
        if not asyncio.iscoroutine(coro_obj):
            raise TypeError(f"submit(...) requires a coroutine; got {type(coro_obj).__name__}")
        if not self.is_accepting_submissions():
            coro_obj.close()
            raise RuntimeError("async runtime is stopping")

        loop = self._loop
        if loop is None:
            coro_obj.close()
            raise RuntimeError("async runtime is stopping")
        try:
            return asyncio.run_coroutine_threadsafe(coro_obj, loop)
        except RuntimeError:
            coro_obj.close()
            raise

    def stop(self) -> None:
        self._stop.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except RuntimeError as exc:
                logger.debug("async loop wakeup skipped during stop", exc_info=exc)
        self._thread.join(timeout=2.0)
