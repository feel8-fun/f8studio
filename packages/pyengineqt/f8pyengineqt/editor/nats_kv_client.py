from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from qtpy import QtCore

import nats  # type: ignore[import-not-found]
from nats.js.errors import BucketNotFoundError  # type: ignore[import-not-found]


@dataclass(frozen=True)
class NatsKvConfig:
    url: str
    bucket: str
    history: int = 64


class NatsKvClient(QtCore.QObject):
    """
    Thin Qt-friendly wrapper over NATS JetStream KV (nats-py).

    Emits updates from a background asyncio loop thread.
    """

    statusChanged = QtCore.Signal(str)
    entryUpdated = QtCore.Signal(str, bytes, int)  # key, value, revision
    messageReceived = QtCore.Signal(str, bytes)  # subject, payload

    def __init__(self, config: NatsKvConfig) -> None:
        super().__init__()
        self._config = config
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._nc: Any = None
        self._kv: Any = None
        self._subs: list[Any] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._thread_main, name="NatsKvClient", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(lambda: None)
            except Exception:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None:
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

    def put(self, key: str, value: bytes) -> None:
        loop = self._loop
        kv = self._kv
        if loop is None or kv is None:
            return

        async def _put() -> None:
            await kv.put(str(key), bytes(value))

        try:
            asyncio.run_coroutine_threadsafe(_put(), loop)
        except Exception:
            pass

    def delete(self, key: str) -> None:
        loop = self._loop
        kv = self._kv
        if loop is None or kv is None:
            return

        async def _delete() -> None:
            await kv.delete(str(key))

        try:
            asyncio.run_coroutine_threadsafe(_delete(), loop)
        except Exception:
            pass

    def publish(self, subject: str, payload: bytes) -> None:
        loop = self._loop
        nc = self._nc
        if loop is None or nc is None:
            return

        async def _pub() -> None:
            await nc.publish(str(subject), bytes(payload))

        try:
            asyncio.run_coroutine_threadsafe(_pub(), loop)
        except Exception:
            pass

    def subscribe(self, subject: str, *, queue: str | None = None) -> None:
        loop = self._loop
        nc = self._nc
        if loop is None or nc is None:
            return

        async def _sub() -> None:
            async def _handler(msg: Any) -> None:
                try:
                    self.messageReceived.emit(str(getattr(msg, "subject", subject)), bytes(getattr(msg, "data", b"") or b""))
                except Exception:
                    return

            sub = await nc.subscribe(str(subject), queue=str(queue) if queue else None, cb=_handler)
            self._subs.append(sub)

        try:
            asyncio.run_coroutine_threadsafe(_sub(), loop)
        except Exception:
            pass

    # ---- Internals ----
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        finally:
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None

    async def _run(self) -> None:
        self.statusChanged.emit(f"nats: connecting {self._config.url}")
        try:
            self._nc = await nats.connect(
                servers=[self._config.url],
                connect_timeout=1,
                reconnect_time_wait=0.5,
                max_reconnect_attempts=-1,
            )
        except Exception as exc:
            self.statusChanged.emit(f"nats: connect failed ({exc})")
            return
        js = self._nc.jetstream()
        try:
            self._kv = await js.key_value(self._config.bucket)
        except BucketNotFoundError:
            self._kv = await js.create_key_value(bucket=self._config.bucket, history=int(self._config.history))
        except Exception as exc:
            self.statusChanged.emit(f"nats: kv open failed ({exc})")
            return

        self.statusChanged.emit(f"nats: kv ready bucket={self._config.bucket}")

        # Watch all changes; filter at consumer side.
        try:
            watcher = await self._kv.watchall()
        except Exception as exc:
            self.statusChanged.emit(f"nats: watch failed ({exc})")
            return

        try:
            while not self._stop_evt.is_set():
                try:
                    entry = await watcher.updates(timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    continue
                if entry is None:
                    continue
                try:
                    key = str(getattr(entry, "key", "") or "")
                    value = bytes(getattr(entry, "value", b"") or b"")
                    rev = int(getattr(entry, "revision", 0) or 0)
                except Exception:
                    continue
                if key:
                    self.entryUpdated.emit(key, value, rev)
        finally:
            subs = list(self._subs)
            self._subs = []
            for sub in subs:
                try:
                    await sub.unsubscribe()
                except Exception:
                    pass
            try:
                await watcher.stop()
            except Exception:
                pass
            try:
                await self._nc.drain()
            except Exception:
                pass
            self._nc = None
            self._kv = None
