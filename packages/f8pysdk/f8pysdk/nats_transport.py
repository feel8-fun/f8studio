from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import nats  # type: ignore[import-not-found]
from nats.js.api import KeyValueConfig, StorageType  # type: ignore[import-not-found]
from nats.js.errors import BucketNotFoundError  # type: ignore[import-not-found]


@dataclass(frozen=True)
class NatsTransportConfig:
    url: str
    kv_bucket: str
    kv_history: int = 1
    kv_storage: StorageType = StorageType.MEMORY
    delete_bucket_on_connect: bool = False
    delete_bucket_on_close: bool = False


class NatsTransport:
    """
    Single-process transport for NATS core pub/sub + JetStream KV.

    Intended to be shared by all RuntimeNode instances in a service process.
    """

    def __init__(self, config: NatsTransportConfig) -> None:
        self._config = config
        self._nc: Any = None
        self._js: Any = None
        self._kv: Any = None
        self._kv_stores: dict[str, Any] = {}
        self._subs: list[Any] = []
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._nc is not None

    @property
    def raw_client(self) -> Any | None:
        """
        Underlying nats.py client (if connected).
        """
        return self._nc

    async def require_client(self) -> Any:
        """
        Return connected nats.py client (connects if needed).
        """
        if self._nc is None:
            await self.connect()
        if self._nc is None:
            raise RuntimeError("NATS not connected")
        return self._nc

    async def connect(self) -> None:
        async with self._lock:
            if self._nc is not None:
                return
            url = str(self._config.url or "nats://127.0.0.1:4222").strip()
            last_log = 0.0
            attempt = 0

            # nats.py's default error callback can print very noisy tracebacks on connect failures.
            # Provide our own callback to keep logs clean.
            last_err_log = 0.0

            async def _error_cb(exc: Exception) -> None:
                nonlocal last_err_log
                now = time.monotonic()
                if (now - last_err_log) < 2.0:
                    return
                last_err_log = now
                try:
                    print(f"[f8] NATS connection error (will retry): {type(exc).__name__}: {exc}")
                except Exception:
                    pass

            while self._nc is None:
                attempt += 1
                try:
                    self._nc = await nats.connect(
                        servers=[url],
                        connect_timeout=2,
                        reconnect_time_wait=0.5,
                        max_reconnect_attempts=-1,
                        error_cb=_error_cb,
                    )
                except Exception as exc:
                    now = time.monotonic()
                    # Friendly reminder without traceback spam.
                    if attempt == 1 or (now - last_log) >= 2.0:
                        last_log = now
                        try:
                            print(
                                f"[f8] NATS server is not reachable at {url!r}. "
                                f"Start `nats-server` or set `F8_NATS_URL`. retrying... ({type(exc).__name__})"
                            )
                        except Exception:
                            pass
                    await asyncio.sleep(min(2.0, 0.2 * attempt))
                    continue
            self._js = self._nc.jetstream()
            if bool(self._config.delete_bucket_on_connect) and self._js is not None:
                try:
                    await self._js.delete_key_value(str(self._config.kv_bucket))
                except Exception:
                    pass
            self._kv = await self._open_kv(self._config.kv_bucket)
            if self._kv is not None:
                self._kv_stores[self._config.kv_bucket] = self._kv

    async def _open_kv(self, bucket: str) -> Any:
        if self._js is None:
            raise RuntimeError("JetStream not initialized")
        try:
            return await self._js.key_value(str(bucket))
        except BucketNotFoundError:
            cfg = KeyValueConfig(
                bucket=str(bucket),
                history=int(self._config.kv_history),
                storage=self._config.kv_storage,
            )
            return await self._js.create_key_value(config=cfg)

    async def close(self) -> None:
        async with self._lock:
            subs = list(self._subs)
            self._subs.clear()
            for sub in subs:
                try:
                    await sub.unsubscribe()
                except Exception:
                    pass

            if bool(self._config.delete_bucket_on_close) and self._js is not None:
                try:
                    await self._js.delete_key_value(str(self._config.kv_bucket))
                except Exception:
                    pass

            if self._nc is not None:
                try:
                    await self._nc.drain()
                except Exception:
                    pass
            self._nc = None
            self._js = None
            self._kv = None
            self._kv_stores.clear()

    # --- Core pub/sub ----------------------------------------------------
    async def publish(self, subject: str, payload: bytes) -> None:
        if self._nc is None:
            await self.connect()
        if self._nc is None:
            return
        await self._nc.publish(str(subject), bytes(payload))

    async def request(
        self, subject: str, payload: bytes, *, timeout: float = 1.0, raise_on_error: bool = False
    ) -> bytes | None:
        """
        Request/reply helper (core NATS).
        """
        if self._nc is None:
            await self.connect()
        if self._nc is None:
            return None
        try:
            msg = await self._nc.request(str(subject), bytes(payload), timeout=float(timeout))
        except Exception:
            if raise_on_error:
                raise
            return None
        try:
            return bytes(msg.data or b"")  # type: ignore[attr-defined]
        except Exception:
            return None

    async def subscribe(
        self,
        subject: str,
        *,
        queue: str | None = None,
        cb: Callable[[str, bytes], Awaitable[None]] | None = None,
    ) -> Any:
        if self._nc is None:
            await self.connect()
        if self._nc is None:
            raise RuntimeError("NATS not connected")

        async def _handler(msg: Any) -> None:
            if cb is None:
                return
            try:
                await cb(str(msg.subject or subject), bytes(msg.data or b""))  # type: ignore[attr-defined]
            except Exception:
                return

        sub = await self._nc.subscribe(str(subject), queue=str(queue) if queue else None, cb=_handler)
        self._subs.append(sub)
        return sub

    async def subscribe_msg(
        self,
        subject: str,
        *,
        queue: str | None = None,
        cb: Callable[[Any], Awaitable[None]] | None = None,
    ) -> Any:
        """
        Subscribe with raw nats.py message objects (needed for request/reply).
        """
        if self._nc is None:
            await self.connect()
        if self._nc is None:
            raise RuntimeError("NATS not connected")

        async def _handler(msg: Any) -> None:
            if cb is None:
                return
            try:
                await cb(msg)
            except Exception:
                return

        sub = await self._nc.subscribe(str(subject), queue=str(queue) if queue else None, cb=_handler)
        self._subs.append(sub)
        return sub

    # --- KV --------------------------------------------------------------
    async def kv_put(self, key: str, value: bytes) -> None:
        if self._kv is None:
            await self.connect()
        if self._kv is None:
            return
        await self._kv.put(str(key), bytes(value))

    async def kv_get(self, key: str) -> bytes | None:
        if self._kv is None:
            await self.connect()
        if self._kv is None:
            return None
        try:
            entry = await self._kv.get(str(key))
        except Exception:
            return None
        if entry is None:
            return None
        try:
            return bytes(entry.value or b"")  # type: ignore[attr-defined]
        except Exception:
            return None

    async def kv_watch(self, key_pattern: str, *, cb: Callable[[str, bytes], Awaitable[None]]) -> Any:
        """
        Watch KV updates for a key pattern (eg. exact key or `prefix.>`).
        """
        if self._kv is None:
            await self.connect()
        if self._kv is None:
            raise RuntimeError("KV not available")

        watcher = await self._kv.watch(str(key_pattern))

        async def _pump() -> None:
            while True:
                try:
                    entry = await watcher.updates(timeout=0.5)
                except Exception:
                    continue
                if entry is None:
                    continue
                try:
                    k = str(entry.key or "")  # type: ignore[attr-defined]
                    v = bytes(entry.value or b"")  # type: ignore[attr-defined]
                except Exception:
                    continue
                if not k:
                    continue
                try:
                    await cb(k, v)
                except Exception:
                    continue

        task = asyncio.create_task(_pump(), name=f"kv_watch:{key_pattern}")
        return (watcher, task)

    async def kv_store(self, bucket: str) -> Any:
        """
        Returns a KV store for the given bucket (creates if missing).
        """
        if self._nc is None or self._js is None:
            await self.connect()
        if self._js is None:
            raise RuntimeError("JetStream not available")
        bucket = str(bucket)
        if bucket in self._kv_stores:
            return self._kv_stores[bucket]
        kv = await self._open_kv(bucket)
        self._kv_stores[bucket] = kv
        return kv

    async def kv_watch_in_bucket(
        self, bucket: str, key_pattern: str, *, cb: Callable[[str, bytes], Awaitable[None]]
    ) -> Any:
        """
        Watch KV updates within an explicit bucket.
        """
        kv = await self.kv_store(bucket)
        watcher = await kv.watch(str(key_pattern))

        async def _pump() -> None:
            while True:
                try:
                    entry = await watcher.updates(timeout=0.5)
                except Exception:
                    continue
                if entry is None:
                    continue
                try:
                    k = str(entry.key or "")  # type: ignore[attr-defined]
                    v = bytes(entry.value or b"")  # type: ignore[attr-defined]
                except Exception:
                    continue
                if not k:
                    continue
                try:
                    await cb(k, v)
                except Exception:
                    continue

        task = asyncio.create_task(_pump(), name=f"kv_watch:{bucket}:{key_pattern}")
        return (watcher, task)

    async def kv_get_in_bucket(self, bucket: str, key: str) -> bytes | None:
        """
        Read a KV entry from an explicit bucket (returns raw bytes or None).
        """
        try:
            kv = await self.kv_store(str(bucket))
        except Exception:
            return None
        try:
            entry = await kv.get(str(key))
        except Exception:
            return None
        if entry is None:
            return None
        try:
            return bytes(entry.value or b"")  # type: ignore[attr-defined]
        except Exception:
            return None


async def reset_kv_bucket(
    *,
    url: str,
    kv_bucket: str,
    kv_storage: StorageType = StorageType.MEMORY,
    timeout_s: float = 2.5,
) -> None:
    """
    Delete and recreate a JetStream KV bucket (best-effort).

    Used by Studio to clear stale runtime state (eg. `ready=true`) before starting a new service process.
    """
    tr = NatsTransport(
        NatsTransportConfig(
            url=str(url).strip(),
            kv_bucket=str(kv_bucket).strip(),
            kv_storage=kv_storage,
            delete_bucket_on_connect=True,
        )
    )
    await asyncio.wait_for(tr.connect(), timeout=float(timeout_s))
    await tr.close()


def reset_kv_bucket_sync(
    *,
    url: str,
    kv_bucket: str,
    kv_storage: StorageType = StorageType.MEMORY,
    timeout_s: float = 2.5,
) -> None:
    """
    Synchronous wrapper for `reset_kv_bucket` (for non-async entrypoints).
    """
    asyncio.run(
        reset_kv_bucket(url=url, kv_bucket=kv_bucket, kv_storage=kv_storage, timeout_s=float(timeout_s))
    )
