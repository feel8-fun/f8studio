from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from .runtime_transport import RequestHandler, TransportCallback
from .zenoh_config import apply_zenoh_shared_memory_config, apply_zenoh_timestamping_config
from .zenoh_naming import (
    zenoh_key_to_state_path,
    zenoh_service_liveliness_key,
)
from .zenoh_shutdown import close_zenoh_session_best_effort

log = logging.getLogger(__name__)
# Zenoh can return a subscriber declaration before local matching is fully
# visible to other sessions. This small setup-only pause prevents losing the
# first state/data sample immediately after a watch is installed.
_SUBSCRIPTION_SETTLE_S = 0.01
_QUERYABLE_SETTLE_S = 0.05
_ZENOH_LIFECYCLE_ERRORS = (OSError, RuntimeError, TypeError, ValueError)
_ZENOH_QUERY_ERRORS = (OSError, RuntimeError, TimeoutError, TypeError, ValueError)
_ZENOH_PUMP_ERRORS = (OSError, RuntimeError, TypeError, ValueError)
_TRANSPORT_CALLBACK_ERRORS = (Exception,)


def _zenoh_error_types() -> tuple[type[BaseException], ...]:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError:
        return (RuntimeError,)
    error_type = zenoh.ZError
    if isinstance(error_type, type) and issubclass(error_type, BaseException):
        return (error_type,)
    return (RuntimeError,)


@dataclass(frozen=True)
class ZenohTransportConfig:
    service_id: str
    runtime_instance_id: str = ""
    announce_service_liveliness: bool = False
    config_path: str | None = None
    connect: tuple[str, ...] = ()
    listen: tuple[str, ...] = ()
    shm_pool_bytes: int = 256 * 1024 * 1024


StreamDeliveryPolicy = Literal["latest", "fifo", "reliable"]


class _ZenohInbox:
    """Bridge Zenoh callback threads into one ordered asyncio consumer."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False

    def receive(self, value: Any) -> None:
        if not self.closed and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._deliver, value)

    def _deliver(self, value: Any) -> None:
        if not self.closed:
            self.queue.put_nowait(value)

    def close(self) -> None:
        self.closed = True
        while not self.queue.empty():
            self.queue.get_nowait()


class _ZenohSubscriptionHandle:
    def __init__(self, declaration: Any, task: asyncio.Task[None], inbox: _ZenohInbox) -> None:
        self._inbox = inbox
        self._declaration = declaration
        self._task = task
        self._closed = False

    async def unsubscribe(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._inbox.close()
        task = self._task
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        try:
            self._declaration.undeclare()
        except (*_ZENOH_LIFECYCLE_ERRORS, *_zenoh_error_types()) as exc:
            log.debug("zenoh undeclare subscription failed", exc_info=exc)

    async def stop(self) -> None:
        await self.unsubscribe()


class _ZenohServeHandle(_ZenohSubscriptionHandle):
    pass


class ZenohTransport:
    """
    Zenoh runtime transport.

    State is service-owned: this transport keeps the latest local KV snapshot in
    memory and publishes retained state samples through Zenoh advanced
    publisher cache/history. Remote watchers receive the current value on
    subscribe without a hot-path queryable or storage-manager dependency.
    """

    def __init__(self, config: ZenohTransportConfig) -> None:
        service_id = str(config.service_id or "").strip()
        self._config = ZenohTransportConfig(
            service_id=service_id,
            runtime_instance_id=str(config.runtime_instance_id or "").strip(),
            announce_service_liveliness=bool(config.announce_service_liveliness),
            config_path=str(config.config_path).strip() if config.config_path else None,
            connect=tuple(str(item).strip() for item in config.connect if str(item).strip()),
            listen=tuple(str(item).strip() for item in config.listen if str(item).strip()),
            shm_pool_bytes=max(0, int(config.shm_pool_bytes)),
        )
        self._session: Any | None = None
        self._liveliness_token: Any | None = None
        self._subs: list[_ZenohSubscriptionHandle] = []
        self._serve_handles: list[_ZenohServeHandle] = []
        self._retained: dict[str, bytes] = {}
        self._state_publishers: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        async with self._lock:
            if self._session is not None:
                return
            try:
                import zenoh  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "Zenoh backend requires the `eclipse-zenoh` Python package. "
                    "Install repo dependencies or use --bus-backend mem for local tests."
                ) from exc

            config = self._build_config(zenoh)
            self._session = await asyncio.to_thread(zenoh.open, config)
            if self._config.announce_service_liveliness:
                runtime_instance_id = self._config.runtime_instance_id
                if not runtime_instance_id:
                    raise ValueError("runtime_instance_id is required when announce_service_liveliness=True")
                self._liveliness_token = self._session.liveliness().declare_token(
                    zenoh_service_liveliness_key(self._config.service_id, runtime_instance_id)
                )

    def _build_config(self, zenoh_module: Any) -> Any:
        path = self._config.config_path
        if path:
            config = zenoh_module.Config.from_file(path)
        else:
            config = zenoh_module.Config()
        if self._config.connect:
            config.insert_json5("connect/endpoints", json.dumps(list(self._config.connect)))
        if self._config.listen:
            config.insert_json5("listen/endpoints", json.dumps(list(self._config.listen)))
        apply_zenoh_shared_memory_config(
            config,
            zenoh_module=zenoh_module,
            shm_pool_bytes=self._config.shm_pool_bytes,
            log_context=f"runtime:{self._config.service_id}",
        )
        apply_zenoh_timestamping_config(
            config,
            zenoh_module=zenoh_module,
            log_context=f"runtime:{self._config.service_id}",
        )
        return config

    async def close(self) -> None:
        async with self._lock:
            handles = [*self._subs, *self._serve_handles]
            self._subs.clear()
            self._serve_handles.clear()
            for handle in handles:
                await handle.unsubscribe()

            for publisher in list(self._state_publishers.values()):
                try:
                    publisher.undeclare()
                except (*_ZENOH_LIFECYCLE_ERRORS, *_zenoh_error_types()) as exc:
                    log.debug("zenoh retained state publisher undeclare failed", exc_info=exc)
            self._state_publishers.clear()

            token = self._liveliness_token
            self._liveliness_token = None
            if token is not None:
                try:
                    token.undeclare()
                except (*_ZENOH_LIFECYCLE_ERRORS, *_zenoh_error_types()) as exc:
                    log.debug("zenoh liveliness undeclare failed", exc_info=exc)

            session = self._session
            self._session = None
            if session is not None:
                close_zenoh_session_best_effort(
                    session,
                    context=f"runtime:{self._config.service_id}",
                    native_close=False,
                )

    async def publish(self, key: str, payload: bytes) -> None:
        await self.publish_stream(key, payload, delivery="latest")

    async def publish_stream(
        self,
        key: str,
        payload: bytes,
        *,
        delivery: StreamDeliveryPolicy = "latest",
    ) -> None:
        session = await self._require_session()
        if delivery == "latest":
            await asyncio.to_thread(_put_realtime_drop, session, _normalize_zenoh_key(key), bytes(payload))
            return
        if delivery == "fifo":
            await asyncio.to_thread(_put_data_fifo, session, _normalize_zenoh_key(key), bytes(payload))
            return
        if delivery == "reliable":
            await asyncio.to_thread(_put_reliable_control, session, _normalize_zenoh_key(key), bytes(payload))
            return
        raise ValueError(f"unsupported stream delivery policy: {delivery!r}")

    async def subscribe(
        self,
        key_expr: str,
        *,
        queue: str | None = None,
        cb: TransportCallback | None = None,
    ) -> _ZenohSubscriptionHandle:
        return await self.subscribe_stream(key_expr, queue=queue, cb=cb)

    async def subscribe_stream(
        self,
        key_expr: str,
        *,
        queue: str | None = None,
        cb: TransportCallback | None = None,
    ) -> _ZenohSubscriptionHandle:
        del queue
        session = await self._require_session()
        key_expr = _normalize_zenoh_key_expr(key_expr)
        inbox = _ZenohInbox()
        declaration = await asyncio.to_thread(session.declare_subscriber, key_expr, inbox.receive)
        task = asyncio.create_task(
            self._pump_subscriber(inbox.queue, key_expr=key_expr, cb=cb, key_converter=lambda item: item),
            name=f"zenoh_sub:{key_expr}",
        )
        handle = _ZenohSubscriptionHandle(declaration, task, inbox)
        self._subs.append(handle)
        await asyncio.sleep(_SUBSCRIPTION_SETTLE_S)
        return handle

    async def request(
        self,
        key: str,
        payload: bytes,
        *,
        timeout: float = 1.0,
        raise_on_error: bool = False,
    ) -> bytes | None:
        try:
            return await self.query_once(key, payload, timeout=timeout, raise_on_error=True)
        except (*_ZENOH_QUERY_ERRORS, *_zenoh_error_types()) as exc:
            if raise_on_error:
                raise
            log.debug("zenoh query request failed key=%s", key, exc_info=exc)
            return None

    async def serve(self, key: str, handler: RequestHandler) -> _ZenohServeHandle:
        return await self.serve_queryable(key, handler)

    async def query_once(
        self,
        key: str,
        payload: bytes,
        *,
        timeout: float = 1.0,
        raise_on_error: bool = False,
    ) -> bytes | None:
        session = await self._require_session()
        query_key = _normalize_zenoh_key(key)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()

        def _on_reply(reply: Any) -> None:
            sample = reply.ok
            if sample is not None:
                data = bytes(sample.payload)
                loop.call_soon_threadsafe(_set_future_result, future, data)
                return
            err = reply.err
            message = "zenoh query returned an error"
            if err is not None:
                try:
                    message = bytes(err.payload).decode("utf-8", errors="replace") or message
                except (TypeError, ValueError, UnicodeError) as exc:
                    log.debug("zenoh query error payload decode failed key=%s", query_key, exc_info=exc)
            loop.call_soon_threadsafe(_set_future_exception, future, RuntimeError(message))

        try:
            await asyncio.to_thread(
                _query_once,
                session,
                query_key,
                bytes(payload),
                _on_reply,
                int(max(1.0, float(timeout) * 1000.0)),
            )
            return await asyncio.wait_for(future, timeout=max(0.001, float(timeout)))
        except asyncio.TimeoutError:
            if raise_on_error:
                raise TimeoutError(f"zenoh query request timed out key={key!r}") from None
            return None
        except RuntimeError:
            if raise_on_error:
                raise
            return None

    async def serve_queryable(self, key: str, handler: RequestHandler) -> _ZenohServeHandle:
        session = await self._require_session()
        key_expr = _normalize_zenoh_key(key)
        inbox = _ZenohInbox()
        declaration = await asyncio.to_thread(_declare_queryable, session, key_expr, inbox.receive)
        task = asyncio.create_task(
            self._pump_queryable(inbox.queue, key_expr=key_expr, handler=handler),
            name=f"zenoh_queryable:{key_expr}",
        )
        handle = _ZenohServeHandle(declaration, task, inbox)
        self._serve_handles.append(handle)
        await asyncio.sleep(_QUERYABLE_SETTLE_S)
        return handle

    async def retained_put(self, key: str, value: bytes) -> None:
        key_s = str(key or "").strip()
        if not key_s:
            raise ValueError("key must be non-empty")
        retained_key = _normalize_zenoh_key(key_s)
        raw = bytes(value)
        self._retained[retained_key] = raw
        session = await self._require_session()
        publisher = self._state_publishers.get(retained_key)
        if publisher is None:
            publisher = await asyncio.to_thread(_declare_retained_state_publisher, session, retained_key)
            self._state_publishers[retained_key] = publisher
            await asyncio.sleep(_SUBSCRIPTION_SETTLE_S)
        await asyncio.to_thread(publisher.put, raw)

    async def retained_get(self, key: str) -> bytes | None:
        return self._retained.get(_normalize_zenoh_key(key))

    async def retained_watch(
        self,
        key_expr: str,
        *,
        cb: TransportCallback,
        with_initial: bool = True,
    ) -> _ZenohSubscriptionHandle:
        session = await self._require_session()
        key_expr_s = _normalize_zenoh_key_expr(key_expr)
        inbox = _ZenohInbox()
        if with_initial:
            declaration = await asyncio.to_thread(_declare_retained_state_subscriber, session, key_expr_s, inbox.receive)
        else:
            declaration = await asyncio.to_thread(session.declare_subscriber, key_expr_s, inbox.receive)

        task = asyncio.create_task(
            self._pump_subscriber(inbox.queue, key_expr=key_expr_s, cb=cb, key_converter=lambda item: item),
            name=f"zenoh_retained_watch:{key_expr_s}",
        )
        handle = _ZenohSubscriptionHandle(declaration, task, inbox)
        self._subs.append(handle)
        await asyncio.sleep(_SUBSCRIPTION_SETTLE_S)
        return handle

    async def _require_session(self) -> Any:
        if self._session is None:
            await self.connect()
        if self._session is None:
            raise RuntimeError("Zenoh not connected")
        return self._session

    async def _pump_subscriber(
        self,
        inbox: asyncio.Queue[Any],
        *,
        key_expr: str,
        cb: TransportCallback | None,
        key_converter: Callable[[str], str],
    ) -> None:
        while True:
            try:
                sample = await inbox.get()
                if cb is not None:
                    try:
                        await cb(key_converter(str(sample.key_expr)), bytes(sample.payload))
                    except _TRANSPORT_CALLBACK_ERRORS as exc:
                        log.error("zenoh subscriber callback failed key_expr=%s", key_expr, exc_info=exc)
                        await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                raise
            except (*_ZENOH_PUMP_ERRORS, *_zenoh_error_types()) as exc:
                log.error("zenoh subscriber pump failed key_expr=%s", key_expr, exc_info=exc)
                await asyncio.sleep(0.05)

    async def _pump_queryable(self, inbox: asyncio.Queue[Any], *, key_expr: str, handler: RequestHandler) -> None:
        while True:
            try:
                query = await inbox.get()
                try:
                    payload = bytes(query.payload) if query.payload is not None else b""
                    try:
                        response = await handler(payload)
                    except _TRANSPORT_CALLBACK_ERRORS as exc:
                        log.error("zenoh queryable handler failed key_expr=%s", key_expr, exc_info=exc)
                        await asyncio.to_thread(_reply_query_error, query, b"query handler failed")
                        continue
                    if response is None:
                        await asyncio.to_thread(_reply_query_error, query, b"empty response")
                        continue
                    await asyncio.to_thread(_reply_query_ok, query, bytes(response))
                finally:
                    query.drop()
            except asyncio.CancelledError:
                raise
            except (*_ZENOH_PUMP_ERRORS, *_zenoh_error_types()) as exc:
                log.error("zenoh queryable pump failed key_expr=%s", key_expr, exc_info=exc)
                await asyncio.sleep(0.05)

def _put_realtime_drop(session: Any, key: str, payload: bytes) -> None:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh backend requires the `eclipse-zenoh` Python package") from exc
    session.put(
        key,
        bytes(payload),
        congestion_control=zenoh.CongestionControl.DROP,
        priority=zenoh.Priority.REAL_TIME,
        express=True,
    )


def _put_data_fifo(session: Any, key: str, payload: bytes) -> None:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh backend requires the `eclipse-zenoh` Python package") from exc
    session.put(
        key,
        bytes(payload),
        congestion_control=zenoh.CongestionControl.BLOCK_FIRST,
        priority=zenoh.Priority.DATA,
        express=False,
    )


def _put_reliable_control(session: Any, key: str, payload: bytes) -> None:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh backend requires the `eclipse-zenoh` Python package") from exc
    session.put(
        key,
        bytes(payload),
        congestion_control=zenoh.CongestionControl.BLOCK,
        priority=zenoh.Priority.INTERACTIVE_HIGH,
        express=True,
    )


def _declare_queryable(session: Any, key_expr: str, callback: Callable[[Any], None]) -> Any:
    return session.declare_queryable(str(key_expr), callback, complete=True)


def _query_once(session: Any, key: str, payload: bytes, on_reply: Callable[[Any], None], timeout_ms: int) -> None:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh backend requires the `eclipse-zenoh` Python package") from exc
    session.get(
        key,
        on_reply,
        payload=bytes(payload),
        encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
        target=zenoh.QueryTarget.BEST_MATCHING,
        consolidation=zenoh.QueryConsolidation.AUTO,
        timeout=int(timeout_ms),
        congestion_control=zenoh.CongestionControl.BLOCK,
        priority=zenoh.Priority.INTERACTIVE_HIGH,
        express=True,
    )


def _reply_query_ok(query: Any, payload: bytes) -> None:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh backend requires the `eclipse-zenoh` Python package") from exc
    query.reply(
        str(query.key_expr),
        bytes(payload),
        encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
        express=True,
    )


def _reply_query_error(query: Any, payload: bytes) -> None:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh backend requires the `eclipse-zenoh` Python package") from exc
    query.reply_err(bytes(payload), encoding=zenoh.Encoding.TEXT_PLAIN)


def _set_future_result(future: asyncio.Future[bytes], value: bytes) -> None:
    if not future.done():
        future.set_result(value)


def _set_future_exception(future: asyncio.Future[bytes], exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


def _normalize_zenoh_key(key: str) -> str:
    text = str(key or "").strip("/")
    if not text:
        raise ValueError("zenoh key must be non-empty")
    return text


def _normalize_zenoh_key_expr(key_expr: str) -> str:
    text = str(key_expr or "").strip("/")
    if not text:
        raise ValueError("zenoh key expression must be non-empty")
    return text


def _declare_retained_state_publisher(session: Any, key: str) -> Any:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh backend requires the `eclipse-zenoh` Python package") from exc
    return zenoh.ext.declare_advanced_publisher(
        session,
        key,
        congestion_control=zenoh.CongestionControl.BLOCK,
        priority=zenoh.Priority.INTERACTIVE_HIGH,
        express=True,
        reliability=zenoh.Reliability.RELIABLE,
        cache=zenoh.ext.CacheConfig(max_samples=1),
        publisher_detection=True,
    )


def _declare_retained_state_subscriber(session: Any, key_expr: str, callback: Callable[[Any], None]) -> Any:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh backend requires the `eclipse-zenoh` Python package") from exc
    return zenoh.ext.declare_advanced_subscriber(
        session,
        key_expr,
        callback,
        history=zenoh.ext.HistoryConfig(detect_late_publishers=True, max_samples=1),
    )


__all__ = ["StreamDeliveryPolicy", "ZenohTransport", "ZenohTransportConfig"]
