from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from .zenoh_config import apply_zenoh_shared_memory_config
from .zenoh_shutdown import close_zenoh_session_best_effort

log = logging.getLogger(__name__)
_SUBSCRIPTION_SETTLE_S = 0.01


@dataclass
class LatestBinarySample:
    payload: memoryview
    _released: bool = field(default=False, init=False, repr=False)

    def payload_copy(self) -> bytes:
        if self._released:
            raise RuntimeError("binary sample payload has been released")
        return bytes(self.payload)

    def release(self) -> None:
        if self._released:
            return
        self.payload.release()
        self._released = True

    def __enter__(self) -> "LatestBinarySample":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


class ZenohLatestBinaryStreamTransport:
    def __init__(
        self,
        *,
        key_expr: str,
        session: Any,
        subscriber: Any | None = None,
        publisher: Any | None = None,
        log_context: str = "stream",
        min_sample_interval_ms: int = 0,
        max_pending_samples: int = 1,
    ) -> None:
        key = str(key_expr or "").strip()
        if not key:
            raise ValueError("key_expr must be non-empty")
        self.key_expr = key
        self._session = session
        self._subscriber = subscriber
        self._publisher = publisher
        self._log_context = str(log_context or "stream")
        self._closed = False
        self._cv = threading.Condition()
        if max_pending_samples < 1:
            raise ValueError("max_pending_samples must be positive")
        self._pending_raw: deque[bytes] = deque(maxlen=max_pending_samples)
        self._min_sample_interval_s = max(0.0, float(int(min_sample_interval_ms)) / 1000.0)
        self._last_accepted_sample_s = 0.0

    def set_min_sample_interval_ms(self, min_sample_interval_ms: int) -> None:
        next_interval_s = max(0.0, float(int(min_sample_interval_ms)) / 1000.0)
        with self._cv:
            self._min_sample_interval_s = next_interval_s
            self._last_accepted_sample_s = 0.0

    @classmethod
    def open_publisher(
        cls,
        key_expr: str,
        *,
        config_path: str | None = None,
        connect: tuple[str, ...] = (),
        listen: tuple[str, ...] = (),
        shm_pool_bytes: int = 256 * 1024 * 1024,
        log_context: str = "stream",
    ) -> "ZenohLatestBinaryStreamTransport":
        session = _open_zenoh_stream_session(
            config_path=config_path,
            connect=connect,
            listen=listen,
            shm_pool_bytes=shm_pool_bytes,
            log_context=log_context,
        )
        publisher = _declare_zenoh_latest_publisher(session, key_expr)
        time.sleep(_SUBSCRIPTION_SETTLE_S)
        return cls(key_expr=key_expr, session=session, publisher=publisher, log_context=log_context)

    @classmethod
    def open_subscriber(
        cls,
        key_expr: str,
        *,
        config_path: str | None = None,
        connect: tuple[str, ...] = (),
        listen: tuple[str, ...] = (),
        shm_pool_bytes: int = 256 * 1024 * 1024,
        log_context: str = "stream",
        min_sample_interval_ms: int = 0,
        max_pending_samples: int = 1,
    ) -> "ZenohLatestBinaryStreamTransport":
        session = _open_zenoh_stream_session(
            config_path=config_path,
            connect=connect,
            listen=listen,
            shm_pool_bytes=shm_pool_bytes,
            log_context=log_context,
        )
        transport = cls(
            key_expr=key_expr,
            session=session,
            log_context=log_context,
            min_sample_interval_ms=int(min_sample_interval_ms),
            max_pending_samples=max_pending_samples,
        )

        def _on_sample(sample: Any) -> None:
            transport._on_sample(sample)

        transport._subscriber = session.declare_subscriber(str(key_expr), _on_sample)
        time.sleep(_SUBSCRIPTION_SETTLE_S)
        return transport

    @classmethod
    def open_pubsub(
        cls,
        key_expr: str,
        *,
        config_path: str | None = None,
        connect: tuple[str, ...] = (),
        listen: tuple[str, ...] = (),
        shm_pool_bytes: int = 256 * 1024 * 1024,
        log_context: str = "stream",
        min_sample_interval_ms: int = 0,
    ) -> "ZenohLatestBinaryStreamTransport":
        session = _open_zenoh_stream_session(
            config_path=config_path,
            connect=connect,
            listen=listen,
            shm_pool_bytes=shm_pool_bytes,
            log_context=log_context,
        )
        transport = cls(
            key_expr=key_expr,
            session=session,
            log_context=log_context,
            min_sample_interval_ms=int(min_sample_interval_ms),
        )

        def _on_sample(sample: Any) -> None:
            transport._on_sample(sample)

        transport._subscriber = session.declare_subscriber(str(key_expr), _on_sample)
        transport._publisher = _declare_zenoh_latest_publisher(session, key_expr)
        time.sleep(_SUBSCRIPTION_SETTLE_S)
        return transport

    def close(self) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._pending_raw.clear()
            self._cv.notify_all()
        publisher = self._publisher
        self._publisher = None
        if publisher is not None:
            try:
                publisher.undeclare()
            except (RuntimeError, OSError) as exc:
                log.debug("zenoh stream publisher undeclare failed key=%s", self.key_expr, exc_info=exc)
        subscriber = self._subscriber
        self._subscriber = None
        if subscriber is not None:
            try:
                subscriber.undeclare()
            except (RuntimeError, OSError) as exc:
                log.debug("zenoh stream subscriber undeclare failed key=%s", self.key_expr, exc_info=exc)
        session = self._session
        self._session = None
        if session is not None:
            close_zenoh_session_best_effort(
                session,
                context=f"{self._log_context}:{self.key_expr}",
                native_close=False,
            )

    def publish_raw(self, payload: bytes | bytearray | memoryview) -> None:
        session = self._session
        if self._closed or session is None:
            raise RuntimeError("zenoh stream transport is closed")
        payload_view = memoryview(payload).cast("B")
        try:
            raw = bytes(payload_view)
        finally:
            payload_view.release()
        try:
            import zenoh  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Zenoh stream transport requires the `eclipse-zenoh` Python package") from exc
        publisher = self._publisher
        if publisher is not None:
            publisher.put(raw, encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM)
            return
        session.put(
            self.key_expr,
            raw,
            encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
            congestion_control=zenoh.CongestionControl.DROP,
            priority=zenoh.Priority.REAL_TIME,
            express=True,
        )

    def poll_latest_raw(self) -> bytes | None:
        with self._cv:
            if not self._pending_raw:
                return None
            return self._pending_raw.popleft()

    def wait_latest_raw(self, timeout_ms: int) -> bytes | None:
        raw = self.poll_latest_raw()
        if raw is not None:
            return raw
        timeout_s = max(0.0, float(int(timeout_ms)) / 1000.0)
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while not self._closed:
                if self._pending_raw:
                    return self._pending_raw.popleft()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining)
        return None

    def _on_sample(self, sample: Any) -> None:
        now_s = time.monotonic()
        with self._cv:
            if self._closed:
                return
            if self._min_sample_interval_s > 0.0:
                elapsed_s = now_s - float(self._last_accepted_sample_s)
                if elapsed_s < self._min_sample_interval_s:
                    return
                self._last_accepted_sample_s = now_s
        try:
            raw = bytes(sample.payload)
        except (TypeError, ValueError) as exc:
            log.debug("zenoh stream sample decode failed key=%s", self.key_expr, exc_info=exc)
            return
        with self._cv:
            if self._closed:
                return
            self._pending_raw.append(raw)
            self._cv.notify_all()


def _open_zenoh_stream_session(
    *,
    config_path: str | None,
    connect: tuple[str, ...],
    listen: tuple[str, ...],
    shm_pool_bytes: int,
    log_context: str,
) -> Any:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh stream transport requires the `eclipse-zenoh` Python package") from exc
    if config_path:
        config = zenoh.Config.from_file(str(config_path))
    else:
        config = zenoh.Config()
    connect_items = tuple(str(item).strip() for item in connect if str(item).strip())
    listen_items = tuple(str(item).strip() for item in listen if str(item).strip())
    if connect_items:
        config.insert_json5("connect/endpoints", json.dumps(list(connect_items)))
    if listen_items:
        config.insert_json5("listen/endpoints", json.dumps(list(listen_items)))
    apply_zenoh_shared_memory_config(
        config,
        zenoh_module=zenoh,
        shm_pool_bytes=int(shm_pool_bytes),
        log_context=str(log_context or "stream"),
    )
    return zenoh.open(config)


def _declare_zenoh_latest_publisher(session: Any, key_expr: str) -> Any:
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Zenoh stream transport requires the `eclipse-zenoh` Python package") from exc
    return session.declare_publisher(
        str(key_expr),
        encoding=zenoh.Encoding.APPLICATION_OCTET_STREAM,
        congestion_control=zenoh.CongestionControl.DROP,
        priority=zenoh.Priority.REAL_TIME,
        express=True,
        reliability=zenoh.Reliability.BEST_EFFORT,
    )


__all__ = [
    "LatestBinarySample",
    "ZenohLatestBinaryStreamTransport",
    "_declare_zenoh_latest_publisher",
    "_open_zenoh_stream_session",
]
