from __future__ import annotations

import logging
import threading
from typing import Protocol

log = logging.getLogger(__name__)

DEFAULT_ZENOH_CLOSE_TIMEOUT_S = 0.5


class ZenohCloseableSession(Protocol):
    def close(self) -> None: ...


_timeout_warning_lock = threading.Lock()
_timeout_warning_contexts: set[str] = set()
_abandoned_session_lock = threading.Lock()
_abandoned_sessions: list[ZenohCloseableSession] = []


def _thread_name(context: str) -> str:
    text = str(context or "").strip() or "session"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in text)
    return f"zenoh-close:{safe[:48]}"


def close_zenoh_session_best_effort(
    session: ZenohCloseableSession,
    *,
    context: str,
    timeout_s: float = DEFAULT_ZENOH_CLOSE_TIMEOUT_S,
    native_close: bool = True,
) -> bool:
    """
    Close a Zenoh session without letting a stuck close block process shutdown.

    Zenoh shutdown can occasionally block in native code. Using asyncio.to_thread()
    for that path leaves a non-daemon ThreadPoolExecutor worker behind, which can
    keep a headless Studio or service process alive after shutdown has begun.
    """

    done = threading.Event()
    context_text = str(context or "").strip()
    errors: list[BaseException] = []

    if not bool(native_close):
        with _abandoned_session_lock:
            _abandoned_sessions.append(session)
        log.debug("zenoh session native close skipped context=%s", context_text)
        return True

    def _close() -> None:
        try:
            session.close()
        except BaseException as exc:
            log.debug("zenoh session close failed context=%s", context_text, exc_info=exc)
            errors.append(exc)
        finally:
            done.set()

    timeout = max(0.0, float(timeout_s))
    thread = threading.Thread(target=_close, name=_thread_name(context), daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if not done.is_set():
        with _timeout_warning_lock:
            should_warn = context_text not in _timeout_warning_contexts
            _timeout_warning_contexts.add(context_text)
        if should_warn:
            log.warning(
                "zenoh session close timed out context=%s timeout=%.3fs; continuing shutdown",
                context_text,
                timeout,
            )
        else:
            log.debug(
                "zenoh session close timed out again context=%s timeout=%.3fs; continuing shutdown",
                context_text,
                timeout,
            )
        return False
    return not errors


__all__ = ["DEFAULT_ZENOH_CLOSE_TIMEOUT_S", "ZenohCloseableSession", "close_zenoh_session_best_effort"]
