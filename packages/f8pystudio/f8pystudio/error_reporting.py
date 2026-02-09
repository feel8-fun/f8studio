from __future__ import annotations

import hashlib
import logging
import threading
import traceback
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExceptionFingerprint:
    """
    Stable-ish fingerprint for exception deduping.

    Purpose: suppress repeated logs from high-frequency loops while keeping the
    first occurrence actionable (context + traceback).
    """

    context: str
    exc_type: str
    message: str
    location: str

    def key(self) -> str:
        payload = f"{self.context}\n{self.exc_type}\n{self.message}\n{self.location}"
        return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


class ExceptionLogOnce:
    """
    Thread-safe "log once per unique exception fingerprint" helper.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: set[str] = set()

    def should_log(self, fingerprint: ExceptionFingerprint) -> bool:
        key = fingerprint.key()
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True


def _exc_location(exc: BaseException) -> str:
    tb = exc.__traceback__
    if tb is None:
        return ""
    frames = traceback.extract_tb(tb)
    if not frames:
        return ""
    last = frames[-1]
    return f"{last.filename}:{last.lineno}"


def fingerprint_exception(*, context: str, exc: BaseException) -> ExceptionFingerprint:
    return ExceptionFingerprint(
        context=str(context or "").strip(),
        exc_type=type(exc).__name__,
        message=str(exc),
        location=_exc_location(exc),
    )


def format_exception_lines(*, context: str, exc: BaseException, level: str = "ERROR") -> list[str]:
    ctx = str(context or "").strip()
    lvl = str(level or "ERROR").strip().upper()
    header = f"[{lvl}] {ctx}: {type(exc).__name__}: {exc}".strip()
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    out: list[str] = [header]
    for raw in tb_lines:
        # `format_exception` returns multi-line chunks; split so QPlainTextEdit doesn't append giant blocks.
        for line in raw.rstrip("\n").splitlines():
            out.append(line)
    return out


def report_exception(
    emit_line: Callable[[str], None],
    *,
    context: str,
    exc: BaseException,
    level: str = "ERROR",
    log_once: ExceptionLogOnce | None = None,
) -> None:
    """
    Best-effort: emit exception (context + traceback) into a UI log collector.

    If `log_once` is provided, identical errors are suppressed after first log.
    """
    try:
        if log_once is not None:
            fp = fingerprint_exception(context=context, exc=exc)
            if not log_once.should_log(fp):
                return
        for line in format_exception_lines(context=context, exc=exc, level=level):
            emit_line(line + "\n")
    except Exception:
        logger.exception("Failed to report exception to UI log collector")
