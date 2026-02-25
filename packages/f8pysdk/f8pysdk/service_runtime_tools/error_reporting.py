from __future__ import annotations

import hashlib
import threading
import traceback
from dataclasses import dataclass


@dataclass(frozen=True)
class ExceptionFingerprint:
    context: str
    exc_type: str
    message: str
    location: str

    def key(self) -> str:
        payload = f"{self.context}\n{self.exc_type}\n{self.message}\n{self.location}"
        return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


class ExceptionLogOnce:
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
