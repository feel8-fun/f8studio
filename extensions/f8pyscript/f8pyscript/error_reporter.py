from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Protocol


_MONITOR_REPORT_ERRORS = (LookupError, OSError, RuntimeError, TypeError, ValueError)
_REPEATING_ERROR_LOG_INTERVAL_MS = 2000


class AsyncMonitorErrorReporter(Protocol):
    async def __call__(
        self,
        code: str,
        message: str,
        *,
        severity: str = "error",
        fingerprint: str | None = None,
    ) -> Any: ...


class PyScriptErrorReporter:
    def __init__(
        self,
        *,
        node_id: str,
        logger: logging.Logger,
        report_error: AsyncMonitorErrorReporter,
        log_context: str = "pyscript",
    ) -> None:
        self._node_id = str(node_id)
        self._logger = logger
        self._report_error = report_error
        self._log_context = str(log_context)
        self._last_error: str | None = None
        self._dedupe: dict[str, int] = {}

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def set_error(self, stage: str, exc: BaseException) -> None:
        message = f"{stage}: {exc}"
        self._last_error = message
        self._logger.error(
            "[%s:%s] error %s",
            self._node_id,
            self._log_context,
            message,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self._report_monitor_error(stage, exc, message),
            name=f"{self._log_context}:reportError:{self._node_id}",
        )

    def log_deduped(self, key: str, message: str, exc: BaseException) -> None:
        now_ms = self._now_ms()
        last_ts = int(self._dedupe.get(key) or 0)
        if (now_ms - last_ts) < _REPEATING_ERROR_LOG_INTERVAL_MS:
            return
        self._dedupe[key] = now_ms
        self._logger.error(
            "[%s:%s] %s",
            self._node_id,
            self._log_context,
            message,
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    async def _report_monitor_error(self, stage: str, exc: BaseException, message: str) -> None:
        try:
            await self._report_error(
                "PYSCRIPT_ERROR",
                message,
                severity="error",
                fingerprint=f"pyscript:{stage}:{type(exc).__name__}:{exc}",
            )
        except _MONITOR_REPORT_ERRORS as report_exc:
            self._logger.error(
                "[%s:%s] report monitor error failed",
                self._node_id,
                self._log_context,
                exc_info=(type(report_exc), report_exc, report_exc.__traceback__),
            )

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000.0)


__all__ = ["AsyncMonitorErrorReporter", "PyScriptErrorReporter"]
