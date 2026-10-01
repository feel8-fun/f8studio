from __future__ import annotations

import logging
import time
from typing import Protocol

from .._runtime_errors import OPERATOR_STATE_PUBLISH_ERRORS

_SCRIPT_MONITOR_PUBLISH_ERRORS = OPERATOR_STATE_PUBLISH_ERRORS
_REPEATING_ERROR_LOG_INTERVAL_MS = 2000


class ScriptMonitorErrorBus(Protocol):
    def report_error(
        self,
        node_id: str,
        code: str,
        message: str,
        *,
        severity: str = "error",
        fingerprint: str | None = None,
        ts_ms: int | None = None,
    ) -> None: ...

    def clear_error(
        self,
        node_id: str,
        fingerprint: str | None = None,
        ts_ms: int | None = None,
    ) -> None: ...


class ScriptErrorReporter:
    def __init__(
        self,
        *,
        node_id: str,
        log_context: str,
        logger: logging.Logger,
        error_code: str,
        fingerprint_prefix: str,
        repeating_log_interval_ms: int = _REPEATING_ERROR_LOG_INTERVAL_MS,
    ) -> None:
        self._node_id = str(node_id)
        self._log_context = str(log_context)
        self._logger = logger
        self._error_code = str(error_code)
        self._fingerprint_prefix = str(fingerprint_prefix)
        self._repeating_log_interval_ms = int(repeating_log_interval_ms)
        self._last_error: str | None = None
        self._error_seq = 0
        self._last_logged_error_fingerprint = ""
        self._last_logged_error_ts_ms = 0
        self._pending_monitor_error_message = ""
        self._pending_monitor_error_fingerprint = ""

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def error_seq(self) -> int:
        return int(self._error_seq)

    def set_error(self, stage: str, exc: BaseException, *, bus: ScriptMonitorErrorBus | None) -> None:
        self._error_seq = int(self._error_seq) + 1
        message = f"{stage}: {exc}"
        self._last_error = message
        fingerprint = self._error_fingerprint(stage, exc)
        if self._should_log_repeating_error(fingerprint, now_ms=self._now_ms()):
            self._logger.error(
                "[%s:%s] error %s",
                self._node_id,
                self._log_context,
                message,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        self._publish_monitor_error(bus=bus, message=message, fingerprint=fingerprint)

    def clear_last_error(self, *, bus: ScriptMonitorErrorBus | None) -> None:
        if not self._last_error:
            return
        self._last_error = None
        self._pending_monitor_error_message = ""
        self._pending_monitor_error_fingerprint = ""
        if bus is None:
            return
        try:
            bus.clear_error(self._node_id)
        except _SCRIPT_MONITOR_PUBLISH_ERRORS as exc:
            self._logger.error(
                "[%s:%s] clear monitor error failed",
                self._node_id,
                self._log_context,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def flush_pending(self, *, bus: ScriptMonitorErrorBus | None) -> None:
        message = str(self._pending_monitor_error_message)
        fingerprint = str(self._pending_monitor_error_fingerprint)
        if not message or not fingerprint:
            return
        self._publish_monitor_error(bus=bus, message=message, fingerprint=fingerprint)
        if bus is not None:
            self._pending_monitor_error_message = ""
            self._pending_monitor_error_fingerprint = ""

    def _publish_monitor_error(
        self,
        *,
        bus: ScriptMonitorErrorBus | None,
        message: str,
        fingerprint: str,
    ) -> None:
        if bus is None:
            self._pending_monitor_error_message = str(message)
            self._pending_monitor_error_fingerprint = str(fingerprint)
            return
        try:
            bus.report_error(
                self._node_id,
                self._error_code,
                str(message),
                severity="error",
                fingerprint=str(fingerprint),
            )
        except _SCRIPT_MONITOR_PUBLISH_ERRORS as exc:
            self._logger.error(
                "[%s:%s] report monitor error failed",
                self._node_id,
                self._log_context,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    def _error_fingerprint(self, stage: str, exc: BaseException) -> str:
        return f"{self._fingerprint_prefix}:{stage}:{type(exc).__name__}:{exc}"

    def _should_log_repeating_error(self, fingerprint: str, *, now_ms: int) -> bool:
        if fingerprint != self._last_logged_error_fingerprint:
            self._last_logged_error_fingerprint = fingerprint
            self._last_logged_error_ts_ms = int(now_ms)
            return True
        elapsed_ms = int(now_ms) - int(self._last_logged_error_ts_ms)
        if elapsed_ms < self._repeating_log_interval_ms:
            return False
        self._last_logged_error_ts_ms = int(now_ms)
        return True

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000.0)
