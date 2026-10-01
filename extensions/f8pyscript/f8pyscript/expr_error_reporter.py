from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


_REPEATING_ERROR_LOG_INTERVAL_MS = 5000


class PyExprErrorReporter:
    def __init__(
        self,
        *,
        report_error: Callable[..., Awaitable[Any]],
        clear_error: Callable[..., Awaitable[Any]],
    ) -> None:
        self._report_error = report_error
        self._clear_error = clear_error
        self._last_error = ""
        self._eval_error_sig = ""
        self._eval_error_ts_ms = 0
        self._unmatched_output_sig = ""
        self._unmatched_output_ts_ms = 0

    @property
    def last_error(self) -> str:
        return self._last_error

    async def set_error(self, message: str) -> None:
        self._last_error = str(message)
        await self._report_error(
            "PYEXPR_ERROR",
            self._last_error,
            severity="error",
            fingerprint=f"pyexpr:{self._last_error}",
        )

    async def clear_error(self) -> None:
        if not self._last_error:
            return
        self._last_error = ""
        await self._clear_error()

    def should_log_eval_error(self, sig: str, *, now_ms: int) -> bool:
        should_log, self._eval_error_sig, self._eval_error_ts_ms = self._should_log_repeating_message(
            sig=sig,
            previous_sig=self._eval_error_sig,
            previous_ts_ms=self._eval_error_ts_ms,
            now_ms=now_ms,
        )
        return should_log

    def should_log_unmatched_output(self, sig: str, *, now_ms: int) -> bool:
        (
            should_log,
            self._unmatched_output_sig,
            self._unmatched_output_ts_ms,
        ) = self._should_log_repeating_message(
            sig=sig,
            previous_sig=self._unmatched_output_sig,
            previous_ts_ms=self._unmatched_output_ts_ms,
            now_ms=now_ms,
        )
        return should_log

    @staticmethod
    def _should_log_repeating_message(
        *,
        sig: str,
        previous_sig: str,
        previous_ts_ms: int,
        now_ms: int,
    ) -> tuple[bool, str, int]:
        current_sig = str(sig)
        current_ts_ms = int(now_ms)
        if current_sig != previous_sig:
            return True, current_sig, current_ts_ms
        if (current_ts_ms - int(previous_ts_ms)) >= _REPEATING_ERROR_LOG_INTERVAL_MS:
            return True, current_sig, current_ts_ms
        return False, previous_sig, previous_ts_ms


__all__ = ["PyExprErrorReporter"]
