import asyncio
from typing import Any

from f8pyscript.expr_error_reporter import PyExprErrorReporter


def test_pyexpr_error_reporter_tracks_and_clears_current_error() -> None:
    async def _run() -> tuple[list[tuple[str, str, str]], int, str]:
        reports: list[tuple[str, str, str]] = []
        clear_count = 0

        async def _report_error(
            code: str,
            message: str,
            *,
            severity: str = "error",
            fingerprint: str | None = None,
        ) -> Any:
            del severity
            reports.append((code, message, str(fingerprint or "")))

        async def _clear_error() -> None:
            nonlocal clear_count
            clear_count += 1

        reporter = PyExprErrorReporter(report_error=_report_error, clear_error=_clear_error)

        await reporter.clear_error()
        await reporter.set_error("eval: boom")
        last_error = reporter.last_error
        await reporter.clear_error()

        return reports, clear_count, last_error

    reports, clear_count, last_error = asyncio.run(_run())

    assert reports == [("PYEXPR_ERROR", "eval: boom", "pyexpr:eval: boom")]
    assert clear_count == 1
    assert last_error == "eval: boom"


def test_pyexpr_error_reporter_suppresses_repeated_warning_signatures() -> None:
    async def _noop_report_error(
        code: str,
        message: str,
        *,
        severity: str = "error",
        fingerprint: str | None = None,
    ) -> Any:
        del code, message, severity, fingerprint

    async def _noop_clear_error() -> None:
        return None

    reporter = PyExprErrorReporter(report_error=_noop_report_error, clear_error=_noop_clear_error)

    assert reporter.should_log_eval_error("ValueError:bad", now_ms=1000) is True
    assert reporter.should_log_eval_error("ValueError:bad", now_ms=2000) is False
    assert reporter.should_log_eval_error("ValueError:bad", now_ms=6000) is True
    assert reporter.should_log_eval_error("TypeError:bad", now_ms=6100) is True

    assert reporter.should_log_unmatched_output("unmatched:a", now_ms=1000) is True
    assert reporter.should_log_unmatched_output("unmatched:a", now_ms=2000) is False
    assert reporter.should_log_unmatched_output("unmatched:a", now_ms=6000) is True
