import logging

from f8pyengine.operators.script_utils.error_reporter import ScriptErrorReporter


class _FakeMonitorBus:
    def __init__(self) -> None:
        self.reported: list[tuple[str, str, str, str | None]] = []
        self.cleared: list[str] = []

    def report_error(
        self,
        node_id: str,
        code: str,
        message: str,
        *,
        severity: str = "error",
        fingerprint: str | None = None,
        ts_ms: int | None = None,
    ) -> None:
        del severity, ts_ms
        self.reported.append((node_id, code, message, fingerprint))

    def clear_error(
        self,
        node_id: str,
        fingerprint: str | None = None,
        ts_ms: int | None = None,
    ) -> None:
        del fingerprint, ts_ms
        self.cleared.append(node_id)


def _reporter() -> ScriptErrorReporter:
    return ScriptErrorReporter(
        node_id="ps",
        log_context="python_script",
        logger=logging.getLogger("test.python_script_error_reporter"),
        error_code="PYTHON_SCRIPT_ERROR",
        fingerprint_prefix="python-script",
    )


def test_script_error_reporter_defers_until_bus_attached() -> None:
    reporter = _reporter()
    bus = _FakeMonitorBus()

    reporter.set_error("compile", SyntaxError("bad"), bus=None)
    assert reporter.last_error == "compile: bad"
    assert reporter.error_seq == 1
    assert bus.reported == []

    reporter.flush_pending(bus=bus)

    assert len(bus.reported) == 1
    node_id, code, message, fingerprint = bus.reported[0]
    assert node_id == "ps"
    assert code == "PYTHON_SCRIPT_ERROR"
    assert message == "compile: bad"
    assert str(fingerprint or "").startswith("python-script:compile:SyntaxError:")


def test_script_error_reporter_clears_monitor_error() -> None:
    reporter = _reporter()
    bus = _FakeMonitorBus()

    reporter.set_error("onMsg", RuntimeError("boom"), bus=bus)
    reporter.clear_last_error(bus=bus)

    assert reporter.last_error is None
    assert reporter.error_seq == 1
    assert bus.cleared == ["ps"]
