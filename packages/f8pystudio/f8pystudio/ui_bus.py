from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from .error_reporting import ExceptionLogOnce, fingerprint_exception

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UiCommand:
    node_id: str
    command: str
    payload: dict[str, Any]
    ts_ms: int | None = None


_ui_sink: Callable[[UiCommand], None] | None = None
_exception_log_once = ExceptionLogOnce()


@runtime_checkable
class UiCommandApplier(Protocol):
    def apply_ui_command(self, cmd: UiCommand) -> None: ...


def set_ui_command_sink(sink: Callable[[UiCommand], None] | None) -> None:
    """
    Set an in-process sink for UI commands (node_id, command, payload, ts_ms).
    """
    global _ui_sink
    _ui_sink = sink


def emit_ui_command(node_id: str, command: str, payload: dict[str, Any], *, ts_ms: int | None = None) -> None:
    sink = _ui_sink
    if sink is None:
        return
    try:
        sink(UiCommand(node_id=str(node_id), command=str(command), payload=dict(payload), ts_ms=ts_ms))
    except Exception as exc:
        fp = fingerprint_exception(context="ui_bus.emit_ui_command", exc=exc)
        if _exception_log_once.should_log(fp):
            try:
                logger.error("UI command sink raised", exc_info=exc)
            except (AttributeError, RuntimeError, TypeError):
                pass
        return
