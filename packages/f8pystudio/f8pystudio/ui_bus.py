from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class UiCommand:
    node_id: str
    command: str
    payload: dict[str, Any]
    ts_ms: int | None = None


_ui_sink: Callable[[UiCommand], None] | None = None


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
    except Exception:
        return
