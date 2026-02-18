from __future__ import annotations

from dataclasses import dataclass

from qtpy import QtCore, QtGui, QtWidgets

from f8pystudio.widgets.f8_prop_value_widgets import F8NumberPropLineEdit


@dataclass
class _FakePos:
    x_value: float

    def x(self) -> float:
        return float(self.x_value)


class _FakeMouseEvent:
    def __init__(
        self,
        *,
        global_x: float,
        button: QtCore.Qt.MouseButton = QtCore.Qt.LeftButton,
        modifiers: QtCore.Qt.KeyboardModifiers = QtCore.Qt.NoModifier,
    ) -> None:
        self._global_x = float(global_x)
        self._button = button
        self._modifiers = modifiers

    def globalPosition(self) -> _FakePos:
        return _FakePos(self._global_x)

    def button(self) -> QtCore.Qt.MouseButton:
        return self._button

    def modifiers(self) -> QtCore.Qt.KeyboardModifiers:
        return self._modifiers


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_scrub_emits_preview_and_commit() -> None:
    _ensure_app()
    widget = F8NumberPropLineEdit(data_type=float)
    widget.set_name("gain")
    widget.set_value(1.0)
    widget.set_scrub_base_step(0.1)

    changing: list[object] = []
    changed: list[object] = []
    widget.value_changing.connect(lambda _name, value: changing.append(value))  # type: ignore[attr-defined]
    widget.value_changed.connect(lambda _name, value: changed.append(value))  # type: ignore[attr-defined]

    widget._scrub_begin(_FakeMouseEvent(global_x=100))
    widget._scrub_update(_FakeMouseEvent(global_x=130), commit=False)
    widget._scrub_update(_FakeMouseEvent(global_x=130), commit=True)
    widget._scrub_end()

    assert changing
    assert len(changed) == 1
    assert float(changing[-1]) == 4.0
    assert float(changed[0]) == 4.0


def test_scrub_shift_and_ctrl_multiplier() -> None:
    _ensure_app()
    widget = F8NumberPropLineEdit(data_type=float)
    widget.set_name("speed")
    widget.set_value(10.0)
    widget.set_scrub_base_step(1.0)

    changed: list[object] = []
    widget.value_changed.connect(lambda _name, value: changed.append(value))  # type: ignore[attr-defined]

    widget._scrub_begin(_FakeMouseEvent(global_x=0))
    widget._scrub_update(_FakeMouseEvent(global_x=10, modifiers=QtCore.Qt.ShiftModifier), commit=True)
    widget._scrub_end()
    assert float(changed[-1]) == 11.0

    widget.set_value(10.0)
    widget._scrub_begin(_FakeMouseEvent(global_x=0))
    widget._scrub_update(_FakeMouseEvent(global_x=10, modifiers=QtCore.Qt.ControlModifier), commit=True)
    widget._scrub_end()
    assert float(changed[-1]) == 110.0


def test_scrub_escape_restores_text_without_commit() -> None:
    _ensure_app()
    widget = F8NumberPropLineEdit(data_type=float)
    widget.set_name("offset")
    widget.setText("12.5")
    widget.set_scrub_base_step(0.5)

    changed: list[object] = []
    widget.value_changed.connect(lambda _name, value: changed.append(value))  # type: ignore[attr-defined]

    widget._scrub_begin(_FakeMouseEvent(global_x=0))
    widget._scrub_update(_FakeMouseEvent(global_x=20), commit=False)
    key_event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier)
    widget.keyPressEvent(key_event)

    assert widget.text() == "12.5"
    assert changed == []


def test_scrub_respects_min_max_clamp() -> None:
    _ensure_app()
    widget = F8NumberPropLineEdit(data_type=int)
    widget.set_name("count")
    widget.set_value(8)
    widget.set_min(0)
    widget.set_max(10)
    widget.set_scrub_base_step(1.0)

    changed: list[object] = []
    widget.value_changed.connect(lambda _name, value: changed.append(value))  # type: ignore[attr-defined]

    widget._scrub_begin(_FakeMouseEvent(global_x=0))
    widget._scrub_update(_FakeMouseEvent(global_x=50), commit=True)
    widget._scrub_end()

    assert changed[-1] == 10
