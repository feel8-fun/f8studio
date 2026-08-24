from __future__ import annotations

import math
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from f8pystudio.ui.components.controls import F8Dial, F8OptionCombo, F8RangeBar
from f8pystudio.ui.support.state_builders import StateControlSpec, build_inline_control_binding
from f8pystudio.ui.components.state_editors import F8IncrementButtonEditor, F8WrapLineEditor


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is not None:
        return app
    return QtWidgets.QApplication([])


def _build_inline_binding(
    *,
    spec: StateControlSpec,
    state: dict[str, Any],
    calls: list[tuple[Any, bool]],
) -> Any:
    return build_inline_control_binding(
        spec=spec,
        read_only=False,
        value_getter=lambda: state.get("value"),
        value_setter=lambda value, push_undo: _record_value(state, calls, value, push_undo),
        property_value_getter=lambda field_name: state.get(str(field_name)),
        pool_resolver=lambda field_name: list(state.get(f"pool:{field_name}", [])),
        code_title="Node - Value",
        code_value_getter=None,
        code_value_setter=None,
        code_target_exists_provider=None,
        assist_context=None,
        assist_context_provider=None,
        editor_session_key=None,
        style_applier=lambda widget: None,
        text_palette_applier=lambda widget: None,
        tooltip_filter_installer=None,
    )


def _record_value(state: dict[str, Any], calls: list[tuple[Any, bool]], value: Any, push_undo: bool) -> None:
    state["value"] = value
    calls.append((value, push_undo))


def _mouse_event(
    event_type: QtCore.QEvent.Type,
    pos: QtCore.QPointF,
    *,
    button: QtCore.Qt.MouseButton,
    buttons: QtCore.Qt.MouseButton,
) -> QtGui.QMouseEvent:
    return QtGui.QMouseEvent(
        event_type,
        pos,
        pos,
        pos,
        button,
        buttons,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def _dial_pos(widget: QtWidgets.QWidget, fraction: float) -> QtCore.QPointF:
    rect = QtCore.QRectF(widget.rect()).adjusted(2.0, 2.0, -2.0, -2.0)
    center = rect.center()
    radius = min(rect.width(), rect.height()) / 2.0
    theta = (float(fraction) * 2.0 * math.pi) - (math.pi / 2.0)
    return QtCore.QPointF(center.x() + math.cos(theta) * radius, center.y() + math.sin(theta) * radius)


def test_wrapline_builder_commits_normalized_text() -> None:
    _ensure_app()
    state = {"value": "before"}
    calls: list[tuple[Any, bool]] = []
    binding = _build_inline_binding(
        spec=StateControlSpec(
            name="expr",
            label="Expr",
            ui_control="wrapline",
            ui_language="plaintext",
            schema_type="string",
            enum_items=[],
            minimum=None,
            maximum=None,
        ),
        state=state,
        calls=calls,
    )

    widget = binding.widget
    assert isinstance(widget, F8WrapLineEditor)
    widget.focusInEvent(QtGui.QFocusEvent(QtCore.QEvent.Type.FocusIn))
    widget.setPlainText("hello\n world")
    widget.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Return, QtCore.Qt.NoModifier))

    assert calls == [("hello world", True)]
    assert widget.toPlainText() == "hello world"


def test_select_builder_refresh_options_preserves_selected_value() -> None:
    _ensure_app()
    state = {"value": "b", "pool:choices": ["a", "b"]}
    calls: list[tuple[Any, bool]] = []
    binding = _build_inline_binding(
        spec=StateControlSpec(
            name="choice",
            label="Choice",
            ui_control="select",
            ui_language="plaintext",
            schema_type="string",
            enum_items=[],
            minimum=None,
            maximum=None,
            select_pool_field="choices",
        ),
        state=state,
        calls=calls,
    )

    widget = binding.widget
    assert isinstance(widget, F8OptionCombo)
    assert widget.count() == 2
    assert widget.value() == "b"
    assert widget._popup is None

    state["pool:choices"] = ["c", "b", "d"]
    assert binding.refresh_options is not None
    binding.refresh_options()

    assert widget.count() == 3
    assert widget.value() == "b"
    assert widget._popup is None


def test_range_slider_builder_edits_min_and_max_in_one_control() -> None:
    _ensure_app()
    state = {"value": [0.2, 0.8]}
    calls: list[tuple[Any, bool]] = []
    binding = _build_inline_binding(
        spec=StateControlSpec(
            name="outputRange",
            label="Output Range",
            ui_control="range_slider",
            ui_language="plaintext",
            schema_type="array",
            enum_items=[],
            minimum=0.0,
            maximum=1.0,
        ),
        state=state,
        calls=calls,
    )

    widget = binding.widget
    assert isinstance(widget, F8RangeBar)
    assert widget.layout().count() == 2
    assert widget.value() == [0.2, 0.8]

    widget.lower_bar().valueCommitted.emit(0.35)
    assert calls[-1] == ([0.35, 0.8], True)

    widget.upper_bar().valueChanging.emit(0.1)
    assert calls[-1] == ([0.35, 0.35], False)


def test_button_builder_marks_invalid_numeric_schema() -> None:
    _ensure_app()
    state = {"value": 0}
    calls: list[tuple[Any, bool]] = []
    binding = _build_inline_binding(
        spec=StateControlSpec(
            name="trigger",
            label="Trigger",
            ui_control="button",
            ui_language="plaintext",
            schema_type="string",
            enum_items=[],
            minimum=None,
            maximum=None,
        ),
        state=state,
        calls=calls,
    )

    widget = binding.widget
    assert isinstance(widget, F8IncrementButtonEditor)
    assert not widget.isEnabled()
    widget.click()
    assert calls == []


def test_dial_builder_wraps_across_seam_and_preserves_commit_semantics() -> None:
    _ensure_app()
    state = {"value": 0.0}
    calls: list[tuple[Any, bool]] = []
    binding = _build_inline_binding(
        spec=StateControlSpec(
            name="pan",
            label="Pan",
            ui_control="dial",
            ui_language="plaintext",
            schema_type="number",
            enum_items=[],
            minimum=-1.0,
            maximum=1.0,
        ),
        state=state,
        calls=calls,
    )

    widget = binding.widget
    assert isinstance(widget, F8Dial)
    widget.resize(96, 96)

    near_max = _dial_pos(widget, 0.99)
    near_min = _dial_pos(widget, 0.01)
    widget.mousePressEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            near_max,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.LeftButton,
        )
    )
    widget.mouseMoveEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseMove,
            near_min,
            button=QtCore.Qt.MouseButton.NoButton,
            buttons=QtCore.Qt.MouseButton.LeftButton,
        )
    )
    widget.mouseReleaseEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            near_min,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.NoButton,
        )
    )

    assert len(calls) == 3
    assert calls[0][1] is False
    assert calls[1][1] is False
    assert calls[2][1] is True
    assert float(calls[0][0]) > 0.9
    assert float(calls[1][0]) < -0.9
    assert float(state["value"]) < -0.9


def test_dial_builder_marks_invalid_non_numeric_schema() -> None:
    _ensure_app()
    state = {"value": "bad"}
    calls: list[tuple[Any, bool]] = []
    binding = _build_inline_binding(
        spec=StateControlSpec(
            name="badDial",
            label="Bad Dial",
            ui_control="dial",
            ui_language="plaintext",
            schema_type="string",
            enum_items=[],
            minimum=None,
            maximum=None,
        ),
        state=state,
        calls=calls,
    )

    widget = binding.widget
    assert isinstance(widget, F8Dial)
    assert not widget.isEnabled()
    assert "integer or number" in str(widget.toolTip() or "")


def test_dial_noloop_builder_clamps_at_max_without_wraparound() -> None:
    _ensure_app()
    state = {"value": 0.0}
    calls: list[tuple[Any, bool]] = []
    binding = _build_inline_binding(
        spec=StateControlSpec(
            name="pan",
            label="Pan",
            ui_control="dial[noloop]",
            ui_language="plaintext",
            schema_type="number",
            enum_items=[],
            minimum=-1.0,
            maximum=1.0,
        ),
        state=state,
        calls=calls,
    )

    widget = binding.widget
    assert isinstance(widget, F8Dial)
    assert widget.loop() is False
    widget.resize(96, 96)

    near_max = _dial_pos(widget, 0.99)
    near_min = _dial_pos(widget, 0.01)
    widget.mousePressEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            near_max,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.LeftButton,
        )
    )
    widget.mouseMoveEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseMove,
            near_min,
            button=QtCore.Qt.MouseButton.NoButton,
            buttons=QtCore.Qt.MouseButton.LeftButton,
        )
    )
    widget.mouseReleaseEvent(
        _mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            near_min,
            button=QtCore.Qt.MouseButton.LeftButton,
            buttons=QtCore.Qt.MouseButton.NoButton,
        )
    )

    assert float(state["value"]) > 0.98
