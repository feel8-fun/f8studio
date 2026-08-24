from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from qtpy import QtCore, QtWidgets

from ...editor_assist.session import EditorSessionKey
from ...editor_assist.workspace import EditorAssistContext
from ..components.controls import (
    F8Dial,
    F8ImageB64Editor,
    F8MultiSelect,
    F8OptionCombo,
    F8RangeBar,
    F8Switch,
    F8ValueBar,
)
from ...ui.support.ui_control import parse_ui_control
from ..components.state_editors import (
    F8BoolSwitchEditor,
    F8CodeButtonEditor,
    F8DialEditor,
    F8ImageValueEditor,
    F8IncrementButtonEditor,
    F8MultiSelectEditor,
    F8NumberLineEditor,
    F8OptionComboEditor,
    F8RangeBarEditor,
    F8ValueBarEditor,
    F8WrapLineEditor,
)
from ..components.wave import (
    make_wave_heatmap_control,
    make_wave_pattern_editor_control,
    make_wave_preview_control,
)
from .studio_theme import inline_action_button_qss, studio_dark_theme


@dataclass(frozen=True)
class StateControlSpec:
    name: str
    label: str
    ui_control: str
    ui_language: str
    schema_type: str
    enum_items: list[str]
    minimum: float | None
    maximum: float | None
    field_tooltip: str = ""
    select_pool_field: str | None = None
    multiselect_pool_field: str | None = None
    is_image_b64: bool = False
    range_integer: bool = False


@dataclass(frozen=True)
class StateControlBinding:
    widget: QtWidgets.QWidget
    apply_value: Callable[[Any], None]
    set_read_only: Callable[[bool], None]
    refresh_options: Callable[[], None] | None = None


def set_control_read_only(widget: QtWidgets.QWidget, *, read_only: bool) -> None:
    if isinstance(widget, F8OptionComboEditor):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8MultiSelectEditor):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8BoolSwitchEditor):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8CodeButtonEditor):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8IncrementButtonEditor):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8OptionCombo):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8MultiSelect):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8ImageValueEditor):
        widget.setEnabled(not bool(read_only))
        return
    if isinstance(widget, F8Switch):
        widget.setEnabled(not bool(read_only))
        return
    if isinstance(widget, F8ValueBarEditor):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8RangeBarEditor):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8ValueBar):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8RangeBar):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8Dial):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8DialEditor):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8ImageB64Editor):
        widget.set_disabled(bool(read_only))
        return

    if isinstance(widget, QtWidgets.QLineEdit):
        widget.setEnabled(True)
        widget.setReadOnly(bool(read_only))
        return
    if isinstance(widget, QtWidgets.QPlainTextEdit):
        widget.setEnabled(True)
        widget.setReadOnly(bool(read_only))
        return
    if isinstance(widget, QtWidgets.QTextEdit):
        widget.setEnabled(True)
        widget.setReadOnly(bool(read_only))
        if read_only:
            widget.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
        else:
            widget.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
        return
    if isinstance(widget, QtWidgets.QAbstractSpinBox):
        widget.setEnabled(True)
        widget.setReadOnly(bool(read_only))
        if read_only:
            widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        else:
            widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        return

    widget.setEnabled(not bool(read_only))


def _binding(
    widget: QtWidgets.QWidget,
    *,
    apply_value: Callable[[Any], None],
    refresh_options: Callable[[], None] | None = None,
) -> StateControlBinding:
    return StateControlBinding(
        widget=widget,
        apply_value=apply_value,
        set_read_only=lambda read_only: set_control_read_only(widget, read_only=read_only),
        refresh_options=refresh_options,
    )


def _dial_loop_mode(ui_control: str) -> bool | None:
    return parse_ui_control(ui_control).dial_loop


def build_panel_control_binding(
    *,
    spec: StateControlSpec,
    fallback_widget: Any,
    pool_resolver: Callable[[str], list[str]],
    editor_title: str,
    assist_context: EditorAssistContext | None,
    assist_context_provider: Callable[[], EditorAssistContext | None] | None,
    editor_session_key: EditorSessionKey | None,
) -> StateControlBinding:
    parsed_ui = parse_ui_control(spec.ui_control)
    ui = parsed_ui.control_name
    dial_loop = parsed_ui.dial_loop

    if spec.is_image_b64:
        widget = F8ImageValueEditor()
        widget.set_name(spec.name)
        return _binding(widget, apply_value=widget.set_value)

    if ui == "code":
        widget = F8CodeButtonEditor(title=editor_title, language=spec.ui_language or "plaintext")
        widget.set_name(spec.name)
        widget.set_editor_assist_context(assist_context)
        widget.set_editor_assist_context_provider(assist_context_provider)
        widget.set_editor_session_key(editor_session_key)
        return _binding(widget, apply_value=widget.set_value)

    if ui == "wrapline":
        widget = F8WrapLineEditor(language=spec.ui_language or "plaintext")
        widget.set_name(spec.name)
        return _binding(widget, apply_value=widget.set_value)

    if ui == "button":
        data_type = int if spec.schema_type == "integer" else float
        widget = F8IncrementButtonEditor(title=spec.label, data_type=data_type)
        widget.set_name(spec.name)
        if spec.schema_type not in {"integer", "number"}:
            widget.set_invalid_reason("Button control requires integer or number state schema.")
        return _binding(widget, apply_value=widget.set_value)

    if spec.multiselect_pool_field or ui in {"multiselect", "multi_select", "multi-select"}:
        widget = F8MultiSelectEditor()
        widget.set_name(spec.name)
        if spec.multiselect_pool_field:
            widget.set_pool(spec.multiselect_pool_field, pool_resolver)
            refresh_options = widget.refresh_options
        else:
            widget.set_items(spec.enum_items)
            refresh_options = None
        return _binding(widget, apply_value=widget.set_value, refresh_options=refresh_options)

    if spec.enum_items or spec.select_pool_field or ui in {"select", "dropdown", "dropbox", "combo", "combobox"}:
        widget = F8OptionComboEditor()
        widget.set_name(spec.name)
        if spec.select_pool_field:
            widget.set_pool(spec.select_pool_field, pool_resolver)
            refresh_options = widget.refresh_options
        else:
            widget.set_items(spec.enum_items)
            refresh_options = None
        return _binding(widget, apply_value=widget.set_value, refresh_options=refresh_options)

    if spec.schema_type == "boolean" or ui in {"switch", "toggle"}:
        widget = F8BoolSwitchEditor()
        widget.set_name(spec.name)
        return _binding(widget, apply_value=widget.set_value)

    if dial_loop is not None:
        data_type = int if spec.schema_type == "integer" else float
        widget = F8DialEditor(data_type=data_type)
        widget.set_name(spec.name)
        widget.set_loop(dial_loop)
        if spec.minimum is not None:
            widget.set_min(spec.minimum)
        if spec.maximum is not None:
            widget.set_max(spec.maximum)
        if spec.field_tooltip:
            widget.set_context_tooltip(spec.field_tooltip)
        if spec.schema_type not in {"integer", "number"}:
            widget.set_invalid_reason("Dial control requires integer or number state schema.")
        return _binding(widget, apply_value=widget.set_value)

    if spec.schema_type in {"integer", "number"} and ui == "slider":
        widget = F8ValueBarEditor(data_type=int if spec.schema_type == "integer" else float)
        widget.set_name(spec.name)
        if spec.minimum is not None:
            widget.set_min(spec.minimum)
        if spec.maximum is not None:
            widget.set_max(spec.maximum)
        return _binding(widget, apply_value=widget.set_value)

    if ui == "range_slider":
        widget = F8RangeBarEditor(data_type=int if spec.range_integer else float)
        widget.set_name(spec.name)
        if spec.minimum is not None:
            widget.set_min(spec.minimum)
        if spec.maximum is not None:
            widget.set_max(spec.maximum)
        if spec.schema_type != "array":
            widget.setEnabled(False)
            widget.setToolTip("Range slider requires an array state schema with two numeric values.")
        return _binding(widget, apply_value=widget.set_value)

    if spec.schema_type in {"integer", "number"}:
        widget = F8NumberLineEditor(data_type=int if spec.schema_type == "integer" else float)
        widget.set_name(spec.name)
        if spec.minimum is not None:
            widget.set_min(spec.minimum)
        if spec.maximum is not None:
            widget.set_max(spec.maximum)
        return _binding(widget, apply_value=widget.set_value)

    fallback_widget.set_name(spec.name)
    return _binding(fallback_widget, apply_value=fallback_widget.set_value)


def build_inline_control_binding(
    *,
    spec: StateControlSpec,
    read_only: bool,
    widget_parent: QtWidgets.QWidget | None = None,
    value_getter: Callable[[], Any],
    value_setter: Callable[[Any, bool], None],
    property_value_getter: Callable[[str], Any],
    pool_resolver: Callable[[str], list[str]],
    code_title: str,
    code_value_getter: Callable[[], str] | None,
    code_value_setter: Callable[[str], bool | None] | None,
    code_target_exists_provider: Callable[[], bool] | None,
    assist_context: EditorAssistContext | None,
    assist_context_provider: Callable[[], EditorAssistContext | None] | None,
    editor_session_key: EditorSessionKey | None,
    style_applier: Callable[[QtWidgets.QWidget], None] | None,
    text_palette_applier: Callable[[QtWidgets.QWidget], None] | None,
    tooltip_filter_installer: Callable[[QtWidgets.QWidget], None] | None,
) -> StateControlBinding:
    parsed_ui = parse_ui_control(spec.ui_control)
    ui = parsed_ui.control_name
    dial_loop = parsed_ui.dial_loop

    if ui == "wave_preview":
        control, apply_value = make_wave_preview_control(
            field_tooltip=spec.field_tooltip,
            widget_parent=widget_parent,
            preview_value_getter=value_getter,
            property_value_getter=property_value_getter,
        )
        return _binding(control, apply_value=apply_value)

    if ui == "wave_heatmap":
        control, apply_value = make_wave_heatmap_control(
            field_tooltip=spec.field_tooltip,
            widget_parent=widget_parent,
            heatmap_value_getter=value_getter,
        )
        return _binding(control, apply_value=apply_value)

    if ui == "wave_pattern_editor":
        control, apply_value = make_wave_pattern_editor_control(
            field_tooltip=spec.field_tooltip,
            widget_parent=widget_parent,
            points_value_getter=value_getter,
            property_value_getter=property_value_getter,
            points_setter=lambda value, push_undo: value_setter(value, push_undo=push_undo),
        )
        control.set_read_only(bool(read_only))
        return _binding(control, apply_value=apply_value, refresh_options=None)

    if ui == "wrapline":
        widget = F8WrapLineEditor(widget_parent, language=spec.ui_language or "plaintext")
        widget.set_name(spec.name)
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        _apply_inline_text_widget_style(
            widget,
            spec.field_tooltip,
            style_applier=style_applier,
            text_palette_applier=text_palette_applier,
            tooltip_filter_installer=tooltip_filter_installer,
        )
        widget.value_changed.connect(lambda _field_name, value: value_setter(value, push_undo=True))  # type: ignore[attr-defined]
        widget.set_value(value_getter())
        widget.setReadOnly(bool(read_only))
        return _binding(widget, apply_value=widget.set_value)

    if ui == "code":
        widget = F8CodeButtonEditor(widget_parent, title=code_title, language=spec.ui_language or "plaintext")
        widget.set_name(spec.name)
        widget.set_editor_assist_context(assist_context)
        widget.set_editor_assist_context_provider(assist_context_provider)
        widget.set_editor_session_key(editor_session_key)
        widget.set_persisted_value_getter(code_value_getter)
        widget.set_persisted_value_setter(code_value_setter)
        widget.set_persisted_target_exists_provider(code_target_exists_provider)
        if spec.field_tooltip:
            widget.setToolTip(spec.field_tooltip)
            if tooltip_filter_installer is not None:
                tooltip_filter_installer(widget)
        widget.setStyleSheet(inline_action_button_qss(accent_color=studio_dark_theme().palette.accent))

        def _apply_code_value(value: Any) -> None:
            text = "" if value is None else str(value)
            widget.set_value(text)
            lines = len(text.splitlines()) if text else 0
            if lines > 0:
                lines_text = f"{lines} line" if lines == 1 else f"{lines} lines"
                tooltip = spec.field_tooltip
                widget.setToolTip(f"{tooltip}\n{lines_text}" if tooltip else lines_text)

        _apply_code_value(value_getter())
        widget.set_read_only(bool(read_only))
        return _binding(widget, apply_value=_apply_code_value)

    if ui == "button":
        button_data_type: type[int] | type[float] = int if spec.schema_type == "integer" else float
        widget = F8IncrementButtonEditor(widget_parent, title=spec.label, data_type=button_data_type)
        widget.set_name(spec.name)
        widget.set_button_text(spec.label)
        widget.setStyleSheet(inline_action_button_qss(accent_color=studio_dark_theme().palette.success))
        if spec.field_tooltip:
            widget.set_context_tooltip(spec.field_tooltip)
            if tooltip_filter_installer is not None:
                tooltip_filter_installer(widget)
        if spec.schema_type not in {"integer", "number"}:
            widget.set_invalid_reason("Button control requires integer or number state schema.")
        widget.value_changed.connect(lambda _field_name, value: value_setter(value, push_undo=True))  # type: ignore[attr-defined]
        widget.set_value(value_getter())
        widget.set_read_only(bool(read_only))
        return _binding(widget, apply_value=widget.set_value)

    if spec.is_image_b64:
        widget = F8ImageB64Editor(widget_parent)
        widget.valueChanged.connect(lambda value: value_setter(str(value or ""), push_undo=True))  # type: ignore[attr-defined]

        def _apply_image_value(value: Any) -> None:
            widget.set_value("" if value is None else str(value))

        _apply_image_value(value_getter())
        widget.set_disabled(bool(read_only))
        return _binding(widget, apply_value=_apply_image_value)

    if spec.multiselect_pool_field or ui in {"multiselect", "multi_select", "multi-select"}:
        widget = F8MultiSelect(widget_parent)
        if spec.field_tooltip:
            widget.set_context_tooltip(spec.field_tooltip)
        items = _resolve_inline_items(spec.multiselect_pool_field, spec.enum_items, pool_resolver)
        widget.set_options(items, labels=items)
        widget.valueChanged.connect(lambda value: value_setter(list(value or []), push_undo=True))  # type: ignore[attr-defined]

        def _apply_multi_value(value: Any) -> None:
            widget.set_value(value)

        def _refresh_multi_options() -> None:
            refreshed_items = _resolve_inline_items(spec.multiselect_pool_field, spec.enum_items, pool_resolver)
            widget.set_options(refreshed_items, labels=refreshed_items)
            widget.set_value(value_getter())

        _apply_multi_value(value_getter())
        widget.set_read_only(bool(read_only))
        return _binding(
            widget,
            apply_value=_apply_multi_value,
            refresh_options=_refresh_multi_options if spec.multiselect_pool_field else None,
        )

    if spec.enum_items or spec.select_pool_field or ui in {"select", "dropdown", "dropbox", "combo", "combobox"}:
        widget = F8OptionCombo(widget_parent)
        if style_applier is not None:
            style_applier(widget)
        items = _resolve_inline_items(spec.select_pool_field, spec.enum_items, pool_resolver)
        widget.set_options(items, labels=items)
        if spec.field_tooltip:
            widget.set_context_tooltip(spec.field_tooltip)
        widget.valueChanged.connect(  # type: ignore[attr-defined]
            lambda value: value_setter("" if value is None else str(value), push_undo=True)
        )

        def _apply_combo_value(value: Any) -> None:
            widget.set_value("" if value is None else str(value))

        def _refresh_combo_options() -> None:
            refreshed_items = _resolve_inline_items(spec.select_pool_field, spec.enum_items, pool_resolver)
            widget.set_options(refreshed_items, labels=refreshed_items)
            _apply_combo_value(value_getter())

        _apply_combo_value(value_getter())
        widget.set_read_only(bool(read_only))
        return _binding(
            widget,
            apply_value=_apply_combo_value,
            refresh_options=_refresh_combo_options if spec.select_pool_field else None,
        )

    if spec.schema_type == "boolean" or ui in {"switch", "toggle"}:
        widget = F8Switch(widget_parent)
        widget.set_labels("True", "False")
        if spec.field_tooltip:
            widget.setToolTip(spec.field_tooltip)

        def _apply_switch_value(value: Any) -> None:
            with QtCore.QSignalBlocker(widget):
                widget.set_value(bool(value) if value is not None else False)

        widget.valueChanged.connect(lambda value: value_setter(bool(value), push_undo=True))  # type: ignore[attr-defined]
        _apply_switch_value(value_getter())
        widget.setEnabled(not bool(read_only))
        return _binding(widget, apply_value=_apply_switch_value)

    if dial_loop is not None:
        widget = F8Dial(widget_parent, integer=(spec.schema_type == "integer"), minimum=0.0, maximum=1.0)
        widget.set_loop(dial_loop)
        widget.set_range(spec.minimum, spec.maximum)
        if spec.field_tooltip:
            widget.set_context_tooltip(spec.field_tooltip)
            if tooltip_filter_installer is not None:
                tooltip_filter_installer(widget)
        if spec.schema_type not in {"integer", "number"}:
            widget.set_invalid_reason("Dial control requires integer or number state schema.")
        widget.valueChanging.connect(lambda value: value_setter(value, push_undo=False))  # type: ignore[attr-defined]
        widget.valueCommitted.connect(lambda value: value_setter(value, push_undo=True))  # type: ignore[attr-defined]
        widget.set_value(value_getter())
        widget.set_read_only(bool(read_only))
        return _binding(widget, apply_value=widget.set_value)

    if spec.schema_type in {"integer", "number"} and ui == "slider":
        widget = F8ValueBar(widget_parent, integer=(spec.schema_type == "integer"), minimum=0.0, maximum=1.0)
        widget.set_range(spec.minimum, spec.maximum)

        def _apply_slider_value(value: Any) -> None:
            widget.set_value(value)

        widget.valueChanging.connect(lambda value: value_setter(value, push_undo=False))  # type: ignore[attr-defined]
        widget.valueCommitted.connect(lambda value: value_setter(value, push_undo=True))  # type: ignore[attr-defined]
        _apply_slider_value(value_getter())
        widget.set_read_only(bool(read_only))
        return _binding(widget, apply_value=_apply_slider_value)

    if ui == "range_slider":
        widget = F8RangeBar(
            widget_parent,
            integer=spec.range_integer,
            minimum=0.0,
            maximum=1.0,
        )
        widget.set_range(spec.minimum, spec.maximum)

        def _apply_range_value(value: Any) -> None:
            widget.set_value(value)

        widget.valueChanging.connect(lambda value: value_setter(value, push_undo=False))  # type: ignore[attr-defined]
        widget.valueCommitted.connect(lambda value: value_setter(value, push_undo=True))  # type: ignore[attr-defined]
        _apply_range_value(value_getter())
        if spec.schema_type != "array":
            widget.setEnabled(False)
            widget.setToolTip("Range slider requires an array state schema with two numeric values.")
        widget.set_read_only(bool(read_only))
        return _binding(widget, apply_value=_apply_range_value)

    if spec.schema_type == "integer" or ui in {"spinbox", "int"}:
        return _build_inline_number_binding(
            spec=spec,
            read_only=read_only,
            widget_parent=widget_parent,
            data_type=int,
            value_getter=value_getter,
            value_setter=value_setter,
            style_applier=style_applier,
            text_palette_applier=text_palette_applier,
            tooltip_filter_installer=tooltip_filter_installer,
        )

    if spec.schema_type == "number" or ui in {"doublespinbox", "float"}:
        return _build_inline_number_binding(
            spec=spec,
            read_only=read_only,
            widget_parent=widget_parent,
            data_type=float,
            value_getter=value_getter,
            value_setter=value_setter,
            style_applier=style_applier,
            text_palette_applier=text_palette_applier,
            tooltip_filter_installer=tooltip_filter_installer,
        )

    widget = QtWidgets.QLineEdit(widget_parent)
    widget.setMinimumWidth(90)
    if style_applier is not None:
        style_applier(widget)
    if text_palette_applier is not None:
        text_palette_applier(widget)
    if spec.field_tooltip:
        widget.setToolTip(spec.field_tooltip)
        if tooltip_filter_installer is not None:
            tooltip_filter_installer(widget)

    def _apply_line_value(value: Any) -> None:
        text = "" if value is None else str(value)
        with QtCore.QSignalBlocker(widget):
            widget.setText(text)

    _apply_line_value(value_getter())
    if read_only:
        widget.setReadOnly(True)
    else:
        widget.editingFinished.connect(lambda: value_setter(widget.text(), push_undo=True))
    return _binding(widget, apply_value=_apply_line_value)


def _apply_inline_text_widget_style(
    widget: QtWidgets.QWidget,
    field_tooltip: str,
    *,
    style_applier: Callable[[QtWidgets.QWidget], None] | None,
    text_palette_applier: Callable[[QtWidgets.QWidget], None] | None,
    tooltip_filter_installer: Callable[[QtWidgets.QWidget], None] | None,
) -> None:
    if style_applier is not None:
        style_applier(widget)
    if text_palette_applier is not None:
        text_palette_applier(widget)
    if field_tooltip:
        widget.setToolTip(field_tooltip)
        if tooltip_filter_installer is not None:
            tooltip_filter_installer(widget)


def _resolve_inline_items(
    pool_field: str | None,
    enum_items: list[str],
    pool_resolver: Callable[[str], list[str]],
) -> list[str]:
    if pool_field:
        return pool_resolver(pool_field)
    return list(enum_items)


def _build_inline_number_binding(
    *,
    spec: StateControlSpec,
    read_only: bool,
    widget_parent: QtWidgets.QWidget | None = None,
    data_type: type[int] | type[float],
    value_getter: Callable[[], Any],
    value_setter: Callable[[Any, bool], None],
    style_applier: Callable[[QtWidgets.QWidget], None] | None,
    text_palette_applier: Callable[[QtWidgets.QWidget], None] | None,
    tooltip_filter_installer: Callable[[QtWidgets.QWidget], None] | None,
) -> StateControlBinding:
    widget = F8NumberLineEditor(widget_parent, data_type=data_type)
    widget.set_name(spec.name)
    widget.setMinimumWidth(90)
    if spec.minimum is not None:
        widget.set_min(spec.minimum)
    if spec.maximum is not None:
        widget.set_max(spec.maximum)
    _apply_inline_text_widget_style(
        widget,
        spec.field_tooltip,
        style_applier=style_applier,
        text_palette_applier=text_palette_applier,
        tooltip_filter_installer=tooltip_filter_installer,
    )
    widget.set_value(value_getter())
    if read_only:
        widget.setReadOnly(True)
    else:
        widget.value_changing.connect(lambda _field_name, value: value_setter(value, push_undo=False))  # type: ignore[attr-defined]
        widget.value_changed.connect(lambda _field_name, value: value_setter(value, push_undo=True))  # type: ignore[attr-defined]
    return _binding(widget, apply_value=widget.set_value)
