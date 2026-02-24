from __future__ import annotations

import json
import logging
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from f8pysdk.schema_helpers import schema_type

import qtawesome as qta

from ...widgets.f8_editor_widgets import (
    F8ImageB64Editor,
    F8MultiSelect,
    F8OptionCombo,
    F8Switch,
    F8ValueBar,
    parse_multiselect_pool,
    parse_select_pool,
)
from ...widgets.f8_prop_value_widgets import F8NumberPropLineEdit, open_code_editor_dialog, open_code_editor_window
from .node_item_core import StateFieldInfo, state_field_info
from .service_toolbar_host import F8ElideToolButton

logger = logging.getLogger(__name__)


def inline_state_input_is_connected(node_item: Any, field_name: str) -> bool:
    """
    True if the state field is upstream-driven via a state-edge.
    """
    name = str(field_name or "").strip()
    if not name:
        return False
    node = node_item._backend_node()
    if node is None:
        return False
    port = node.get_input(f"[S]{name}")
    if port is None:
        return False
    return bool(port.connected_ports())


def set_inline_state_control_read_only(control: QtWidgets.QWidget, *, read_only: bool) -> None:
    """
    Best-effort toggle for inline state controls hosted in the node item.
    """
    if isinstance(control, F8OptionCombo):
        control.set_read_only(bool(read_only))
        return
    if isinstance(control, F8Switch):
        control.setEnabled(not bool(read_only))
        return
    if isinstance(control, F8ValueBar):
        control.setEnabled(not bool(read_only))
        return
    if isinstance(control, QtWidgets.QLineEdit):
        control.setEnabled(True)
        control.setReadOnly(bool(read_only))
        return
    if isinstance(control, QtWidgets.QPlainTextEdit):
        control.setEnabled(True)
        control.setReadOnly(bool(read_only))
        return
    if isinstance(control, QtWidgets.QTextEdit):
        control.setEnabled(True)
        control.setReadOnly(bool(read_only))
        if read_only:
            control.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
        else:
            control.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
        return
    control.setEnabled(not bool(read_only))


def refresh_inline_state_read_only(node_item: Any) -> None:
    """
    Refresh readonly state for already-built inline state controls.
    """
    node = node_item._backend_node()
    if node is None:
        return
    try:
        fields = list(node.effective_state_fields() or [])
    except Exception:
        fields = []
    for field in fields:
        info = state_field_info(field)
        if info is None or not info.show_on_node:
            continue
        name = info.name
        ctrl = node_item._state_inline_controls.get(name)
        if ctrl is None:
            continue
        read_only = info.access_str == "ro" or inline_state_input_is_connected(node_item, name)
        set_inline_state_control_read_only(ctrl, read_only=bool(read_only))


def on_graph_property_changed(node_item: Any, node: Any, name: str, value: Any) -> None:
    """
    Keep inline state widgets in sync with NodeGraphQt properties.

    The inspector already tracks these through NodeGraphQt's own property
    widgets; since inline widgets are custom QWidgets, mirror updates here to
    get the same "two-way binding" behavior.
    """
    try:
        if str(node.id or "") != str(node_item.id or ""):
            return
    except (AttributeError, TypeError):
        return
    key = str(name or "").strip()
    if not key:
        return
    updater = node_item._state_inline_updaters.get(key)
    if not updater:
        refresh_option_pool_for_changed_field(node_item, key)
        return
    try:
        updater(value)
    except Exception:
        try:
            node_id = str(node_item.id or "")
        except Exception:
            node_id = ""
        logger.exception("inline state updater failed nodeId=%s key=%s", node_id, key)
    refresh_option_pool_for_changed_field(node_item, key)


def refresh_option_pool_for_changed_field(node_item: Any, changed_field: str) -> None:
    """
    If `changed_field` is used as an option-pool, refresh all dependent option controls.
    """
    pool = str(changed_field or "").strip()
    if not pool:
        return
    if pool not in set(node_item._state_inline_option_pools.values()):
        return
    node = node_item._backend_node()
    if node is None:
        return
    try:
        pool_value = node.get_property(pool)
    except Exception:
        pool_value = None
    if isinstance(pool_value, (list, tuple)):
        items = [str(item) for item in pool_value]
    else:
        items = []
    for field, pool_name in list(node_item._state_inline_option_pools.items()):
        if pool_name != pool:
            continue
        ctrl = node_item._state_inline_controls.get(field)
        if not isinstance(ctrl, (F8OptionCombo, F8MultiSelect)):
            continue
        try:
            cur = ctrl.value()
            ctrl.set_options(items, labels=items)
            ctrl.set_value(cur)
        except (AttributeError, RuntimeError, TypeError):
            continue


def on_state_toggle(node_item: Any, name: str, expanded: bool) -> None:
    state_name = str(name)
    old_scene_rect = None
    try:
        old_scene_rect = node_item.mapToScene(node_item.boundingRect()).boundingRect()
    except RuntimeError:
        old_scene_rect = None

    node_item._state_inline_expanded[state_name] = bool(expanded)
    node = node_item._backend_node()
    if node is not None:
        try:
            ui = dict(node.ui_overrides() or {})
            store = ui.get("stateInlineExpanded")
            if not isinstance(store, dict):
                store = {}
            store[state_name] = bool(expanded)
            ui["stateInlineExpanded"] = store
            node.set_ui_overrides(ui, rebuild=False)
        except AttributeError:
            logger.exception("node missing ui_overrides/set_ui_overrides; cannot persist expand state")
    btn = node_item._state_inline_toggles.get(state_name)
    if btn is not None:
        try:
            btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        except RuntimeError:
            pass
    body = node_item._state_inline_bodies.get(state_name)
    if body is not None:
        try:
            body.setVisible(bool(expanded))
        except RuntimeError:
            pass

    def _redraw_and_invalidate() -> None:
        node_item.draw_node()
        new_scene_rect = node_item.mapToScene(node_item.boundingRect()).boundingRect()
        rect = new_scene_rect
        if old_scene_rect is not None:
            rect = old_scene_rect.united(new_scene_rect)
        rect = rect.adjusted(-6, -6, 6, 6)
        scene = node_item.scene()
        if scene is not None:
            scene.update(rect)
        viewer = node_item.viewer()
        if viewer is not None:
            viewer.viewport().update()

    try:
        QtCore.QTimer.singleShot(0, _redraw_and_invalidate)
    except RuntimeError:
        _redraw_and_invalidate()


def make_state_inline_control(node_item: Any, state_field: StateFieldInfo) -> QtWidgets.QWidget:
    name = state_field.name
    ui_raw = state_field.ui_control
    ui = str(ui_raw or "").strip().lower()
    schema = state_field.value_schema
    access_s = state_field.access_str
    schema_type_value = (schema_type(schema) or "") if schema is not None else ""

    enum_items = node_item._schema_enum_items(schema)
    lo, hi = node_item._schema_numeric_range(schema)
    select_pool_field = parse_select_pool(ui_raw)
    multiselect_pool_field = parse_multiselect_pool(ui_raw)
    field_tooltip = state_field.tooltip if state_field.tooltip != name else ""

    def _common_style(widget: QtWidgets.QWidget) -> None:
        # Make controls readable on dark node themes.
        widget.setStyleSheet(
            """
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
                color: rgb(235, 235, 235);
                background: rgba(0, 0, 0, 45);
                border: 1px solid rgba(255, 255, 255, 55);
                border-radius: 3px;
                padding: 1px 4px;
            }
            QPlainTextEdit, QTextEdit {
                selection-background-color: rgb(80, 130, 180);
            }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView {
                color: rgb(235, 235, 235);
                background: rgb(35, 35, 35);
                selection-background-color: rgb(80, 130, 180);
            }
            QCheckBox { color: rgb(235, 235, 235); }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border: 1px solid rgba(255, 255, 255, 90);
                background: rgba(0, 0, 0, 35);
                border-radius: 2px;
            }
            QCheckBox::indicator:checked { background: rgba(120, 200, 255, 90); }
            """
        )

    def _set_node_value(value: Any, *, push_undo: bool) -> None:
        node = node_item._backend_node()
        if node is None or not name:
            return
        try:
            node.set_property(name, value, push_undo=push_undo)
        except TypeError:
            node.set_property(name, value)

    def _get_node_value() -> Any:
        node = node_item._backend_node()
        if node is None or not name:
            return None
        try:
            return node.get_property(name)
        except KeyError:
            return None

    def _pool_items(pool_field: str | None) -> list[str]:
        if not pool_field:
            return []
        node = node_item._backend_node()
        if node is None:
            return []
        try:
            value = node.get_property(pool_field)
        except Exception:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        # Allow pools stored as JSON strings (eg. "[]", ["a","b"]).
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                return []
            if isinstance(parsed, (list, tuple)):
                out: list[str] = []
                for item in parsed:
                    if isinstance(item, str):
                        token = item.strip()
                        if token:
                            out.append(token)
                        continue
                    if isinstance(item, dict):
                        token = str(item.get("id") or "").strip()
                        if token:
                            out.append(token)
                        continue
                    token = str(item).strip()
                    if token:
                        out.append(token)
                return out
        return []

    # Create control.
    read_only = access_s == "ro" or node_item._inline_state_input_is_connected(name)

    if ui in {"wrapline"}:

        class _InlineWrapLineEdit(QtWidgets.QPlainTextEdit):
            def __init__(self, parent: QtWidgets.QWidget | None = None):
                super().__init__(parent)
                self._prev = ""

            @staticmethod
            def _normalize(value: str) -> str:
                if "\n" not in value and "\r" not in value:
                    return value.strip()
                parts = [p.strip() for p in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
                return " ".join([p for p in parts if p]).strip()

            def focusInEvent(self, event):  # type: ignore[override]
                super().focusInEvent(event)
                self._prev = str(self.toPlainText() or "")

            def focusOutEvent(self, event):  # type: ignore[override]
                super().focusOutEvent(event)
                txt_raw = str(self.toPlainText() or "")
                txt = self._normalize(txt_raw)
                if txt != txt_raw:
                    with QtCore.QSignalBlocker(self):
                        self.setPlainText(txt)
                if txt != self._prev:
                    self._prev = txt
                    _set_node_value(txt, push_undo=True)

            def keyPressEvent(self, event):  # type: ignore[override]
                if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    txt_raw = str(self.toPlainText() or "")
                    txt = self._normalize(txt_raw)
                    if txt != txt_raw:
                        with QtCore.QSignalBlocker(self):
                            self.setPlainText(txt)
                    if txt != self._prev:
                        self._prev = txt
                        _set_node_value(txt, push_undo=True)
                    self.clearFocus()
                    event.accept()
                    return
                super().keyPressEvent(event)

            def insertFromMimeData(self, source: QtCore.QMimeData) -> None:  # type: ignore[override]
                if source is None or not source.hasText():
                    return super().insertFromMimeData(source)
                txt = self._normalize(str(source.text() or ""))
                if txt:
                    self.textCursor().insertText(txt)
                return

        edit = _InlineWrapLineEdit()
        edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        edit.setTabStopDistance(4 * edit.fontMetrics().horizontalAdvance(" "))
        try:
            font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            edit.setFont(font)
        except (AttributeError, RuntimeError, TypeError):
            pass
        edit.setMinimumWidth(0)
        edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        edit.setMinimumHeight(38)
        edit.setMaximumHeight(64)
        _common_style(edit)
        edit.document().setDocumentMargin(4.0)
        if field_tooltip:
            edit.setToolTip(field_tooltip)

        def _apply_value(value: Any) -> None:
            text = "" if value is None else str(value)
            text_normalized = _InlineWrapLineEdit._normalize(text)
            with QtCore.QSignalBlocker(edit):
                edit.setPlainText(text_normalized)
            try:
                edit._prev = text_normalized  # type: ignore[attr-defined]
            except RuntimeError:
                pass

        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if read_only:
            edit.setReadOnly(True)
        return edit

    if ui in {"code_inline", "multiline"}:

        class _InlineExprEdit(QtWidgets.QPlainTextEdit):
            def __init__(self, parent: QtWidgets.QWidget | None = None):
                super().__init__(parent)
                self._prev = ""

            def focusInEvent(self, event):  # type: ignore[override]
                super().focusInEvent(event)
                self._prev = str(self.toPlainText() or "")

            def focusOutEvent(self, event):  # type: ignore[override]
                super().focusOutEvent(event)
                txt = str(self.toPlainText() or "")
                if txt != self._prev:
                    self._prev = txt
                    _set_node_value(txt, push_undo=True)

            def keyPressEvent(self, event):  # type: ignore[override]
                if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and bool(
                    event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
                ):
                    txt = str(self.toPlainText() or "")
                    if txt != self._prev:
                        self._prev = txt
                        _set_node_value(txt, push_undo=True)
                    event.accept()
                    return
                super().keyPressEvent(event)

        edit = _InlineExprEdit()
        edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        edit.setTabStopDistance(4 * edit.fontMetrics().horizontalAdvance(" "))
        try:
            font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            edit.setFont(font)
        except (AttributeError, RuntimeError, TypeError):
            pass
        edit.setMinimumWidth(160)
        edit.setMinimumHeight(44)
        edit.setMaximumHeight(88)
        _common_style(edit)
        edit.document().setDocumentMargin(4.0)
        if field_tooltip:
            edit.setToolTip(field_tooltip)

        def _apply_value(value: Any) -> None:
            text = "" if value is None else str(value)
            with QtCore.QSignalBlocker(edit):
                edit.setPlainText(text)
            try:
                edit._prev = text  # type: ignore[attr-defined]
            except RuntimeError:
                pass

        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if read_only:
            edit.setReadOnly(True)
        return edit

    if ui in {"code"}:
        btn = QtWidgets.QToolButton()
        btn.setAutoRaise(True)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setText("Edit...")
        try:
            btn.setIcon(qta.icon("fa5s.code", color="white"))
        except (AttributeError, RuntimeError, TypeError):
            pass
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        if field_tooltip:
            btn.setToolTip(field_tooltip)

        def _apply_value(value: Any) -> None:
            text = "" if value is None else str(value)
            lines = len(text.splitlines()) if text else 0
            tip = field_tooltip or ""
            if lines:
                tip2 = f"{lines} line" if lines == 1 else f"{lines} lines"
                btn.setToolTip((tip + "\n" if tip else "") + tip2)

        def _on_click() -> None:
            current = _get_node_value()

            def _on_saved(updated: str) -> None:
                _set_node_value(updated, push_undo=True)

            try:
                dlg = open_code_editor_window(
                    None,
                    title=f"{node_item.name} - {state_field.label}",
                    code="" if current is None else str(current),
                    language=state_field.ui_language or "plaintext",
                    on_saved=_on_saved,
                )
                node_item._open_code_editors.append(dlg)

                def _cleanup() -> None:
                    alive: list[QtWidgets.QDialog] = []
                    for widget in node_item._open_code_editors:
                        if widget is None:
                            continue
                        try:
                            _ = widget.isVisible()
                            alive.append(widget)
                        except RuntimeError:
                            continue
                    node_item._open_code_editors = alive

                dlg.destroyed.connect(_cleanup)  # type: ignore[attr-defined]
            except Exception:
                updated = open_code_editor_dialog(
                    None,
                    title=f"{node_item.name} - {state_field.label}",
                    code="" if current is None else str(current),
                    language=state_field.ui_language or "plaintext",
                )
                if updated is None:
                    return
                _set_node_value(updated, push_undo=True)

        btn.clicked.connect(_on_click)  # type: ignore[attr-defined]
        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if read_only:
            btn.setDisabled(True)
        return btn

    is_image_b64 = schema_type_value == "string" and (ui in {"image", "image_b64", "img"} or "b64" in name.lower())
    if is_image_b64:
        img = F8ImageB64Editor()

        def _apply_value(value: Any) -> None:
            img.set_value("" if value is None else str(value))

        img.valueChanged.connect(lambda value: _set_node_value(str(value or ""), push_undo=True))  # type: ignore[attr-defined]
        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if read_only:
            img.set_disabled(True)
        return img

    if multiselect_pool_field or ui in {"multiselect", "multi_select", "multi-select"}:
        multi = F8MultiSelect()
        if field_tooltip:
            multi.set_context_tooltip(field_tooltip)

        items = _pool_items(multiselect_pool_field) if multiselect_pool_field else list(enum_items)
        multi.set_options(items, labels=items)

        def _apply_value(value: Any) -> None:
            multi.set_value(value)

        multi.valueChanged.connect(lambda value: _set_node_value(list(value or []), push_undo=True))  # type: ignore[attr-defined]
        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if multiselect_pool_field:
            node_item._state_inline_option_pools[name] = multiselect_pool_field
        if read_only:
            multi.set_read_only(True)
        return multi

    if enum_items or select_pool_field or ui in {"select", "dropdown", "dropbox", "combo", "combobox"}:
        combo = F8OptionCombo()
        _common_style(combo)

        items = _pool_items(select_pool_field) if select_pool_field else list(enum_items)
        combo.set_options(items, labels=items)
        if field_tooltip:
            combo.set_context_tooltip(field_tooltip)

        def _apply_value(value: Any) -> None:
            combo.set_value("" if value is None else str(value))

        combo.valueChanged.connect(  # type: ignore[attr-defined]
            lambda value: _set_node_value("" if value is None else str(value), push_undo=True)
        )
        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if select_pool_field:
            node_item._state_inline_option_pools[name] = select_pool_field
        if read_only:
            combo.set_read_only(True)
        return combo

    if schema_type_value == "boolean" or ui in {"switch", "toggle"}:
        sw = F8Switch()
        sw.set_labels("True", "False")
        if field_tooltip:
            sw.setToolTip(field_tooltip)

        def _apply_value(value: Any) -> None:
            with QtCore.QSignalBlocker(sw):
                sw.set_value(bool(value) if value is not None else False)

        sw.valueChanged.connect(lambda value: _set_node_value(bool(value), push_undo=True))  # type: ignore[attr-defined]
        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if read_only:
            sw.setDisabled(True)
        return sw

    if schema_type_value in {"integer", "number"} and ui == "slider":
        is_int = schema_type_value == "integer"
        bar = F8ValueBar(integer=is_int, minimum=0.0, maximum=1.0)
        bar.set_range(lo, hi)

        def _apply_value(value: Any) -> None:
            bar.set_value(value)

        bar.valueChanging.connect(lambda value: _set_node_value(value, push_undo=False))  # type: ignore[attr-defined]
        bar.valueCommitted.connect(lambda value: _set_node_value(value, push_undo=True))  # type: ignore[attr-defined]
        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if read_only:
            bar.setDisabled(True)
        return bar

    if schema_type_value == "integer" or ui in {"spinbox", "int"}:
        line = F8NumberPropLineEdit(data_type=int)
        line.set_name(name)
        _common_style(line)
        line.setMinimumWidth(90)
        if lo is not None:
            line.set_min(int(lo))
        if hi is not None:
            line.set_max(int(hi))
        if field_tooltip:
            line.setToolTip(field_tooltip)

        def _apply_value(value: Any) -> None:
            line.set_value(value)

        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if read_only:
            line.setReadOnly(True)
        else:
            line.value_changing.connect(lambda _field_name, value: _set_node_value(value, push_undo=False))  # type: ignore[attr-defined]
            line.value_changed.connect(lambda _field_name, value: _set_node_value(value, push_undo=True))  # type: ignore[attr-defined]
        return line

    if schema_type_value == "number" or ui in {"doublespinbox", "float"}:
        line = F8NumberPropLineEdit(data_type=float)
        line.set_name(name)
        _common_style(line)
        line.setMinimumWidth(90)
        if lo is not None:
            line.set_min(float(lo))
        if hi is not None:
            line.set_max(float(hi))
        if field_tooltip:
            line.setToolTip(field_tooltip)

        def _apply_value(value: Any) -> None:
            line.set_value(value)

        _apply_value(_get_node_value())
        node_item._state_inline_updaters[name] = _apply_value
        if read_only:
            line.setReadOnly(True)
        else:
            line.value_changing.connect(lambda _field_name, value: _set_node_value(value, push_undo=False))  # type: ignore[attr-defined]
            line.value_changed.connect(lambda _field_name, value: _set_node_value(value, push_undo=True))  # type: ignore[attr-defined]
        return line

    # default: text input.
    line = QtWidgets.QLineEdit()
    line.setMinimumWidth(90)
    _common_style(line)

    def _apply_value(value: Any) -> None:
        text = "" if value is None else str(value)
        with QtCore.QSignalBlocker(line):
            line.setText(text)

    _apply_value(_get_node_value())
    node_item._state_inline_updaters[name] = _apply_value
    if read_only:
        line.setReadOnly(True)
    else:
        line.editingFinished.connect(lambda: _set_node_value(line.text(), push_undo=True))
    return line


def ensure_inline_state_widgets(node_item: Any) -> None:
    node_item._ensure_graph_property_hook()
    node = node_item._backend_node()
    if node is None:
        return
    try:
        fields = list(node.effective_state_fields() or [])
    except Exception:
        try:
            spec = node.spec
        except Exception:
            spec = None
        if spec is None:
            fields = []
        else:
            try:
                fields = list(spec.stateFields or [])
            except Exception:
                fields = []

    show: list[StateFieldInfo] = []
    for field in fields:
        info = state_field_info(field)
        if info is None or not info.show_on_node:
            continue
        show.append(info)

    desired = [info.name for info in show]

    # delete stale widgets.
    for name in list(node_item._state_inline_proxies.keys()):
        if name in desired:
            continue
        proxy = node_item._state_inline_proxies.pop(name, None)
        node_item._state_inline_controls.pop(name, None)
        node_item._state_inline_updaters.pop(name, None)
        node_item._state_inline_toggles.pop(name, None)
        node_item._state_inline_headers.pop(name, None)
        node_item._state_inline_bodies.pop(name, None)
        node_item._state_inline_expanded.pop(name, None)
        node_item._state_inline_option_pools.pop(name, None)
        node_item._state_inline_ctrl_serial.pop(name, None)
        if proxy is None:
            continue
        old = None
        try:
            old = proxy.widget()
        except Exception:
            old = None
        try:
            proxy.setWidget(None)
        except RuntimeError:
            pass
        if old is not None:
            try:
                old.setParent(None)
            except RuntimeError:
                pass
            try:
                old.deleteLater()
            except RuntimeError:
                pass
        try:
            proxy.setParentItem(None)
            if node_item.scene() is not None:
                node_item.scene().removeItem(proxy)
        except RuntimeError:
            pass

    def _ctrl_serial(info: StateFieldInfo) -> str:
        """
        Signature for deciding when the control widget must be rebuilt.
        (Exclude label/description; those can be updated in-place.)
        """
        try:
            vs = info.value_schema
            enum_items = node_item._schema_enum_items(vs)
            return json.dumps(
                {
                    "access": info.access_str,
                    "required": info.required,
                    "uiControl": info.ui_control,
                    "uiLanguage": info.ui_language,
                    "schemaType": str(schema_type(vs) or ""),
                    "enum": [str(item) for item in enum_items],
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        except Exception:
            return ""

    for info in show:
        # Always keep label/tooltip up to date without rebuilding.
        name = info.name
        label = info.label or name
        tip = info.tooltip or name
        btn_existing = node_item._state_inline_toggles.get(name)
        if btn_existing is not None:
            try:
                btn_existing.setFullText(label)
            except RuntimeError:
                pass
            try:
                btn_existing.setToolTip(tip)
            except RuntimeError:
                pass

        ctrl_sig = _ctrl_serial(info)
        if name in node_item._state_inline_proxies and ctrl_sig and ctrl_sig == node_item._state_inline_ctrl_serial.get(name, ""):
            continue

        # Default collapsed; restore persisted expand state from ui overrides.
        expanded = False
        ui = node.ui_overrides() or {}
        store = ui.get("stateInlineExpanded") if isinstance(ui, dict) else None
        if isinstance(store, dict) and name in store:
            expanded = bool(store.get(name))
        expanded = bool(node_item._state_inline_expanded.get(name, expanded))
        control = node_item._make_state_inline_control(info)

        # Header: toggle button (state name).
        header = QtWidgets.QWidget()
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(0, 0, 0, 0)
        header_lay.setSpacing(6)
        header.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        header.setStyleSheet("background: transparent;")

        btn = F8ElideToolButton()
        btn.setCheckable(True)
        btn.setChecked(expanded)
        btn.setAutoRaise(True)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)

        btn.setFullText(label)
        btn.setToolTip(tip)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn.setStyleSheet(
            """
            QToolButton {
                color: rgb(235, 235, 235);
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 18);
                border-radius: 4px;
                padding: 2px 8px;
                text-align: left;
            }
            QToolButton:hover { background: transparent; }
            QToolButton:checked { background: transparent; }
            """
        )

        header_lay.addWidget(btn, 1)

        # Body: control widget (collapsed by default).
        body = QtWidgets.QWidget()
        body_lay = QtWidgets.QVBoxLayout(body)
        body_lay.setContentsMargins(8, 0, 8, 6)
        body_lay.setSpacing(0)
        body_lay.addWidget(control)
        body.setVisible(expanded)
        body.setStyleSheet(
            """
            QWidget {
                background: transparent;
                border: 0px;
            }
            """
        )

        panel = QtWidgets.QWidget()
        panel_lay = QtWidgets.QVBoxLayout(panel)
        panel_lay.setContentsMargins(0, 0, 0, 0)
        panel_lay.setSpacing(0)
        panel_lay.addWidget(header)
        panel_lay.addWidget(body)
        panel.setProperty("_f8_state_panel", True)
        panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        panel.setStyleSheet("background: transparent;")

        # Connect toggle.
        btn.toggled.connect(lambda v, _n=name: node_item._on_state_toggle(_n, bool(v)))  # type: ignore[attr-defined]
        btn.pressed.connect(node_item._select_node_from_embedded_widget)  # type: ignore[attr-defined]

        # Install/replace proxy.
        proxy = node_item._state_inline_proxies.get(name)
        if proxy is None:
            proxy = QtWidgets.QGraphicsProxyWidget(node_item)
            proxy.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
            node_item._state_inline_proxies[name] = proxy

        old = None
        try:
            old = proxy.widget()
        except Exception:
            old = None
        proxy.setWidget(panel)
        if old is not None and old is not panel:
            try:
                old.setParent(None)
            except RuntimeError:
                pass
            try:
                old.deleteLater()
            except RuntimeError:
                pass

        node_item._state_inline_controls[name] = control
        node_item._state_inline_toggles[name] = btn
        node_item._state_inline_headers[name] = header
        node_item._state_inline_bodies[name] = body
        node_item._state_inline_expanded[name] = expanded
        if ctrl_sig:
            node_item._state_inline_ctrl_serial[name] = ctrl_sig
