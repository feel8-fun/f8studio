from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any, Callable

from NodeGraphQt import PropertiesBinWidget
from NodeGraphQt.constants import NodeEnum, NodePropWidgetEnum
from NodeGraphQt.custom_widgets.properties_bin.node_property_factory import NodePropertyWidgetFactory
from NodeGraphQt.custom_widgets.properties_bin.node_property_widgets import PropLineEdit
from NodeGraphQt.custom_widgets.properties_bin.prop_widgets_base import PropLabel

from qtpy import QtWidgets, QtCore, QtGui
import qtawesome as qta

from f8pysdk import (
    F8Command,
    F8CommandParam,
    F8DataPortSpec,
    F8DataTypeSchema,
    F8OperatorSpec,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
)
from f8pysdk.schema_helpers import schema_default

from .f8_prop_value_widgets import (
    F8CodePropWidget as _F8CodePropWidget,
    F8CodeButtonPropWidget as _F8CodeButtonPropWidget,
    F8JsonPropTextEdit as _F8JsonPropTextEdit,
)
from .f8_editor_widgets import (
    F8OptionCombo,
    F8PropBoolSwitch,
    F8PropMultiSelect,
    F8PropOptionCombo,
    F8Switch,
    F8ValueBar,
)
from .f8_spec_ops import (
    add_command as _spec_add_command,
    add_state_field as _spec_add_state_field,
    delete_command as _spec_delete_command,
    delete_state_field as _spec_delete_state_field,
    replace_command as _spec_replace_command,
    replace_state_field as _spec_replace_state_field,
    set_ports as _spec_set_ports,
)
from .f8_state_widget_builder import (
    build_state_value_widget as _build_state_value_widget,
    effective_state_fields as _effective_state_fields,
    schema_enum_items as _schema_enum_items,
    schema_numeric_range as _schema_numeric_range,
    schema_type_any as _schema_type,
    state_field_access as _state_field_access,
    state_field_schema as _state_field_schema,
    state_field_ui_control as _state_field_ui_control,
    state_field_ui_language as _state_field_ui_language,
)
from .f8_ui_override_ops import (
    base_command_show_on_node as _base_command_show_on_node,
    base_data_port_show_on_node as _base_data_port_show_on_node,
    find_base_state_field as _find_base_state_field,
    set_command_show_on_node_override as _set_command_show_on_node_override,
    set_data_port_show_on_node_override as _set_data_port_show_on_node_override,
    set_state_field_ui_override as _set_state_field_ui_override,
)
from ..command_ui_protocol import CommandUiHandler, CommandUiSource


logger = logging.getLogger(__name__)


def _apply_read_only_widget(widget: QtWidgets.QWidget) -> None:
    """
    Apply a read-only UX without disabling selection/copy.

    For text-based editors, prefer `setReadOnly(True)` over `setDisabled(True)`
    so users can still select and copy values.
    """
    if isinstance(widget, F8PropOptionCombo):
        widget.set_read_only(True)
        return
    if isinstance(widget, F8PropMultiSelect):
        widget.set_read_only(True)
        return
    if isinstance(widget, F8PropBoolSwitch):
        widget.set_read_only(True)
        return
    if isinstance(widget, _F8CodeButtonPropWidget):
        widget.set_read_only(True)
        return
    if isinstance(widget, QtWidgets.QLineEdit):
        widget.setEnabled(True)
        widget.setReadOnly(True)
        return
    if isinstance(widget, QtWidgets.QPlainTextEdit):
        widget.setEnabled(True)
        widget.setReadOnly(True)
        return
    if isinstance(widget, QtWidgets.QTextEdit):
        widget.setEnabled(True)
        widget.setReadOnly(True)
        widget.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        return
    if isinstance(widget, QtWidgets.QAbstractSpinBox):
        widget.setEnabled(True)
        widget.setReadOnly(True)
        try:
            widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        except Exception:
            pass
        return
    widget.setDisabled(True)


def _set_read_only_widget(widget: QtWidgets.QWidget, *, read_only: bool) -> None:
    """
    Best-effort toggle for read-only UX.

    This mirrors `_apply_read_only_widget`, but also supports restoring editability
    when `read_only=False` (eg. when a state-edge is disconnected).
    """
    if read_only:
        _apply_read_only_widget(widget)
        return

    if isinstance(widget, F8PropOptionCombo):
        widget.set_read_only(False)
        return
    if isinstance(widget, F8PropMultiSelect):
        widget.set_read_only(False)
        return
    if isinstance(widget, F8PropBoolSwitch):
        widget.set_read_only(False)
        return
    if isinstance(widget, _F8CodeButtonPropWidget):
        widget.set_read_only(False)
        return
    if isinstance(widget, QtWidgets.QLineEdit):
        widget.setEnabled(True)
        widget.setReadOnly(False)
        return
    if isinstance(widget, QtWidgets.QPlainTextEdit):
        widget.setEnabled(True)
        widget.setReadOnly(False)
        return
    if isinstance(widget, QtWidgets.QTextEdit):
        widget.setEnabled(True)
        widget.setReadOnly(False)
        widget.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
        return
    if isinstance(widget, QtWidgets.QAbstractSpinBox):
        widget.setEnabled(True)
        widget.setReadOnly(False)
        try:
            widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        except Exception:
            pass
        return
    widget.setEnabled(True)


def _state_input_is_connected(node: Any, field_name: str) -> bool:
    name = str(field_name or "").strip()
    if not name:
        return False
    p = node.get_input(f"[S]{name}")
    if p is None:
        return False
    return bool(p.connected_ports())

def _model_extra(obj: Any) -> dict[str, Any]:
    try:
        extra = obj.model_extra
    except Exception:
        try:
            extra = obj.__pydantic_extra__
        except Exception:
            return {}
    if not isinstance(extra, dict):
        return {}
    return dict(extra)


def _extra_bool(obj: Any, key: str, default: bool = False) -> bool:
    try:
        extra = _model_extra(obj)
        if key in extra:
            return bool(extra.get(key))
    except Exception:
        pass
    return bool(default)


def _get_node_spec(node: Any) -> Any | None:
    try:
        return node.spec
    except Exception:
        return None


def _to_jsonable(value: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable primitives (dict/list/str/num/bool/None).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    # Enum-like: use `.value` if present.
    if not isinstance(value, (bytes, bytearray)):
        try:
            return _to_jsonable(value.value)
        except Exception:
            pass
    # Pydantic models (BaseModel / RootModel).
    try:
        dump = value.model_dump(mode="json")
    except Exception:
        try:
            dump = value.model_dump()
        except Exception:
            dump = None
    if dump is not None:
        return _to_jsonable(dump)
    # RootModel inner `.root`.
    try:
        root = value.root
    except Exception:
        root = None
    if root is not None:
        return _to_jsonable(root)
    return str(value)


def _schema_to_json_obj(schema: Any) -> Any:
    if schema is None:
        return None
    if isinstance(schema, (dict, list, str, int, float, bool)) or schema is None:
        return schema
    try:
        return schema.model_dump(mode="json")
    except Exception:
        pass
    try:
        root = schema.root
    except Exception:
        root = None
    if root is not None:
        try:
            return root.model_dump(mode="json")
        except Exception:
            return root
    return str(schema)


def _schema_from_json_obj(obj: Any) -> F8DataTypeSchema:
    if isinstance(obj, F8DataTypeSchema):
        return obj
    return F8DataTypeSchema.model_validate(obj)


class _F8JsonEditorDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title: str, value: Any):
        super().__init__(parent)
        self.setWindowTitle(title)

        self._edit = QtWidgets.QPlainTextEdit()
        self._edit.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            text = json.dumps(_to_jsonable(value), ensure_ascii=False, indent=2)
        self._edit.setPlainText(text)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._edit, 1)
        layout.addWidget(buttons)

    def value(self) -> Any:
        text = self._edit.toPlainText().strip()
        if not text:
            return None
        return json.loads(text)


class _F8StateContainer(QtWidgets.QWidget):
    """
    Node properties container widget that displays nodes properties under
    a tab in the ``NodePropWidget`` widget.
    """

    class _ElideLabel(QtWidgets.QLabel):
        def __init__(self, text: str, parent: QtWidgets.QWidget | None = None):
            super().__init__("", parent)
            self._full_text = str(text or "")
            self.setText(self._full_text)

        def setText(self, text: str) -> None:  # type: ignore[override]
            self._full_text = str(text or "")
            self._update_elide()

        def resizeEvent(self, event):  # type: ignore[override]
            super().resizeEvent(event)
            self._update_elide()

        def _update_elide(self) -> None:
            try:
                fm = QtGui.QFontMetrics(self.font())
                elided = fm.elidedText(self._full_text, QtCore.Qt.ElideRight, max(10, int(self.width())))
                super().setText(elided)
            except Exception:
                super().setText(self._full_text)

    def __init__(self, parent=None):
        super(_F8StateContainer, self).__init__(parent)
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setColumnStretch(1, 1)
        self.__layout.setSpacing(6)
        self.__layout.setColumnMinimumWidth(0, 90)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addLayout(self.__layout)

        self.__property_widgets = {}

    def __repr__(self):
        return "<{} object at {}>".format(self.__class__.__name__, hex(id(self)))

    def add_widget(self, name, widget, value, label=None, tooltip=None):
        """
        Add a property widget to the window.

        Args:
            name (str): property name to be displayed.
            widget (BaseProperty): property widget.
            value (object): property value.
            label (str): custom label to display.
            tooltip (str): custom tooltip.
        """
        label = label or name
        label_widget = _F8StateContainer._ElideLabel(label)
        # Keep the label column bounded so value widgets (eg. sliders) remain usable
        # in narrow PropertiesBin panels.
        label_widget.setMaximumWidth(150)
        if tooltip:
            widget.setToolTip("{}\n{}".format(name, tooltip))
            label_widget.setToolTip("{}\n{}".format(name, tooltip))
        else:
            widget.setToolTip(name)
            label_widget.setToolTip(name)
        widget.set_value(value)
        row = self.__layout.rowCount()
        if row > 0:
            row += 1

        label_flags = QtCore.Qt.AlignCenter | QtCore.Qt.AlignRight
        if widget.__class__.__name__ == "PropTextEdit":
            label_flags = label_flags | QtCore.Qt.AlignTop

        self.__layout.addWidget(label_widget, row, 0, label_flags)
        self.__layout.addWidget(widget, row, 1)
        self.__property_widgets[name] = widget

    def get_widget(self, name):
        """
        Returns the property widget from the name.

        Args:
            name (str): property name.

        Returns:
            QtWidgets.QWidget: property widget.
        """
        return self.__property_widgets.get(name)

    def get_all_widgets(self):
        """
        Returns the node property widgets.

        Returns:
            dict: {name: widget}
        """
        return self.__property_widgets


class _F8StateStackContainer(QtWidgets.QWidget):
    """
    State tab container with vertical layout:
      row1: name + edit stateField button
      row2: editor widget (full width)

    This avoids squeezing value widgets into a narrow 2-column grid.
    """

    edit_state_field_requested = QtCore.Signal(str)
    delete_state_field_requested = QtCore.Signal(str)
    add_state_field_requested = QtCore.Signal()
    toggle_state_field_show_on_node_requested = QtCore.Signal(str, bool)

    class _ElideLabel(QtWidgets.QLabel):
        def __init__(self, text: str, parent: QtWidgets.QWidget | None = None):
            super().__init__("", parent)
            self._full_text = str(text or "")
            self.setText(self._full_text)

        def setText(self, text: str) -> None:  # type: ignore[override]
            self._full_text = str(text or "")
            self._update_elide()

        def resizeEvent(self, event):  # type: ignore[override]
            super().resizeEvent(event)
            self._update_elide()

        def _update_elide(self) -> None:
            try:
                fm = QtGui.QFontMetrics(self.font())
                elided = fm.elidedText(self._full_text, QtCore.Qt.ElideRight, max(10, int(self.width())))
                super().setText(elided)
            except Exception:
                super().setText(self._full_text)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__property_widgets: dict[str, QtWidgets.QWidget] = {}

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(10)
        self._layout.setAlignment(QtCore.Qt.AlignTop)

        self._header = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(self._header)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        title = QtWidgets.QLabel("State Fields")
        f = title.font()
        try:
            f.setBold(True)
        except Exception:
            pass
        title.setFont(f)
        self._btn_add = QtWidgets.QToolButton()
        self._btn_add.setAutoRaise(True)
        self._btn_add.setToolTip("Add state field")
        self._btn_add.setIcon(_icon_from_style(self._btn_add, QtWidgets.QStyle.SP_FileDialogNewFolder, "list-add"))
        self._btn_add.clicked.connect(self.add_state_field_requested.emit)
        h.addWidget(title, 1)
        h.addWidget(self._btn_add, 0)
        self._layout.addWidget(self._header)

    def set_add_visible(self, visible: bool) -> None:
        self._btn_add.setVisible(bool(visible))

    def add_widget(
        self,
        name,
        widget,
        value,
        label=None,
        tooltip=None,
        *,
        allow_delete: bool = False,
        show_on_node: bool = True,
    ):
        label = label or name

        section = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(section)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        header = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(header)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        label_widget = _F8StateStackContainer._ElideLabel(label)
        f = label_widget.font()
        f.setBold(True)
        label_widget.setFont(f)

        edit_btn = QtWidgets.QToolButton()
        edit_btn.setAutoRaise(True)
        edit_btn.setToolTip("Edit stateField…")
        edit_btn.setIcon(_icon_from_style(edit_btn, QtWidgets.QStyle.SP_FileDialogDetailedView, "document-edit"))
        edit_btn.setProperty("_state_field_name", str(name or "").strip())
        edit_btn.clicked.connect(self._on_edit_clicked)

        del_btn = QtWidgets.QToolButton()
        del_btn.setAutoRaise(True)
        del_btn.setToolTip("Delete stateField")
        del_btn.setIcon(_icon_from_style(del_btn, QtWidgets.QStyle.SP_TrashIcon, "edit-delete"))
        del_btn.setVisible(bool(allow_delete))
        del_btn.setProperty("_state_field_name", str(name or "").strip())
        del_btn.clicked.connect(self._on_delete_clicked)

        eye_btn = QtWidgets.QToolButton()
        eye_btn.setAutoRaise(True)
        eye_btn.setCheckable(True)
        eye_btn.setChecked(bool(show_on_node))
        eye_btn.setToolTip("Show on node")
        icon_name = "fa5s.eye" if bool(show_on_node) else "fa5s.eye-slash"
        eye_btn.setIcon(qta.icon(icon_name, color="white"))
        eye_btn.setProperty("_state_field_name", str(name or "").strip())
        eye_btn.toggled.connect(self._on_eye_toggled)  # type: ignore[attr-defined]

        h.addWidget(label_widget, 1)
        h.addWidget(edit_btn, 0)
        h.addWidget(eye_btn, 0)
        h.addWidget(del_btn, 0)

        if tooltip:
            tip = "{}\n{}".format(name, tooltip)
            label_widget.setToolTip(tip)
            edit_btn.setToolTip("Edit stateField…\n" + tip)
            del_btn.setToolTip("Delete stateField\n" + tip)
            widget.setToolTip(tip)
        else:
            label_widget.setToolTip(str(name))
            widget.setToolTip(str(name))

        widget.set_value(value)
        v.addWidget(header)

        body = QtWidgets.QWidget()
        body_l = QtWidgets.QVBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(0)
        body_l.addWidget(widget)
        v.addWidget(body)

        self._layout.addWidget(section)
        self.__property_widgets[name] = widget

    def _on_edit_clicked(self, _checked: bool = False) -> None:
        btn = self.sender()
        name = str(btn.property("_state_field_name") or "").strip() if btn is not None else ""
        if name:
            self.edit_state_field_requested.emit(name)

    def _on_delete_clicked(self, _checked: bool = False) -> None:
        btn = self.sender()
        name = str(btn.property("_state_field_name") or "").strip() if btn is not None else ""
        if name:
            self.delete_state_field_requested.emit(name)

    def _on_eye_toggled(self, checked: bool) -> None:
        btn = self.sender()
        name = str(btn.property("_state_field_name") or "").strip() if btn is not None else ""
        if not name:
            return
        icon_name = "fa5s.eye" if bool(checked) else "fa5s.eye-slash"
        btn.setIcon(qta.icon(icon_name, color="white"))
        self.toggle_state_field_show_on_node_requested.emit(name, bool(checked))

    def get_widget(self, name):
        return self.__property_widgets.get(name)

    def get_all_widgets(self):
        return self.__property_widgets


def _icon_from_style(
    widget: QtWidgets.QWidget, style_icon: QtWidgets.QStyle.StandardPixmap, fallback: str
) -> QtGui.QIcon:
    icon = widget.style().standardIcon(style_icon)
    if not icon.isNull():
        return icon
    icon = QtGui.QIcon.fromTheme(fallback)
    if not icon.isNull():
        return icon
    return QtGui.QIcon()


class _F8SpecListSection(QtWidgets.QWidget):
    """
    Sidebar-friendly list group with a header and a "+" add button.
    """

    add_clicked = QtCore.Signal()

    def __init__(self, parent=None, *, title: str):
        super().__init__(parent)
        self._title = title

        header_label = QtWidgets.QLabel(title)
        f = header_label.font()
        f.setBold(True)
        header_label.setFont(f)

        self._add_btn = QtWidgets.QToolButton()
        self._add_btn.setAutoRaise(True)
        self._add_btn.setToolTip("Add")
        self._add_btn.setIcon(_icon_from_style(self._add_btn, QtWidgets.QStyle.SP_FileDialogNewFolder, "list-add"))
        self._add_btn.clicked.connect(self.add_clicked.emit)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(header_label)
        header.addStretch(1)
        header.addWidget(self._add_btn)

        self._list_layout = QtWidgets.QVBoxLayout()
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addLayout(header)
        outer.addLayout(self._list_layout)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

    def set_add_visible(self, visible: bool) -> None:
        try:
            self._add_btn.setVisible(bool(visible))
        except Exception:
            pass

    def clear(self) -> None:
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def add_row(self, row: QtWidgets.QWidget) -> None:
        self._list_layout.addWidget(row)

    def rows(self) -> list[QtWidgets.QWidget]:
        out: list[QtWidgets.QWidget] = []
        for i in range(self._list_layout.count()):
            w = self._list_layout.itemAt(i).widget()
            if w is not None:
                out.append(w)
        return out


class _F8SpecNameRow(QtWidgets.QWidget):
    edit_clicked = QtCore.Signal()
    delete_clicked = QtCore.Signal()
    name_committed = QtCore.Signal(str)
    show_on_node_changed = QtCore.Signal(bool)

    def __init__(self, parent=None, *, name: str, placeholder: str, show_eye: bool = False):
        super().__init__(parent)

        self.name_edit = QtWidgets.QLineEdit(name)
        self.name_edit.setPlaceholderText(placeholder)
        self.name_edit.setClearButtonEnabled(True)
        self.name_edit.editingFinished.connect(self._emit_commit)

        self.edit_btn = QtWidgets.QToolButton()
        self.edit_btn.setAutoRaise(True)
        self.edit_btn.setToolTip("Edit")
        self.edit_btn.setIcon(_icon_from_style(self.edit_btn, QtWidgets.QStyle.SP_FileDialogDetailedView, "document-edit"))
        self.edit_btn.clicked.connect(self.edit_clicked.emit)

        self.eye_btn = QtWidgets.QToolButton()
        self.eye_btn.setAutoRaise(True)
        self.eye_btn.setCheckable(True)
        self.eye_btn.setToolTip("Show on node")
        self.eye_btn.toggled.connect(self._on_eye_toggled)  # type: ignore[attr-defined]
        self._update_eye_icon(True)

        self.del_btn = QtWidgets.QToolButton()
        self.del_btn.setAutoRaise(True)
        self.del_btn.setToolTip("Delete")
        self.del_btn.setIcon(_icon_from_style(self.del_btn, QtWidgets.QStyle.SP_TrashIcon, "edit-delete"))
        self.del_btn.clicked.connect(self.delete_clicked.emit)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.name_edit, 1)
        layout.addWidget(self.edit_btn)
        layout.addWidget(self.eye_btn)
        layout.addWidget(self.del_btn)
        self.eye_btn.setVisible(bool(show_eye))
        self.eye_btn.setEnabled(bool(show_eye))

    def set_row_editable(self, *, allow_rename: bool, allow_delete: bool, allow_edit: bool = True) -> None:
        self.name_edit.setReadOnly(not bool(allow_rename))
        self.del_btn.setVisible(bool(allow_delete))
        self.edit_btn.setVisible(bool(allow_edit))
        self.edit_btn.setEnabled(bool(allow_edit))

    def set_show_on_node(self, show: bool) -> None:
        with QtCore.QSignalBlocker(self.eye_btn):
            self.eye_btn.setChecked(bool(show))
        self._update_eye_icon(bool(show))

    def _update_eye_icon(self, show: bool) -> None:
        icon_name = "fa5s.eye" if bool(show) else "fa5s.eye-slash"
        self.eye_btn.setIcon(qta.icon(icon_name, color="white"))

    def _on_eye_toggled(self, checked: bool) -> None:
        self._update_eye_icon(bool(checked))
        self.show_on_node_changed.emit(bool(checked))

    def _emit_commit(self) -> None:
        self.name_committed.emit(str(self.name_edit.text() or "").strip())


class _F8EditExecPortDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title: str, name: str):
        super().__init__(parent)
        self.setWindowTitle(title)

        self._name = QtWidgets.QLineEdit(name)
        self._name.setClearButtonEnabled(True)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self._name)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def name(self) -> str:
        return str(self._name.text() or "").strip()


class _F8EditDataPortDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title: str, port: F8DataPortSpec, ui_only: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._ui_only = bool(ui_only)
        self._schema = port.valueSchema or _schema_from_json_obj({"type": "any"})

        self._name = QtWidgets.QLineEdit(str(port.name or ""))
        self._name.setClearButtonEnabled(True)
        self._required = QtWidgets.QCheckBox()
        self._required.setChecked(bool(port.required))
        self._show_on_node = QtWidgets.QCheckBox()
        self._show_on_node.setChecked(bool(port.showOnNode))
        self._desc = QtWidgets.QPlainTextEdit(str(port.description or ""))

        self._schema_summary = QtWidgets.QLabel("")
        self._schema_summary.setStyleSheet("color: #888;")
        self._refresh_schema_summary()

        schema_btn = QtWidgets.QPushButton("Edit Schema…")
        schema_btn.clicked.connect(self._edit_schema)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self._name)
        form.addRow("Required", self._required)
        form.addRow("Show On Node", self._show_on_node)
        form.addRow("Description", self._desc)

        schema_row = QtWidgets.QHBoxLayout()
        schema_row.addWidget(self._schema_summary, 1)
        schema_row.addWidget(schema_btn)
        form.addRow("valueSchema", schema_row)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if self._ui_only:
            for w in (self._name, self._required, schema_btn):
                w.setEnabled(False)

    def _refresh_schema_summary(self) -> None:
        t = _schema_type(self._schema)
        self._schema_summary.setText(t or "unknown")

    def _edit_schema(self) -> None:
        init = _schema_to_json_obj(self._schema) or {"type": "any"}
        dlg = _F8JsonEditorDialog(self, title="Edit valueSchema", value=init)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            self._schema = _schema_from_json_obj(dlg.value() or {"type": "any"})
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid schema", str(e))
            return
        self._refresh_schema_summary()

    def port(self) -> F8DataPortSpec:
        name = str(self._name.text() or "").strip()
        required = bool(self._required.isChecked())
        show_on_node = bool(self._show_on_node.isChecked())
        desc = str(self._desc.toPlainText() or "").strip() or None
        port = F8DataPortSpec(name=name, required=required, description=desc, valueSchema=self._schema)
        try:
            return port.model_copy(update={"showOnNode": bool(show_on_node)})
        except Exception:
            return port


class _F8EditStateFieldDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title: str, field: F8StateSpec, ui_only: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        try:
            self._schema = field.valueSchema or _schema_from_json_obj({"type": "any"})
        except Exception:
            self._schema = _schema_from_json_obj({"type": "any"})
        self._ui_only = bool(ui_only)

        self._name = QtWidgets.QLineEdit(str(field.name or ""))
        self._name.setClearButtonEnabled(True)

        self._access = QtWidgets.QComboBox()
        self._access.addItems([e.value for e in F8StateAccess])
        try:
            self._access.setCurrentText(str(field.access.value))
        except Exception:
            self._access.setCurrentText("rw")

        self._required = QtWidgets.QCheckBox()
        self._required.setChecked(bool(field.required))

        self._show_on_node = QtWidgets.QCheckBox()
        self._show_on_node.setChecked(bool(field.showOnNode))

        self._label = QtWidgets.QLineEdit(str(field.label or ""))
        self._label.setClearButtonEnabled(True)
        self._desc = QtWidgets.QPlainTextEdit(str(field.description or ""))
        self._ui_control = QtWidgets.QLineEdit(str(field.uiControl or ""))
        self._ui_control.setClearButtonEnabled(True)
        self._ui_lang = QtWidgets.QLineEdit(str(field.uiLanguage or ""))
        self._ui_lang.setClearButtonEnabled(True)

        self._schema_summary = QtWidgets.QLabel("")
        self._schema_summary.setStyleSheet("color: #888;")
        self._refresh_schema_summary()

        schema_btn = QtWidgets.QPushButton("Edit Schema…")
        schema_btn.clicked.connect(self._edit_schema)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self._name)
        form.addRow("Access", self._access)
        form.addRow("Required", self._required)
        form.addRow("Show On Node", self._show_on_node)
        form.addRow("Label", self._label)
        form.addRow("Description", self._desc)
        form.addRow("uiControl", self._ui_control)
        form.addRow("uiLanguage", self._ui_lang)

        schema_row = QtWidgets.QHBoxLayout()
        schema_row.addWidget(self._schema_summary, 1)
        schema_row.addWidget(schema_btn)
        form.addRow("valueSchema", schema_row)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if self._ui_only:
            for w in (self._name, self._access, self._required, schema_btn, self._schema_summary):
                w.setEnabled(False)
            self._name.setToolTip("Locked by spec (required/non-editable).")

    def _refresh_schema_summary(self) -> None:
        t = _schema_type(self._schema)
        self._schema_summary.setText(t or "unknown")

    def _edit_schema(self) -> None:
        init = _schema_to_json_obj(self._schema) or {"type": "any"}
        dlg = _F8JsonEditorDialog(self, title="Edit valueSchema", value=init)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            self._schema = _schema_from_json_obj(dlg.value() or {"type": "any"})
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid schema", str(e))
            return
        self._refresh_schema_summary()

    def field(self) -> F8StateSpec:
        name = str(self._name.text() or "").strip()
        access_s = str(self._access.currentText() or "rw")
        try:
            access = F8StateAccess(access_s)
        except Exception:
            access = F8StateAccess.rw
        required = bool(self._required.isChecked())
        show_on_node = bool(self._show_on_node.isChecked())
        label = str(self._label.text() or "").strip() or None
        desc = str(self._desc.toPlainText() or "").strip() or None
        ui_control = str(self._ui_control.text() or "").strip() or None
        ui_lang = str(self._ui_lang.text() or "").strip() or None
        return F8StateSpec(
            name=name,
            label=label,
            description=desc,
            valueSchema=self._schema,
            access=access,
            required=required,
            uiControl=ui_control,
            uiLanguage=ui_lang,
            showOnNode=show_on_node,
        )


class _F8SpecPortEditor(QtWidgets.QWidget):
    """
    Narrow-sidebar friendly spec ports editor.
    """

    spec_applied = QtCore.Signal()

    def __init__(self, parent=None, node=None, on_apply: Callable[[], None] | None = None):
        super().__init__(parent)
        self._node = node
        self._on_apply = on_apply
        self._editable_exec_in = False
        self._editable_exec_out = False
        self._editable_data_in = False
        self._editable_data_out = False

        self._sec_exec_in = _F8SpecListSection(title="Exec In")
        self._sec_exec_out = _F8SpecListSection(title="Exec Out")
        self._sec_data_in = _F8SpecListSection(title="Data In")
        self._sec_data_out = _F8SpecListSection(title="Data Out")

        self._sec_exec_in.add_clicked.connect(lambda: self._add_exec(True))
        self._sec_exec_out.add_clicked.connect(lambda: self._add_exec(False))
        self._sec_data_in.add_clicked.connect(lambda: self._add_data(True))
        self._sec_data_out.add_clicked.connect(lambda: self._add_data(False))

        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(8)
        v.addWidget(self._sec_exec_in)
        v.addWidget(self._sec_exec_out)
        v.addWidget(self._sec_data_in)
        v.addWidget(self._sec_data_out)
        v.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(content)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self._load_from_spec()

    def _load_from_spec(self) -> None:
        try:
            spec = self._node.spec
        except Exception:
            spec = None
        is_operator = isinstance(spec, F8OperatorSpec)
        self._sec_exec_in.setVisible(is_operator)
        self._sec_exec_out.setVisible(is_operator)

        self._sec_exec_in.clear()
        self._sec_exec_out.clear()
        self._sec_data_in.clear()
        self._sec_data_out.clear()

        if spec is None:
            return

        try:
            self._editable_data_in = bool(spec.editableDataInPorts)  # type: ignore[attr-defined]
        except Exception:
            self._editable_data_in = _extra_bool(spec, "editableDataInPorts", False)
        try:
            self._editable_data_out = bool(spec.editableDataOutPorts)  # type: ignore[attr-defined]
        except Exception:
            self._editable_data_out = _extra_bool(spec, "editableDataOutPorts", False)
        if is_operator:
            try:
                self._editable_exec_in = bool(spec.editableExecInPorts)  # type: ignore[attr-defined]
            except Exception:
                self._editable_exec_in = _extra_bool(spec, "editableExecInPorts", False)
            try:
                self._editable_exec_out = bool(spec.editableExecOutPorts)  # type: ignore[attr-defined]
            except Exception:
                self._editable_exec_out = _extra_bool(spec, "editableExecOutPorts", False)
        else:
            self._editable_exec_in = False
            self._editable_exec_out = False

        self._sec_exec_in.set_add_visible(bool(self._editable_exec_in))
        self._sec_exec_out.set_add_visible(bool(self._editable_exec_out))
        self._sec_data_in.set_add_visible(bool(self._editable_data_in))
        self._sec_data_out.set_add_visible(bool(self._editable_data_out))

        if is_operator:
            for name in list(spec.execInPorts or []):
                self._sec_exec_in.add_row(self._make_exec_row(str(name), is_in=True))
            for name in list(spec.execOutPorts or []):
                self._sec_exec_out.add_row(self._make_exec_row(str(name), is_in=False))

        try:
            data_in_ports = list(spec.dataInPorts or [])
        except Exception:
            data_in_ports = []
        try:
            data_out_ports = list(spec.dataOutPorts or [])
        except Exception:
            data_out_ports = []

        for p in data_in_ports:
            self._sec_data_in.add_row(self._make_data_row(p, is_in=True))
        for p in data_out_ports:
            self._sec_data_out.add_row(self._make_data_row(p, is_in=False))

    def _make_exec_row(self, name: str, *, is_in: bool) -> _F8SpecNameRow:
        row = _F8SpecNameRow(name=name, placeholder="port name")
        row.edit_clicked.connect(lambda: self._edit_exec(row))
        row.delete_clicked.connect(lambda: self._delete_row(row))
        row.name_committed.connect(lambda _v: self._commit())
        row.setProperty("_port_dir", "exec_in" if is_in else "exec_out")
        editable = bool(self._editable_exec_in if is_in else self._editable_exec_out)
        row.set_row_editable(allow_rename=editable, allow_delete=editable, allow_edit=editable)
        return row

    def _make_data_row(self, port: F8DataPortSpec, *, is_in: bool) -> _F8SpecNameRow:
        row = _F8SpecNameRow(name=str(port.name or ""), placeholder="port name", show_eye=True)
        row.setProperty("_port", port)
        row.edit_clicked.connect(lambda: self._edit_data(row))
        row.delete_clicked.connect(lambda: self._delete_row(row))
        row.name_committed.connect(lambda v: self._rename_data(row, v))
        row.show_on_node_changed.connect(lambda v: self._toggle_data_show_on_node(row, bool(v)))  # type: ignore[attr-defined]
        row.setToolTip(self._data_tooltip(port))
        row.setProperty("_port_dir", "data_in" if is_in else "data_out")
        editable = bool(self._editable_data_in if is_in else self._editable_data_out)
        # Even when spec ports are not editable, allow opening the dialog to edit UI-only fields (showOnNode).
        row.set_row_editable(allow_rename=editable, allow_delete=editable, allow_edit=True)
        show = bool(self._node.data_port_show_on_node(str(port.name or ""), is_in=bool(is_in)))  # type: ignore[attr-defined]
        row.set_show_on_node(bool(show))
        return row

    def _toggle_data_show_on_node(self, row: _F8SpecNameRow, show_on_node: bool) -> None:
        dir_s = str(row.property("_port_dir") or "")
        is_in = dir_s == "data_in"
        port = row.property("_port")
        name = ""
        if isinstance(port, F8DataPortSpec):
            name = str(port.name or "")
        if not name:
            name = str(row.name_edit.text() or "").strip()
        self._apply_data_port_ui_override(name, bool(show_on_node), is_in=bool(is_in))
        row.set_show_on_node(bool(show_on_node))

    def _data_tooltip(self, port: F8DataPortSpec) -> str:
        req = bool(port.required)
        desc = str(port.description or "").strip()
        vs = port.valueSchema
        t = _schema_type(vs)
        parts = [f"required={req}", f"type={t or 'unknown'}"]
        if desc:
            parts.append(desc)
        return "\n".join(parts)

    def _edit_exec(self, row: _F8SpecNameRow) -> None:
        dir_s = str(row.property("_port_dir") or "")
        if (dir_s == "exec_in" and not self._editable_exec_in) or (dir_s == "exec_out" and not self._editable_exec_out):
            return
        dlg = _F8EditExecPortDialog(self, title="Edit exec port", name=row.name_edit.text())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        row.name_edit.setText(dlg.name())
        self._commit()

    def _edit_data(self, row: _F8SpecNameRow) -> None:
        dir_s = str(row.property("_port_dir") or "")
        ui_only = bool((dir_s == "data_in" and not self._editable_data_in) or (dir_s == "data_out" and not self._editable_data_out))
        port = row.property("_port")
        if not isinstance(port, F8DataPortSpec):
            port = F8DataPortSpec(
                name=row.name_edit.text(), required=True, valueSchema=_schema_from_json_obj({"type": "any"})
            )
        dlg = _F8EditDataPortDialog(self, title="Edit data port", port=port, ui_only=ui_only)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_port = dlg.port()
        if ui_only:
            show_on_node = bool(new_port.showOnNode)
            self._apply_data_port_ui_override(str(port.name or ""), bool(show_on_node), is_in=(dir_s == "data_in"))
            self._load_from_spec()
            return
        row.setProperty("_port", new_port)
        row.name_edit.setText(str(new_port.name or ""))
        row.setToolTip(self._data_tooltip(new_port))
        self._commit()

    def _apply_data_port_ui_override(self, name: str, show_on_node: bool, *, is_in: bool) -> None:
        node = self._node
        if node is None:
            return
        spec = node.spec
        base_show = _base_data_port_show_on_node(spec, name=str(name or "").strip(), is_in=bool(is_in))
        _set_data_port_show_on_node_override(
            node,
            name=str(name or "").strip(),
            is_in=bool(is_in),
            show_on_node=bool(show_on_node),
            base_show_on_node=bool(base_show),
        )

    def _rename_data(self, row: _F8SpecNameRow, name: str) -> None:
        dir_s = str(row.property("_port_dir") or "")
        if (dir_s == "data_in" and not self._editable_data_in) or (dir_s == "data_out" and not self._editable_data_out):
            return
        port = row.property("_port")
        if not isinstance(port, F8DataPortSpec):
            port = F8DataPortSpec(name=name, required=True, valueSchema=_schema_from_json_obj({"type": "any"}))
        else:
            port = port.model_copy(deep=True)
            port.name = name
        row.setProperty("_port", port)
        row.setToolTip(self._data_tooltip(port))
        self._commit()

    def _delete_row(self, row: QtWidgets.QWidget) -> None:
        dir_s = str(row.property("_port_dir") or "")
        if dir_s == "exec_in" and not self._editable_exec_in:
            return
        if dir_s == "exec_out" and not self._editable_exec_out:
            return
        if dir_s == "data_in" and not self._editable_data_in:
            return
        if dir_s == "data_out" and not self._editable_data_out:
            return
        row.setParent(None)
        row.deleteLater()
        self._commit()

    def _add_exec(self, is_in: bool) -> None:
        if not (self._editable_exec_in if is_in else self._editable_exec_out):
            return
        row = self._make_exec_row("", is_in=is_in)
        (self._sec_exec_in if is_in else self._sec_exec_out).add_row(row)
        row.name_edit.setFocus()

    def _add_data(self, is_in: bool) -> None:
        if not (self._editable_data_in if is_in else self._editable_data_out):
            return
        port = F8DataPortSpec(
            name="", required=True, description=None, valueSchema=_schema_from_json_obj({"type": "any"})
        )
        row = self._make_data_row(port, is_in=is_in)
        (self._sec_data_in if is_in else self._sec_data_out).add_row(row)
        self._edit_data(row)

    def _commit(self) -> None:
        if self._node is None:
            return
        spec = self._node.spec

        exec_in: list[str] = []
        exec_out: list[str] = []
        if isinstance(spec, F8OperatorSpec):
            for r in self._sec_exec_in.rows():
                name = str(r.name_edit.text() or "").strip()
                if name:
                    exec_in.append(name)
            for r in self._sec_exec_out.rows():
                name = str(r.name_edit.text() or "").strip()
                if name:
                    exec_out.append(name)

        data_in: list[F8DataPortSpec] = []
        data_out: list[F8DataPortSpec] = []
        for r in self._sec_data_in.rows():
            port = r.property("_port")
            if isinstance(port, F8DataPortSpec) and str(port.name or "").strip():
                data_in.append(port)
        for r in self._sec_data_out.rows():
            port = r.property("_port")
            if isinstance(port, F8DataPortSpec) and str(port.name or "").strip():
                data_out.append(port)

        spec2 = _spec_set_ports(spec, data_in=data_in, data_out=data_out, exec_in=exec_in, exec_out=exec_out)
        if spec2 is not spec:
            self._node.spec = spec2

        if self._on_apply:
            self._on_apply()
        self.spec_applied.emit()


class _F8EditCommandParamDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title: str, param: F8CommandParam, ui_only: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        try:
            self._schema = param.valueSchema or _schema_from_json_obj({"type": "any"})
        except Exception:
            self._schema = _schema_from_json_obj({"type": "any"})
        self._ui_only = bool(ui_only)

        self._name = QtWidgets.QLineEdit(str(param.name or ""))
        self._name.setClearButtonEnabled(True)

        self._required = QtWidgets.QCheckBox()
        self._required.setChecked(bool(param.required))

        self._label = QtWidgets.QLineEdit(str(param.label or ""))
        self._label.setClearButtonEnabled(True)
        self._desc = QtWidgets.QPlainTextEdit(str(param.description or ""))
        self._ui_control = QtWidgets.QLineEdit(str(param.uiControl or ""))
        self._ui_control.setClearButtonEnabled(True)

        self._schema_summary = QtWidgets.QLabel("")
        self._schema_summary.setStyleSheet("color: #888;")
        self._refresh_schema_summary()

        schema_btn = QtWidgets.QPushButton("Edit Schema…")
        schema_btn.clicked.connect(self._edit_schema)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self._name)
        form.addRow("Required", self._required)
        form.addRow("Label", self._label)
        form.addRow("Description", self._desc)
        form.addRow("uiControl", self._ui_control)

        schema_row = QtWidgets.QHBoxLayout()
        schema_row.addWidget(self._schema_summary, 1)
        schema_row.addWidget(schema_btn)
        form.addRow("valueSchema", schema_row)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if self._ui_only:
            for w in (self._name, self._required, schema_btn):
                try:
                    w.setEnabled(False)
                except Exception:
                    pass

    def _refresh_schema_summary(self) -> None:
        t = _schema_type(self._schema)
        self._schema_summary.setText(t or "unknown")

    def _edit_schema(self) -> None:
        init = _schema_to_json_obj(self._schema) or {"type": "any"}
        dlg = _F8JsonEditorDialog(self, title="Edit valueSchema", value=init)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            self._schema = _schema_from_json_obj(dlg.value() or {"type": "any"})
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid schema", str(e))
            return
        self._refresh_schema_summary()

    def param(self) -> F8CommandParam:
        name = str(self._name.text() or "").strip()
        required = bool(self._required.isChecked())
        label = str(self._label.text() or "").strip() or None
        desc = str(self._desc.toPlainText() or "").strip() or None
        ui_control = str(self._ui_control.text() or "").strip() or None
        return F8CommandParam(name=name, required=required, label=label, description=desc, uiControl=ui_control, valueSchema=self._schema)


class _F8EditCommandDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title: str, cmd: F8Command, ui_only: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._ui_only = bool(ui_only)
        self._params: list[F8CommandParam] = list(cmd.params or [])

        self._name = QtWidgets.QLineEdit(str(cmd.name or ""))
        self._name.setClearButtonEnabled(True)
        self._desc = QtWidgets.QPlainTextEdit(str(cmd.description or ""))
        self._show_on_node = QtWidgets.QCheckBox()
        self._show_on_node.setChecked(bool(cmd.showOnNode))

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self._name)
        form.addRow("Show On Node", self._show_on_node)
        form.addRow("Description", self._desc)

        self._params_list = QtWidgets.QListWidget()
        self._params_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._refresh_params_list()

        btn_add = QtWidgets.QPushButton("Add Param…")
        btn_edit = QtWidgets.QPushButton("Edit Param…")
        btn_del = QtWidgets.QPushButton("Delete Param")
        btn_add.clicked.connect(self._add_param)
        btn_edit.clicked.connect(self._edit_param)
        btn_del.clicked.connect(self._delete_param)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(btn_add)
        row.addWidget(btn_edit)
        row.addWidget(btn_del)
        row.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("Params"))
        layout.addWidget(self._params_list, 1)
        layout.addLayout(row)
        layout.addWidget(buttons)

        if self._ui_only:
            for w in (self._name, self._desc, btn_add, btn_edit, btn_del):
                try:
                    w.setEnabled(False)
                except Exception:
                    pass

    def _refresh_params_list(self) -> None:
        self._params_list.clear()
        for p in self._params:
            name = str(p.name or "")
            req = bool(p.required)
            item = QtWidgets.QListWidgetItem(f"{name}{' *' if req else ''}")
            item.setData(QtCore.Qt.UserRole, p)
            self._params_list.addItem(item)

    def _selected_index(self) -> int:
        row = int(self._params_list.currentRow())
        if row < 0 or row >= len(self._params):
            return -1
        return row

    def _add_param(self) -> None:
        if self._ui_only:
            return
        dlg = _F8EditCommandParamDialog(self, title="Add command param", param=F8CommandParam(name="", valueSchema=_schema_from_json_obj({"type": "any"})))
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        p = dlg.param()
        if not str(p.name or "").strip():
            return
        self._params.append(p)
        self._refresh_params_list()

    def _edit_param(self) -> None:
        if self._ui_only:
            return
        idx = self._selected_index()
        if idx < 0:
            return
        dlg = _F8EditCommandParamDialog(self, title="Edit command param", param=self._params[idx])
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._params[idx] = dlg.param()
        self._refresh_params_list()

    def _delete_param(self) -> None:
        if self._ui_only:
            return
        idx = self._selected_index()
        if idx < 0:
            return
        self._params.pop(idx)
        self._refresh_params_list()

    def command(self) -> F8Command:
        name = str(self._name.text() or "").strip()
        desc = str(self._desc.toPlainText() or "").strip() or None
        show = bool(self._show_on_node.isChecked())
        return F8Command(name=name, description=desc, showOnNode=show, params=list(self._params))


class _F8CommandRow(QtWidgets.QWidget):
    invoke_clicked = QtCore.Signal(str)
    edit_clicked = QtCore.Signal(str)
    delete_clicked = QtCore.Signal(str)
    show_on_node_changed = QtCore.Signal(bool)

    def __init__(
        self,
        parent=None,
        *,
        name: str,
        description: str,
        allow_edit: bool,
        allow_delete: bool,
        show_on_node: bool,
    ):
        super().__init__(parent)
        self._name = str(name or "")
        self._base_tooltip = str(description or "").strip()

        self._btn_invoke = QtWidgets.QPushButton(self._name)
        self._btn_invoke.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._btn_invoke.clicked.connect(self._on_invoke_clicked)

        self._eye_btn = QtWidgets.QToolButton()
        self._eye_btn.setAutoRaise(True)
        self._eye_btn.setCheckable(True)
        self._eye_btn.setToolTip("Show on node")
        self._eye_btn.toggled.connect(self._on_eye_toggled)  # type: ignore[attr-defined]

        self._btn_edit = QtWidgets.QToolButton()
        self._btn_edit.setAutoRaise(True)
        self._btn_edit.setToolTip("Edit command…")
        self._btn_edit.setIcon(_icon_from_style(self._btn_edit, QtWidgets.QStyle.SP_FileDialogDetailedView, "document-edit"))
        self._btn_edit.setEnabled(bool(allow_edit))
        self._btn_edit.setVisible(True)
        self._btn_edit.clicked.connect(self._on_edit_clicked)

        self._btn_del = QtWidgets.QToolButton()
        self._btn_del.setAutoRaise(True)
        self._btn_del.setToolTip("Delete command")
        self._btn_del.setIcon(_icon_from_style(self._btn_del, QtWidgets.QStyle.SP_TrashIcon, "edit-delete"))
        self._btn_del.setEnabled(bool(allow_delete))
        self._btn_del.setVisible(bool(allow_delete))
        self._btn_del.clicked.connect(self._on_delete_clicked)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._btn_invoke, 1)
        layout.addWidget(self._btn_edit, 0)
        layout.addWidget(self._eye_btn, 0)
        layout.addWidget(self._btn_del, 0)

        if self._base_tooltip:
            self._btn_invoke.setToolTip(self._base_tooltip)
            self._btn_edit.setToolTip("Edit command…\n" + self._base_tooltip)

        self.set_show_on_node(bool(show_on_node))

    def set_show_on_node(self, show: bool) -> None:
        with QtCore.QSignalBlocker(self._eye_btn):
            self._eye_btn.setChecked(bool(show))
        self._update_eye_icon(bool(show))

    def _update_eye_icon(self, show: bool) -> None:
        icon_name = "fa5s.eye" if bool(show) else "fa5s.eye-slash"
        self._eye_btn.setIcon(qta.icon(icon_name, color="white"))

    def _on_eye_toggled(self, checked: bool) -> None:
        self._update_eye_icon(bool(checked))
        self.show_on_node_changed.emit(bool(checked))

    def set_invoke_enabled(self, enabled: bool, *, disabled_reason: str = "Service not running") -> None:
        """
        Enable/disable the invoke button (eg. based on service process running state).
        """
        en = bool(enabled)
        try:
            self._btn_invoke.setEnabled(en)
        except Exception:
            return
        if en:
            if self._base_tooltip:
                try:
                    self._btn_invoke.setToolTip(self._base_tooltip)
                except Exception:
                    pass
            return
        msg = str(disabled_reason or "").strip() or "Service not running"
        tip = (self._base_tooltip + "\n" + msg) if self._base_tooltip else msg
        try:
            self._btn_invoke.setToolTip(tip)
        except Exception:
            pass

    def _on_invoke_clicked(self, _checked: bool = False) -> None:
        self.invoke_clicked.emit(self._name)

    def _on_edit_clicked(self, _checked: bool = False) -> None:
        self.edit_clicked.emit(self._name)

    def _on_delete_clicked(self, _checked: bool = False) -> None:
        self.delete_clicked.emit(self._name)


class _F8SpecCommandEditor(QtWidgets.QWidget):
    def __init__(self, parent=None, *, node: Any, on_apply: Callable[[], None] | None):
        super().__init__(parent)
        self._node = node
        self._on_apply = on_apply
        self._bridge_proc_hooked = False
        self._cmd_rows: dict[str, _F8CommandRow] = {}

        self._sec = _F8SpecListSection(title="Commands")
        self._sec.add_clicked.connect(self._add_command)

        content = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(content)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(8)
        v.addWidget(self._sec)
        v.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(content)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self._load()

    def _bridge(self) -> Any | None:
        try:
            g = self._node.graph
        except Exception:
            return None
        try:
            return g.service_bridge
        except Exception:
            return None

    def _service_id(self) -> str:
        try:
            return str(self._node.id or "").strip()
        except Exception:
            return ""

    def _ensure_bridge_process_hook(self) -> None:
        if self._bridge_proc_hooked:
            return
        bridge = self._bridge()
        if bridge is None:
            return
        try:
            bridge.service_process_state.connect(self._on_bridge_service_process_state)  # type: ignore[attr-defined]
            self._bridge_proc_hooked = True
        except Exception:
            self._bridge_proc_hooked = False

    def _is_service_running(self) -> bool:
        bridge = self._bridge()
        sid = self._service_id()
        if bridge is None or not sid:
            return False
        try:
            return bool(bridge.is_service_running(sid))
        except Exception:
            return False

    @QtCore.Slot(str, bool)
    def _on_bridge_service_process_state(self, service_id: str, running: bool) -> None:
        if str(service_id or "").strip() != self._service_id():
            return
        self._apply_running_state(bool(running))

    def _apply_running_state(self, running: bool) -> None:
        enabled = bool(running)
        for row in list(self._cmd_rows.values()):
            try:
                row.set_invoke_enabled(enabled)
            except Exception:
                continue

    def _load(self) -> None:
        self._ensure_bridge_process_hook()
        self._sec.clear()
        self._cmd_rows = {}
        try:
            spec = self._node.spec
        except Exception:
            spec = None
        if not isinstance(spec, F8ServiceSpec):
            self._sec.set_add_visible(False)
            return
        try:
            editable = bool(spec.editableCommands)  # type: ignore[attr-defined]
        except Exception:
            editable = _extra_bool(spec, "editableCommands", False)
        self._sec.set_add_visible(bool(editable))

        running = self._is_service_running()
        try:
            cmds = list(self._node.effective_commands() or [])
        except Exception:
            cmds = list(spec.commands or [])
        for c in cmds:
            try:
                name = str(c.name or "")
            except Exception:
                name = ""
            if not name:
                continue
            try:
                desc = str(c.description or "")
            except Exception:
                desc = ""
            try:
                show_on_node = bool(c.showOnNode)
            except Exception:
                show_on_node = False
            row = _F8CommandRow(
                name=name,
                description=desc,
                allow_edit=True,
                allow_delete=editable,
                show_on_node=bool(show_on_node),
            )
            row.invoke_clicked.connect(self._invoke_command)
            row.edit_clicked.connect(self._edit_command)
            row.delete_clicked.connect(self._delete_command)
            row.show_on_node_changed.connect(lambda v, _n=str(name): self._toggle_command_show_on_node(_n, bool(v)))  # type: ignore[attr-defined]
            try:
                row.set_invoke_enabled(bool(running))
            except Exception:
                pass
            self._cmd_rows[str(name)] = row
            self._sec.add_row(row)

    def _toggle_command_show_on_node(self, name: str, show_on_node: bool) -> None:
        n = str(name or "").strip()
        if not n:
            return
        self._apply_command_ui_override(n, bool(show_on_node))
        row = self._cmd_rows.get(n)
        if row is not None:
            row.set_show_on_node(bool(show_on_node))

    def _prompt_command_args(self, cmd: F8Command) -> dict[str, Any] | None:
        params = list(cmd.params or [])
        if not params:
            return {}
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(str(cmd.name or "Command"))
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)

        editors: dict[str, Callable[[], Any]] = {}
        widgets: dict[str, QtWidgets.QWidget] = {}

        for p in params:
            pname = str(p.name or "").strip()
            if not pname:
                continue
            required = bool(p.required)
            ui = str(p.uiControl or "").strip().lower()
            schema = p.valueSchema
            t = _schema_type(schema) if schema is not None else ""
            enum_items = _schema_enum_items(schema) if schema is not None else []
            lo, hi = _schema_numeric_range(schema) if schema is not None else (None, None)
            if isinstance(schema, F8DataTypeSchema):
                default_value = schema_default(schema)
            else:
                try:
                    default_value = schema.root.default
                except Exception:
                    default_value = None

            label = f"{pname} *" if required else pname
            tooltip = str(p.description or "").strip()

            if enum_items or ui in {"select", "dropdown", "dropbox", "combo", "combobox"}:
                combo = F8OptionCombo()
                items = [str(x) for x in enum_items]
                combo.set_options(items, labels=items)
                if tooltip:
                    combo.set_context_tooltip(tooltip)
                if default_value is not None:
                    combo.set_value(str(default_value))
                widgets[pname] = combo
                editors[pname] = lambda _c=combo: _c.value()
                form.addRow(label, combo)
                continue

            if t == "boolean" or ui in {"switch", "toggle"}:
                sw = F8Switch()
                sw.set_labels("True", "False")
                if tooltip:
                    sw.setToolTip(tooltip)
                if default_value is not None:
                    sw.set_value(bool(default_value))
                widgets[pname] = sw
                editors[pname] = lambda _s=sw: bool(_s.value())
                form.addRow(label, sw)
                continue

            if t in {"integer", "number"} and ui == "slider":
                is_int = t == "integer"
                bar = F8ValueBar(integer=is_int, minimum=0.0, maximum=1.0)
                bar.set_range(lo, hi)
                if default_value is not None:
                    bar.set_value(default_value)
                widgets[pname] = bar
                editors[pname] = lambda _b=bar, _is_int=is_int: (int(_b.value()) if _is_int else float(_b.value()))
                form.addRow(label, bar)
                continue

            w = QtWidgets.QLineEdit()
            if default_value is not None:
                w.setText(str(default_value))
            widgets[pname] = w
            editors[pname] = lambda _w=w: str(_w.text() or "").strip()
            form.addRow(label, w)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        while True:
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return None
            args: dict[str, Any] = {}
            missing: list[str] = []
            for p in params:
                pname = str(p.name or "").strip()
                if not pname or pname not in editors:
                    continue
                required = bool(p.required)
                v = editors[pname]()
                if isinstance(v, str) and v.strip() == "":
                    v = None
                if required and v is None:
                    missing.append(pname)
                    continue
                if v is not None:
                    args[pname] = v
            if missing:
                QtWidgets.QMessageBox.warning(dlg, "Missing required fields", "Please fill: " + ", ".join(missing))
                continue
            return args

    def _invoke_command(self, name: str) -> None:
        try:
            spec = self._node.spec
        except Exception:
            spec = None
        if not isinstance(spec, F8ServiceSpec):
            return
        cmd = None
        for c in list(spec.commands or []):
            try:
                cname = str(c.name or "").strip()
            except Exception:
                continue
            if cname == str(name or "").strip():
                cmd = c
                break
        if cmd is None:
            return

        # Mirror NodeGraph behavior: commands are only invokable when the service is running.
        if not self._is_service_running():
            return

        # Allow node-specific UI to override command invocation (eg. open a custom dialog).
        if isinstance(self._node, CommandUiHandler):
            parent = None
            try:
                parent = self.window()
            except Exception:
                parent = None
            try:
                if bool(self._node.handle_command_ui(cmd, parent=parent, source=CommandUiSource.PROPERTIES_BIN)):
                    return
            except Exception:
                node_id = ""
                try:
                    node_id = str(self._node.id or "").strip()
                except Exception:
                    node_id = ""
                logger.exception("handle_command_ui failed command=%s nodeId=%s", name, node_id)

        bridge = self._bridge()
        sid = self._service_id()
        if bridge is None or not sid:
            return
        args = {}
        params = list(cmd.params or [])
        if params:
            args = self._prompt_command_args(cmd)
            if args is None:
                return
        try:
            bridge.invoke_remote_command(sid, str(cmd.name or ""), args or {})
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Command failed", str(e))

    def _add_command(self) -> None:
        try:
            spec = self._node.spec
        except Exception:
            spec = None
        if not isinstance(spec, F8ServiceSpec):
            return
        try:
            editable = bool(spec.editableCommands)  # type: ignore[attr-defined]
        except Exception:
            editable = _extra_bool(spec, "editableCommands", False)
        if not editable:
            return
        cmd = F8Command(name="", description=None, showOnNode=False, params=[])
        dlg = _F8EditCommandDialog(self, title="Add command", cmd=cmd, ui_only=False)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_cmd = dlg.command()
        if not str(new_cmd.name or "").strip():
            return
        spec2 = _spec_add_command(spec, cmd=new_cmd)
        if spec2 is not spec:
            self._node.spec = spec2
        if self._on_apply:
            self._on_apply()
        self._load()

    def _edit_command(self, name: str) -> None:
        try:
            spec = self._node.spec
        except Exception:
            spec = None
        if not isinstance(spec, F8ServiceSpec):
            return
        try:
            editable = bool(spec.editableCommands)  # type: ignore[attr-defined]
        except Exception:
            editable = _extra_bool(spec, "editableCommands", False)
        cmds = list(spec.commands or [])
        idx = -1
        for i, c in enumerate(cmds):
            try:
                cname = str(c.name or "").strip()
            except Exception:
                continue
            if cname == str(name or "").strip():
                idx = i
                break
        if idx < 0:
            return
        # If not editable, only allow UI override edits (showOnNode).
        # Apply current UI override to dialog initial state (best-effort).
        init_cmd = cmds[idx]
        if not editable:
            try:
                for c in list(self._node.effective_commands() or []):
                    try:
                        if str(c.name or "").strip() == str(name or "").strip():
                            init_cmd = c
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        dlg = _F8EditCommandDialog(self, title="Edit command", cmd=init_cmd, ui_only=not editable)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        edited = dlg.command()
        if editable:
            spec2 = _spec_replace_command(spec, name=str(name or "").strip(), cmd=edited)
            if spec2 is not spec:
                self._node.spec = spec2
            if self._on_apply:
                self._on_apply()
        else:
            self._apply_command_ui_override(str(init_cmd.name or ""), bool(edited.showOnNode))
        self._load()

    def _apply_command_ui_override(self, name: str, show_on_node: bool) -> None:
        n = str(name or "").strip()
        if not n:
            return
        node = self._node
        try:
            spec = node.spec
        except Exception:
            spec = None
        base_show = _base_command_show_on_node(spec, name=n)
        _set_command_show_on_node_override(node, name=n, show_on_node=bool(show_on_node), base_show_on_node=bool(base_show))

    def _delete_command(self, name: str) -> None:
        try:
            spec = self._node.spec
        except Exception:
            spec = None
        if not isinstance(spec, F8ServiceSpec):
            return
        try:
            editable = bool(spec.editableCommands)  # type: ignore[attr-defined]
        except Exception:
            editable = _extra_bool(spec, "editableCommands", False)
        if not editable:
            return
        n = str(name or "").strip()
        if QtWidgets.QMessageBox.question(self, "Delete command", f"Delete '{n}'?") != QtWidgets.QMessageBox.Yes:
            return
        spec2 = _spec_delete_command(spec, name=n)
        if spec2 is not spec:
            self._node.spec = spec2
        if self._on_apply:
            self._on_apply()
        self._load()


class F8StudioNodePropEditorWidget(QtWidgets.QWidget):
    """
    Node properties editor widget for display a Node object.

    Args:
        parent (QtWidgets.QWidget): parent object.
        node (NodeGraphQt.NodeObject): node.
    """

    #: signal (node_id, prop_name, prop_value)
    property_changed = QtCore.Signal(str, str, object)
    property_closed = QtCore.Signal(str)

    def __init__(self, parent=None, node=None):
        super(F8StudioNodePropEditorWidget, self).__init__(parent)
        self._node = node
        self.__node_id = node.id
        self.__tab_windows = {}
        self.__tab = QtWidgets.QTabWidget()
        self._option_pool_dependents: dict[str, list[Any]] = {}
        self._reload_pending = False
        self._reload_debounce_ms = 50
        self._reload_timer = QtCore.QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.timeout.connect(self._reload_now)

        close_btn = QtWidgets.QPushButton()
        close_btn.setIcon(QtGui.QIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton)))
        close_btn.setMaximumWidth(40)
        close_btn.setToolTip("close property")
        close_btn.clicked.connect(self._on_close)

        pixmap = QtGui.QPixmap()
        if node.icon():
            pixmap = QtGui.QPixmap(node.icon())

            if pixmap.size().height() > NodeEnum.ICON_SIZE.value:
                pixmap = pixmap.scaledToHeight(NodeEnum.ICON_SIZE.value, QtCore.Qt.SmoothTransformation)
            if pixmap.size().width() > NodeEnum.ICON_SIZE.value:
                pixmap = pixmap.scaledToWidth(NodeEnum.ICON_SIZE.value, QtCore.Qt.SmoothTransformation)

        self.icon_label = QtWidgets.QLabel(self)
        self.icon_label.setPixmap(pixmap)
        self.icon_label.setStyleSheet("background: transparent;")

        self.name_wgt = PropLineEdit()
        self.name_wgt.set_name("name")
        self.name_wgt.setToolTip("name\nSet the node name.")
        self.name_wgt.set_value(node.name())
        self.name_wgt.value_changed.connect(self._on_property_changed)

        self.type_wgt = QtWidgets.QLabel(node.type_)
        self.type_wgt.setAlignment(QtCore.Qt.AlignRight)
        self.type_wgt.setToolTip("type_\nNode type identifier followed by the class name.")
        font = self.type_wgt.font()
        font.setPointSize(10)
        self.type_wgt.setFont(font)

        name_layout = QtWidgets.QHBoxLayout()
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.addWidget(self.icon_label)
        name_layout.addWidget(QtWidgets.QLabel("name"))
        name_layout.addWidget(self.name_wgt)
        name_layout.addWidget(close_btn)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.addLayout(name_layout)
        layout.addWidget(self.__tab)
        layout.addWidget(self.type_wgt)

        self._port_connections = self._read_node(node)

    def __repr__(self):
        return "<{} object at {}>".format(self.__class__.__name__, hex(id(self)))

    def _on_close(self):
        """
        called by the close button.
        """
        self.property_closed.emit(self.__node_id)

    def _on_property_changed(self, name, value):
        """
        slot function called when a property widget has changed.

        Args:
            name (str): property name.
            value (object): new value.
        """
        self.property_changed.emit(self.__node_id, name, value)
        self.refresh_option_pool(str(name or ""))

    def refresh_option_pool(self, pool_name: str) -> None:
        """
        Refresh option widgets that depend on a given pool state field.
        """
        pool = str(pool_name or "").strip()
        if not pool:
            return
        if pool not in self._option_pool_dependents:
            return
        for w in list(self._option_pool_dependents.get(pool) or []):
            try:
                w.refresh_options()
            except Exception:
                continue

    def open_state_field_editor(self, field_name: str) -> None:
        """
        Open the edit dialog for a state field and apply changes.
        """
        name = str(field_name or "").strip()
        if not name:
            return
        node = self._node
        if node is None:
            return
        spec = _get_node_spec(node)
        if spec is None:
            return

        # Find current effective field + base field.
        eff_fields = _effective_state_fields(node)
        if not eff_fields:
            try:
                eff_fields = list(spec.stateFields or [])
            except Exception:
                eff_fields = []
        current = None
        for f in eff_fields:
            try:
                if str(f.name or "").strip() == name:
                    current = f
                    break
            except Exception:
                continue
        if current is None:
            return

        try:
            editable = bool(spec.editableStateFields)  # type: ignore[attr-defined]
        except Exception:
            editable = _extra_bool(spec, "editableStateFields", False)
        try:
            required = bool(current.required)
        except Exception:
            required = False
        ui_only = (not editable) or required

        # If UI-only, we still want to allow editing UI fields (showOnNode/uiControl/etc).
        dlg = _F8EditStateFieldDialog(self, title="Edit state field", field=current, ui_only=ui_only)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_field = dlg.field()

        if ui_only:
            self._apply_state_field_ui_override(name, new_field)
        else:
            self._apply_state_field_spec_replace(name, new_field)

        self._refresh()

    def add_state_field(self) -> None:
        node = self._node
        if node is None:
            return
        spec = _get_node_spec(node)
        if not isinstance(spec, (F8ServiceSpec, F8OperatorSpec)):
            return
        try:
            editable = bool(spec.editableStateFields)  # type: ignore[attr-defined]
        except Exception:
            editable = _extra_bool(spec, "editableStateFields", False)
        if not editable:
            return
        field = F8StateSpec(name="", valueSchema=_schema_from_json_obj({"type": "any"}), access=F8StateAccess.rw)
        dlg = _F8EditStateFieldDialog(self, title="Add state field", field=field, ui_only=False)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_field = dlg.field()
        if not str(new_field.name or "").strip():
            return
        self._apply_state_field_spec_add(new_field)
        self._refresh()

    def delete_state_field(self, field_name: str) -> None:
        name = str(field_name or "").strip()
        if not name:
            return
        node = self._node
        if node is None:
            return
        spec = _get_node_spec(node)
        if not isinstance(spec, (F8ServiceSpec, F8OperatorSpec)):
            return
        try:
            editable = bool(spec.editableStateFields)  # type: ignore[attr-defined]
        except Exception:
            editable = _extra_bool(spec, "editableStateFields", False)
        if not editable:
            return
        # required fields are protected
        eff = _effective_state_fields(node)
        if not eff:
            try:
                eff = list(spec.stateFields or [])
            except Exception:
                eff = []
        for f in eff:
            try:
                if str(f.name or "").strip() == name and bool(f.required):
                    return
            except Exception:
                continue
        if QtWidgets.QMessageBox.question(self, "Delete state field", f"Delete '{name}'?") != QtWidgets.QMessageBox.Yes:
            return
        self._apply_state_field_spec_delete(name)
        self._refresh()

    def _apply_state_field_spec_replace(self, old_name: str, new_field: F8StateSpec) -> None:
        node = self._node
        if node is None:
            return
        spec = _get_node_spec(node)
        if spec is None:
            return
        spec2 = _spec_replace_state_field(spec, old_name=old_name, new_field=new_field)
        if spec2 is not spec:
            node.spec = spec2
        self._resync_node_from_spec()

    def _apply_state_field_spec_add(self, new_field: F8StateSpec) -> None:
        node = self._node
        if node is None:
            return
        spec = _get_node_spec(node)
        if spec is None:
            return
        spec2 = _spec_add_state_field(spec, field=new_field)
        if spec2 is not spec:
            node.spec = spec2
        self._resync_node_from_spec()

    def _apply_state_field_spec_delete(self, name: str) -> None:
        node = self._node
        if node is None:
            return
        spec = _get_node_spec(node)
        if spec is None:
            return
        spec2 = _spec_delete_state_field(spec, name=name)
        if spec2 is not spec:
            node.spec = spec2
        self._resync_node_from_spec()

    def _apply_state_field_ui_override(self, name: str, edited: F8StateSpec) -> None:
        node = self._node
        if node is None:
            return
        spec = _get_node_spec(node)
        base = _find_base_state_field(spec, name=name) if spec is not None else None
        _set_state_field_ui_override(node, field_name=name, base=base or edited, edited=edited)

    def _toggle_state_field_show_on_node(self, field_name: str, show_on_node: bool) -> None:
        node = self._node
        if node is None:
            return
        name = str(field_name or "").strip()
        if not name:
            return
        spec = _get_node_spec(node)
        base = _find_base_state_field(spec, name=name) if spec is not None else None
        if base is None:
            base = F8StateSpec(name=name, valueSchema=_schema_from_json_obj({"type": "any"}), access=F8StateAccess.rw)
        edited = base.model_copy(deep=True)
        edited.showOnNode = bool(show_on_node)
        self._apply_state_field_ui_override(name, edited)

    def _resync_node_from_spec(self) -> None:
        node = self._node
        if node is None:
            return
        try:
            node.sync_from_spec()
        except Exception:
            return

    def _read_node(self, node):
        """
        Populate widget from a node.

        Args:
            node (NodeGraphQt.BaseNode): node class.

        Returns:
            _PortConnectionsContainer: ports container widget.
        """
        model = node.model
        graph_model = node.graph.model

        common_props = graph_model.get_node_common_properties(node.type_)

        # sort tabs and properties.
        tab_mapping = defaultdict(list)
        for prop_name, prop_val in model.custom_properties.items():
            tab_name = model.get_tab_name(prop_name)
            tab_mapping[tab_name].append((prop_name, prop_val))

        # add tabs.
        reserved_tabs = ["Node", "Port", "Commands"]
        for tab in sorted(tab_mapping.keys()):
            if tab in reserved_tabs:
                print('tab name "{}" is reserved by the "NodePropWidget" ' "please use a different tab name.")
                continue
            self.add_tab(tab)

        # property widget factory.
        widget_factory = NodePropertyWidgetFactory()

        # populate tab properties.
        for tab in sorted(tab_mapping.keys()):
            prop_window = self.__tab_windows[tab]
            if tab == "State" and isinstance(prop_window, _F8StateStackContainer):
                # Build the State tab from stateFields so we can attach edit/delete/add UI.
                spec = _get_node_spec(node)
                if spec is None:
                    editable_state = False
                else:
                    try:
                        editable_state = bool(spec.editableStateFields)  # type: ignore[attr-defined]
                    except Exception:
                        editable_state = _extra_bool(spec, "editableStateFields", False)
                prop_window.set_add_visible(bool(editable_state))
                # Map property values.
                values = dict(model.custom_properties)
                # Order by effective state fields (applies UI overrides).
                eff_fields = _effective_state_fields(node)
                if not eff_fields and spec is not None:
                    try:
                        eff_fields = list(spec.stateFields or [])
                    except Exception:
                        eff_fields = []
                for f in eff_fields:
                    try:
                        name = str(f.name or "").strip()
                    except Exception:
                        name = ""
                    if not name:
                        continue
                    if name not in values:
                        continue
                    value = values.get(name)
                    wid_type = model.get_widget_type(name)
                    if wid_type == 0:
                        continue
                    widget = _build_state_value_widget(
                        node=node,
                        prop_name=name,
                        widget_type=wid_type,
                        widget_factory=widget_factory,
                        register_option_pool_dependent=lambda pool, w: self._option_pool_dependents.setdefault(pool, []).append(w),
                    )

                    tooltip = None
                    if name in common_props.keys() and "tooltip" in common_props[name].keys():
                        tooltip = common_props[name]["tooltip"]
                    access = _state_field_access(node, name)
                    read_only = access == F8StateAccess.ro or _state_input_is_connected(node, name)
                    _set_read_only_widget(widget, read_only=bool(read_only))
                    # Delete is only allowed when editableStateFields and not required.
                    required = bool(f.required)
                    allow_delete = bool(editable_state and not required)
                    label_txt = str(f.label or "").strip()
                    desc_txt = str(f.description or "").strip()
                    show_on_node = bool(f.showOnNode)
                    prop_window.add_widget(
                        name=name,
                        widget=widget,
                        value=value,
                        label=(label_txt or name).replace("_", " "),
                        tooltip=desc_txt or tooltip,
                        allow_delete=allow_delete,
                        show_on_node=bool(show_on_node),
                    )
                    widget.value_changed.connect(self._on_property_changed)
                continue
            for prop_name, value in tab_mapping[tab]:
                wid_type = model.get_widget_type(prop_name)
                if wid_type == 0:
                    continue
                widget = _build_state_value_widget(
                    node=node,
                    prop_name=prop_name,
                    widget_type=wid_type,
                    widget_factory=widget_factory,
                    register_option_pool_dependent=lambda pool, w: self._option_pool_dependents.setdefault(pool, []).append(w),
                )

                tooltip = None
                if prop_name in common_props.keys():
                    if "items" in common_props[prop_name].keys():
                        widget.set_items(common_props[prop_name]["items"])
                    if "range" in common_props[prop_name].keys():
                        prop_range = common_props[prop_name]["range"]
                        try:
                            widget.set_min(prop_range[0])
                            widget.set_max(prop_range[1])
                        except Exception:
                            try:
                                widget.setMinimum(prop_range[0])
                                widget.setMaximum(prop_range[1])
                            except Exception:
                                pass
                    if "tooltip" in common_props[prop_name].keys():
                        tooltip = common_props[prop_name]["tooltip"]

                if wid_type == NodePropWidgetEnum.QTEXT_EDIT.value and _is_json_state_value(node, prop_name):
                    widget = _F8JsonPropTextEdit()
                    widget.set_name(prop_name)

                # Dialog-backed code editor (eg. python_script code).
                try:
                    ui_control_raw = _state_field_ui_control(node, prop_name)
                    ui_control = str(ui_control_raw or "").strip().lower()
                    spec = _get_node_spec(node)
                    is_legacy_python_script_code = (
                        isinstance(spec, F8OperatorSpec)
                        and str(spec.operatorClass or "") == "f8.python_script"
                        and str(prop_name) == "code"
                    )
                    if ui_control == "code" or is_legacy_python_script_code:
                        ui_language = _state_field_ui_language(node, prop_name)
                        widget = _F8CodeButtonPropWidget(title=f"{node.name()} — {prop_name}", language=ui_language or "plaintext")
                        widget.set_name(prop_name)
                except Exception:
                    pass
                access = _state_field_access(node, prop_name)
                if access == F8StateAccess.ro:
                    _apply_read_only_widget(widget)
                # Enrich tooltips for option/switch editors.
                if isinstance(widget, (F8PropOptionCombo, F8PropMultiSelect, F8PropBoolSwitch)):
                    desc = ""
                    for f in _effective_state_fields(node):
                        try:
                            if str(f.name or "").strip() == str(prop_name):
                                desc = str(f.description or "").strip()
                                break
                        except Exception:
                            continue
                    if desc:
                        try:
                            widget.set_context_tooltip(desc)
                        except AttributeError:
                            pass
                prop_window.add_widget(
                    name=prop_name, widget=widget, value=value, label=prop_name.replace("_", " "), tooltip=tooltip
                )
                widget.value_changed.connect(self._on_property_changed)

        # add "Node" tab properties. (default props)
        self.add_tab("Node")
        default_props = {
            "color": "Node base color.",
            "text_color": "Node text color.",
            "border_color": "Node border color.",
            "disabled": "Disable/Enable node state.",
            "id": "Unique identifier string to the node.",
        }
        prop_window = self.__tab_windows["Node"]
        for prop_name, tooltip in default_props.items():
            wid_type = model.get_widget_type(prop_name)
            widget = widget_factory.get_widget(wid_type)
            widget.set_name(prop_name)
            prop_window.add_widget(
                name=prop_name,
                widget=widget,
                value=model.get_property(prop_name),
                label=prop_name.replace("_", " "),
                tooltip=tooltip,
            )

            widget.value_changed.connect(self._on_property_changed)

        spec = _get_node_spec(node)
        if isinstance(spec, F8OperatorSpec):
            try:
                svc_id = str(node.svcId or "")  # type: ignore[attr-defined]
            except Exception:
                svc_id = ""
            sys_widget = PropLabel()
            sys_widget.set_name("__sys_svcId")
            prop_window.add_widget(
                name="__sys_svcId",
                widget=sys_widget,
                value=svc_id,
                label="svcId",
                tooltip="Bound service container id.",
            )

        self.type_wgt.setText(model.get_property("type_") or "")

        # built-in spec editors (if node has F8 spec).
        if isinstance(spec, (F8OperatorSpec, F8ServiceSpec)):
            if isinstance(spec, F8ServiceSpec):
                cmd_editor = _F8SpecCommandEditor(self, node=node, on_apply=self._on_spec_applied)
                self.__tab.addTab(cmd_editor, "Commands")
            spec_ports = _F8SpecPortEditor(self, node=node, on_apply=self._on_spec_applied)
            self.__tab.addTab(spec_ports, "Port")

        # hide/remove empty tabs with no property widgets.
        tab_index = {self.__tab.tabText(x): x for x in range(self.__tab.count())}
        current_idx = None
        for tab_name, prop_window in self.__tab_windows.items():
            prop_widgets = prop_window.get_all_widgets()
            if not prop_widgets:
                # I prefer to hide the tab but in older version of pyside this
                # attribute doesn't exist we'll just remove.
                try:
                    self.__tab.setTabVisible(tab_index[tab_name], False)
                except Exception:
                    self.__tab.removeTab(tab_index[tab_name])
                continue
            if current_idx is None:
                current_idx = tab_index[tab_name]

        # Order: State, Commands, Port, Node (Node last).
        _reorder_tabs(self.__tab, ["State", "Commands", "Port", "Node"])

        # Default tab: first existing among preferred, else 0.
        preferred_default = None
        for t in ["State", "Commands", "Port", "Node"]:
            for i in range(self.__tab.count()):
                if self.__tab.tabText(i) == t:
                    preferred_default = i
                    break
            if preferred_default is not None:
                break
        self.__tab.setCurrentIndex(preferred_default if preferred_default is not None else 0)

        return None

    def _on_spec_applied(self) -> None:
        node = self._node
        if node is None:
            return
        try:
            node.sync_from_spec()
        except Exception:
            pass
        self.reload()

    def _refresh(self) -> None:
        """
        Backwards-compatible alias for triggering a node sync + UI reload.

        Some editor actions (eg. state field schema edits) still call `_refresh()`.
        """
        self._on_spec_applied()

    def reload(self) -> None:
        """
        Coalesce multiple reload requests into a single UI rebuild.

        Some services update state at high frequency; rebuilding the entire
        properties UI per update can freeze the UI and exhaust native window
        handles if removed widgets are not properly released.
        """
        if self._reload_pending:
            return
        self._reload_pending = True
        # Debounced rebuild to coalesce bursts of updates.
        self._reload_timer.start(int(self._reload_debounce_ms))

    def _clear_tabs(self) -> None:
        # `removeTab()` does not delete the widget.
        # Avoid `setParent(None)` to prevent transient top-level window flashes.
        while self.__tab.count():
            w = self.__tab.widget(0)
            self.__tab.removeTab(0)
            if w is None:
                continue
            try:
                w.setVisible(False)
            except Exception:
                logger.exception("Failed to hide tab widget before deleteLater")
            w.deleteLater()

    def _reload_now(self) -> None:
        self._reload_pending = False
        node = self._node
        if node is None:
            return
        prev_tab = None
        scroll_pos: dict[str, int] = {}
        try:
            idx = self.__tab.currentIndex()
            if idx >= 0:
                prev_tab = self.__tab.tabText(idx)
        except Exception:
            prev_tab = None
        try:
            for i in range(self.__tab.count()):
                tab_name = self.__tab.tabText(i)
                w = self.__tab.widget(i)
                if not w:
                    continue
                areas = w.findChildren(QtWidgets.QScrollArea)
                if not areas:
                    continue
                try:
                    scroll_pos[tab_name] = int(areas[0].verticalScrollBar().value())
                except Exception:
                    pass
        except Exception:
            scroll_pos = {}

        self._clear_tabs()
        self.__tab_windows = {}
        self._option_pool_dependents = {}
        self._port_connections = self._read_node(node)
        if prev_tab:
            try:
                for i in range(self.__tab.count()):
                    if self.__tab.tabText(i) == prev_tab:
                        self.__tab.setCurrentIndex(i)
                        break
            except Exception:
                pass
        if scroll_pos:

            def _restore() -> None:
                for i in range(self.__tab.count()):
                    tab_name = self.__tab.tabText(i)
                    if tab_name not in scroll_pos:
                        continue
                    w = self.__tab.widget(i)
                    if not w:
                        continue
                    areas = w.findChildren(QtWidgets.QScrollArea)
                    if not areas:
                        continue
                    try:
                        areas[0].verticalScrollBar().setValue(scroll_pos[tab_name])
                    except Exception:
                        pass

            QtCore.QTimer.singleShot(0, _restore)

    def node_id(self):
        """
        Returns the node id linked to the widget.

        Returns:
            str: node id
        """
        return self.__node_id

    def add_widget(self, name, widget, tab="Properties"):
        """
        add new node property widget.

        Args:
            name (str): property name.
            widget (BaseProperty): property widget.
            tab (str): tab name.
        """
        if tab not in self.__tab_windows.keys():
            tab = "Properties"
        if tab not in self.__tab_windows.keys():
            self.add_tab(tab)
        window = self.__tab_windows[tab]
        window.add_widget(name, widget)
        widget.value_changed.connect(self._on_property_changed)

    def add_tab(self, name):
        """
        add a new tab.

        Args:
            name (str): tab name.

        Returns:
            PropListWidget: tab child widget.
        """
        if name in self.__tab_windows.keys():
            raise AssertionError("Tab name {} already taken!".format(name))
        window = _F8StateStackContainer(self) if name == "State" else _F8StateContainer(self)
        self.__tab_windows[name] = window
        if name == "State":
            assert isinstance(window, _F8StateStackContainer)
            window.edit_state_field_requested.connect(self.open_state_field_editor)
            window.delete_state_field_requested.connect(self.delete_state_field)
            window.add_state_field_requested.connect(self.add_state_field)
            window.toggle_state_field_show_on_node_requested.connect(self._toggle_state_field_show_on_node)
        self.__tab.addTab(window, name)
        return window

    def get_tab_widget(self):
        """
        Returns the underlying tab widget.

        Returns:
            QtWidgets.QTabWidget: tab widget.
        """
        return self.__tab

    def get_widget(self, name):
        """
        get property widget.

        Args:
            name (str): property name.

        Returns:
            NodeGraphQt.custom_widgets.properties_bin.prop_widgets_abstract.BaseProperty: property widget.
        """
        if name == "name":
            return self.name_wgt
        for prop_win in self.__tab_windows.values():
            widget = prop_win.get_widget(name)
            if widget:
                return widget

    def get_all_property_widgets(self):
        """
        get all the node property widgets.

        Returns:
            list[BaseProperty]: property widgets.
        """
        widgets = [self.name_wgt]
        for prop_win in self.__tab_windows.values():
            for widget in prop_win.get_all_widgets().values():
                widgets.append(widget)
        return widgets

    def get_port_connection_widget(self):
        """
        Returns the ports connections container widget.

        Returns:
            _PortConnectionsContainer: port container widget.
        """
        return self._port_connections

    def set_port_lock_widgets_disabled(self, disabled=True):
        """
        Enable/Disable port lock column widgets.

        Args:
            disabled (bool): true to disable checkbox.
        """
        return


class F8StudioPropertiesBinWidget(PropertiesBinWidget):
    """
    Customized Properties Bin Widget for F8PyStudio.
    """

    def __init__(self, parent=None, node_graph=None):
        super(F8StudioPropertiesBinWidget, self).__init__(parent=parent, node_graph=node_graph)

    def create_property_editor(self, node):
        return F8StudioNodePropEditorWidget(node=node)


class F8StudioSingleNodePropertiesWidget(QtWidgets.QWidget):
    """
    Single-node properties panel (no PropertiesBinWidget/QTableWidget).

    NodeGraphQt's PropertiesBinWidget hosts editors inside a QTableWidget, which
    can scroll-jump on focus/click. Since Studio only needs one active editor,
    we present a single `F8StudioNodePropEditorWidget` inside a QScrollArea.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, node_graph: Any) -> None:
        super().__init__(parent)
        self._node_graph = node_graph
        self._node_id: str | None = None
        self._editor: F8StudioNodePropEditorWidget | None = None
        self._block_signal = False
        self._last_node_click_ts: float = 0.0
        self._selection_timer = QtCore.QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.timeout.connect(self._apply_graph_selection)

        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QtWidgets.QWidget(self._scroll)
        self._container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self._container_layout = QtWidgets.QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(0)

        self._empty = QtWidgets.QLabel("Select a node to view properties.", self._container)
        self._empty.setAlignment(QtCore.Qt.AlignCenter)
        self._empty.setStyleSheet("color: rgba(235,235,235,140); padding: 14px;")
        self._container_layout.addWidget(self._empty, 1)

        self._scroll.setWidget(self._container)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll, 1)

        self._wire_graph_signals()
        QtCore.QTimer.singleShot(0, self._sync_container_width)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_container_width()

    def _sync_container_width(self) -> None:
        """
        Keep the content widget width aligned to the scroll viewport width.

        This prevents QScrollArea from showing a horizontal scrollbar due to
        the container/editor having a slightly larger size hint.
        """
        try:
            vp_w = int(self._scroll.viewport().width())
        except Exception:
            return
        if vp_w <= 0:
            return
        try:
            self._container.setMinimumWidth(vp_w)
        except Exception:
            pass

    def _wire_graph_signals(self) -> None:
        g = self._node_graph
        if g is None:
            return
        g.node_selected.connect(self._on_node_selected)  # type: ignore[attr-defined]
        g.node_double_clicked.connect(self._on_node_selected)  # type: ignore[attr-defined]
        g.node_selection_changed.connect(self._on_node_selection_changed)  # type: ignore[attr-defined]
        g.nodes_deleted.connect(self._on_nodes_deleted)  # type: ignore[attr-defined]
        g.property_changed.connect(self._on_graph_property_changed)  # type: ignore[attr-defined]
        g.port_connected.connect(self._on_graph_ports_changed)  # type: ignore[attr-defined]
        g.port_disconnected.connect(self._on_graph_ports_changed)  # type: ignore[attr-defined]

    def _on_graph_ports_changed(self, _in_port: Any, _out_port: Any) -> None:
        """
        Toggle read-only state for State-tab widgets when state-edge bindings change.
        """
        try:
            in_name = str(_in_port.name() or "")
            out_name = str(_out_port.name() or "")
        except (AttributeError, TypeError):
            return
        if not (in_name.startswith("[S]") or in_name.endswith("[S]") or out_name.startswith("[S]") or out_name.endswith("[S]")):
            return
        if self._editor is None or self._node_id is None:
            return
        g = self._node_graph
        if g is None:
            return
        node = g.get_node_by_id(self._node_id)  # type: ignore[attr-defined]
        if node is None:
            return
        spec = _get_node_spec(node)
        if spec is None:
            return
        eff_fields = _effective_state_fields(node)
        if not eff_fields:
            eff_fields = list(spec.stateFields or [])

        for f in eff_fields:
            name = str(f.name or "").strip()
            if not name:
                continue
            w = self._editor.get_widget(name)
            if w is None:
                continue
            access = _state_field_access(node, name)
            read_only = access == F8StateAccess.ro or _state_input_is_connected(node, name)
            _set_read_only_widget(w, read_only=bool(read_only))

    def _clear_editor(self, *, clear_node_id: bool = True) -> None:
        if clear_node_id:
            self._node_id = None
        editor = self._editor
        if editor is not None:
            self._editor = None
            self._container_layout.removeWidget(editor)
            try:
                editor.setVisible(False)
            except Exception:
                logger.exception("Failed to hide editor before deleteLater")
            editor.deleteLater()
        try:
            self._empty.setVisible(True)
        except Exception:
            pass

    def _set_editor(self, editor: F8StudioNodePropEditorWidget) -> None:
        # Preserve the node id that the caller (set_node) just set. We are
        # swapping the editor widget, not clearing the selection.
        self._clear_editor(clear_node_id=False)
        self._editor = editor
        try:
            self._empty.setVisible(False)
        except Exception:
            pass
        self._container_layout.addWidget(editor, 0)
        self._sync_container_width()
        try:
            editor.property_changed.connect(self._on_editor_property_changed)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            editor.property_closed.connect(self._on_editor_closed)  # type: ignore[attr-defined]
        except Exception:
            pass

    def set_node(self, node: Any | None, *, force_clear: bool = False) -> None:
        if node is None:
            # Avoid transient clear -> re-set flicker caused by selection jitter.
            # Only clear when explicitly forced (eg. node deleted) or when the
            # panel is currently empty.
            if force_clear or self._editor is None:
                self._clear_editor(clear_node_id=True)
            return
        try:
            node_id = str(node.id or "")
        except Exception:
            node_id = ""
        if not node_id:
            self._clear_editor(clear_node_id=True)
            return
        if self._node_id == node_id and self._editor is not None:
            return
        self._node_id = node_id
        self._set_editor(F8StudioNodePropEditorWidget(self._container, node=node))
        try:
            self._scroll.verticalScrollBar().setValue(0)
        except Exception:
            pass

    def _on_node_selected(self, node: Any) -> None:
        self._last_node_click_ts = time.monotonic()
        self.set_node(node)

    def _on_node_selection_changed(self, selected: list[Any], _deselected: list[Any]) -> None:
        # NodeGraphQt can emit transient selection updates (eg. deselect then
        # select). Clicking on embedded widgets inside a node can also cause
        # selection to briefly clear. Debounce and query the final selection.
        try:
            self._selection_timer.start(0)
        except Exception:
            # Fallback: behave like the default signal payload.
            if selected:
                self.set_node(selected[0])

    def _on_nodes_deleted(self, node_ids: list[str]) -> None:
        if not self._node_id:
            return
        if self._node_id in set(str(x) for x in (node_ids or [])):
            self.set_node(None, force_clear=True)

    def _on_editor_closed(self, _node_id: str) -> None:
        # User closed the editor explicitly; clear the view.
        self.set_node(None, force_clear=True)

    def _apply_graph_selection(self) -> None:
        """
        Apply the current graph selection to the properties panel.

        Keep showing the last node when selection is empty.

        Some embedded node controls (eg. inline state expand/collapse) can
        temporarily clear selection during the click sequence. Clearing the
        panel on empty selection causes a visible flash. Since Studio only
        needs a single active properties view, keep the last shown node until
        another node is selected or the node is deleted.
        """
        g = self._node_graph
        selected_nodes: list[Any] = []
        if g is not None:
            try:
                selected_nodes = list(g.selected_nodes() or [])  # type: ignore[attr-defined]
            except Exception:
                selected_nodes = []
        if selected_nodes:
            self.set_node(selected_nodes[0])
            return
        # No selection: keep current panel content (do not clear).
        return

    def _on_editor_property_changed(self, node_id: str, prop_name: str, prop_value: Any) -> None:
        if self._block_signal:
            return
        g = self._node_graph
        if g is None:
            return
        nid = str(node_id or "").strip()
        if not nid:
            return
        try:
            node = g.get_node_by_id(nid)  # type: ignore[attr-defined]
        except Exception:
            node = None
        if node is None:
            return
        try:
            node.set_property(prop_name, prop_value, push_undo=True)
        except Exception:
            logger.exception("set_property failed nodeId=%s prop=%s", nid, prop_name)

    def _on_graph_property_changed(self, node: Any, prop_name: str, prop_value: Any) -> None:
        """
        Keep UI in sync when node properties are updated externally (runtime sync, undo, etc.).
        """
        if self._editor is None or self._node_id is None:
            return
        try:
            if str(node.id or "") != self._node_id:
                return
        except Exception:
            return
        prop_key = str(prop_name or "").strip()
        if not prop_key:
            return

        # Always try pool refresh even if the pool field has no visible editor widget.
        try:
            self._editor.refresh_option_pool(prop_key)
        except Exception:
            pass

        try:
            w = self._editor.get_widget(prop_name)
        except Exception:
            w = None
        if w is None:
            return
        try:
            cur = w.get_value()
        except Exception:
            cur = None
        if cur == prop_value:
            return
        self._block_signal = True
        try:
            w.set_value(prop_value)
        except Exception:
            pass
        finally:
            self._block_signal = False


def _is_json_state_value(node: Any, prop_name: str) -> bool:
    """
    True if the property is a state field whose schema is object/array/any.
    """
    spec = _get_node_spec(node)
    if spec is None:
        return False
    try:
        fields = list(spec.stateFields or [])
    except Exception:
        fields = []
    for f in fields:
        try:
            if str(f.name or "").strip() != prop_name:
                continue
        except Exception:
            continue
        try:
            vs = f.valueSchema
        except Exception:
            vs = None
        return _schema_type(vs) in {"object", "array", "any"}
    return False


def _reorder_tabs(tab_widget: QtWidgets.QTabWidget, preferred: list[str]) -> None:
    """
    Reorder tabs so `preferred` (if present) are first in that order,
    then any remaining tabs (in their current relative order).
    """
    if tab_widget.count() <= 1:
        return

    current = [tab_widget.tabText(i) for i in range(tab_widget.count())]
    preferred_present = [t for t in preferred if t in current]
    rest = [t for t in current if t not in preferred_present]
    target = preferred_present + rest

    # Move tabs into target order using the tab bar.
    bar = tab_widget.tabBar()
    for dst, name in enumerate(target):
        for src in range(bar.count()):
            if tab_widget.tabText(src) == name:
                if src != dst:
                    bar.moveTab(src, dst)
                break
