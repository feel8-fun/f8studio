from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable

from NodeGraphQt import PropertiesBinWidget
from NodeGraphQt.constants import NodeEnum, NodePropWidgetEnum
from NodeGraphQt.custom_widgets.properties_bin.node_property_factory import NodePropertyWidgetFactory
from NodeGraphQt.custom_widgets.properties_bin.node_property_widgets import PropLineEdit

from qtpy import QtWidgets, QtCore, QtGui

from f8pysdk import (
    F8DataPortSpec,
    F8DataTypeSchema,
    F8OperatorSpec,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
)


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
    if hasattr(value, "value") and not isinstance(value, (bytes, bytearray)):
        try:
            return _to_jsonable(getattr(value, "value"))
        except Exception:
            pass
    # Pydantic models (BaseModel / RootModel).
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump(mode="json"))
        except Exception:
            try:
                return _to_jsonable(value.model_dump())
            except Exception:
                pass
    # RootModel inner `.root`.
    root = getattr(value, "root", None)
    if root is not None:
        return _to_jsonable(root)
    return str(value)


def _schema_to_json_obj(schema: Any) -> Any:
    if schema is None:
        return None
    if isinstance(schema, (dict, list, str, int, float, bool)) or schema is None:
        return schema
    if hasattr(schema, "model_dump"):
        try:
            return schema.model_dump(mode="json")
        except Exception:
            pass
    root = getattr(schema, "root", None)
    if root is not None:
        if hasattr(root, "model_dump"):
            try:
                return root.model_dump(mode="json")
            except Exception:
                pass
        return root
    return str(schema)


def _schema_from_json_obj(obj: Any) -> F8DataTypeSchema:
    if isinstance(obj, F8DataTypeSchema):
        return obj
    return F8DataTypeSchema.model_validate(obj)


def _schema_type(schema: Any) -> str:
    try:
        inner = getattr(schema, "root", schema)
        t = getattr(inner, "type", None)
        if hasattr(t, "value"):
            return str(t.value)
        return str(t)
    except Exception:
        return ""


def _state_field_schema(node: Any, prop_name: str) -> Any | None:
    spec = getattr(node, "spec", None)
    fields = list(getattr(spec, "stateFields", None) or [])
    for f in fields:
        if str(getattr(f, "name", "") or "").strip() == prop_name:
            return getattr(f, "valueSchema", None)
    return None


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


class _F8JsonPropTextEdit(QtWidgets.QTextEdit):
    """
    QTextEdit property widget that round-trips JSON values as python objects.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name: str | None = None
        self._prev_text = ""
        self._prev_value: Any = None

    def get_name(self) -> str:
        return self._name or ""

    def set_name(self, name: str) -> None:
        self._name = name

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._prev_text = self.toPlainText()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if self._prev_text == self.toPlainText():
            return
        text = self.toPlainText().strip()
        if not text:
            self._prev_value = None
            self.value_changed.emit(self.get_name(), None)
            self._prev_text = ""
            return
        try:
            obj = json.loads(text)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid JSON", str(e))
            self.setPlainText(self._prev_text)
            return
        self._prev_value = obj
        self.value_changed.emit(self.get_name(), obj)
        self._prev_text = ""

    def get_value(self) -> Any:
        return self._prev_value

    def set_value(self, value: Any) -> None:
        self._prev_value = value
        try:
            text = "" if value is None else json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            text = str(value)
        if text != self.toPlainText():
            self.setPlainText(text)


class _F8NumberPropLineEdit(QtWidgets.QLineEdit):
    """
    QLineEdit property widget that parses int/float and emits typed values.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, data_type: type[int] | type[float]):
        super().__init__(parent)
        self._name = ""
        self._data_type = data_type
        self._min: float | int | None = None
        self._max: float | int | None = None
        self._prev_text = ""
        self.setClearButtonEnabled(True)
        self.editingFinished.connect(self._on_editing_finished)

    def set_name(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    def set_min(self, v) -> None:
        self._min = v

    def set_max(self, v) -> None:
        self._max = v

    def get_value(self):
        txt = str(self.text() or "").strip()
        if txt == "":
            return None
        try:
            if self._data_type is int:
                return int(float(txt))
            return float(txt)
        except Exception:
            return None

    def set_value(self, value) -> None:
        if value is None:
            if self.text() != "":
                self.setText("")
            return
        try:
            txt = str(int(value) if self._data_type is int else float(value))
        except Exception:
            txt = str(value)
        if txt != self.text():
            self.setText(txt)

    def _on_editing_finished(self) -> None:
        raw = str(self.text() or "").strip()
        if raw == self._prev_text:
            return
        self._prev_text = raw
        value = self.get_value()
        if value is None and raw != "":
            # restore previous valid value if any (best-effort).
            return
        if value is not None:
            try:
                if self._min is not None:
                    value = max(value, self._min)
                if self._max is not None:
                    value = min(value, self._max)
            except Exception:
                pass
            self.set_value(value)
        self.value_changed.emit(self.get_name(), value)


class _F8StateContainer(QtWidgets.QWidget):
    """
    Node properties container widget that displays nodes properties under
    a tab in the ``NodePropWidget`` widget.
    """

    def __init__(self, parent=None):
        super(_F8StateContainer, self).__init__(parent)
        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setColumnStretch(1, 1)
        self.__layout.setSpacing(6)

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
        label_widget = QtWidgets.QLabel(label)
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


def _icon_from_style(
    widget: QtWidgets.QWidget, style_icon: QtWidgets.QStyle.StandardPixmap, fallback: str
) -> QtGui.QIcon:
    try:
        icon = widget.style().standardIcon(style_icon)
        if not icon.isNull():
            return icon
    except Exception:
        pass
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

        add_btn = QtWidgets.QToolButton()
        add_btn.setAutoRaise(True)
        add_btn.setToolTip("Add")
        add_btn.setIcon(_icon_from_style(add_btn, QtWidgets.QStyle.SP_FileDialogNewFolder, "list-add"))
        add_btn.clicked.connect(self.add_clicked.emit)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(header_label)
        header.addStretch(1)
        header.addWidget(add_btn)

        self._list_layout = QtWidgets.QVBoxLayout()
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)
        outer.addLayout(header)
        outer.addLayout(self._list_layout)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

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

    def __init__(self, parent=None, *, name: str, placeholder: str):
        super().__init__(parent)

        self.name_edit = QtWidgets.QLineEdit(name)
        self.name_edit.setPlaceholderText(placeholder)
        self.name_edit.setClearButtonEnabled(True)
        self.name_edit.editingFinished.connect(self._emit_commit)

        edit_btn = QtWidgets.QToolButton()
        edit_btn.setAutoRaise(True)
        edit_btn.setToolTip("Edit")
        edit_btn.setIcon(_icon_from_style(edit_btn, QtWidgets.QStyle.SP_FileDialogDetailedView, "document-edit"))
        edit_btn.clicked.connect(self.edit_clicked.emit)

        del_btn = QtWidgets.QToolButton()
        del_btn.setAutoRaise(True)
        del_btn.setToolTip("Delete")
        del_btn.setIcon(_icon_from_style(del_btn, QtWidgets.QStyle.SP_TrashIcon, "edit-delete"))
        del_btn.clicked.connect(self.delete_clicked.emit)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.name_edit, 1)
        layout.addWidget(edit_btn)
        layout.addWidget(del_btn)

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
    def __init__(self, parent=None, *, title: str, port: F8DataPortSpec):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._schema = getattr(port, "valueSchema", None) or _schema_from_json_obj({"type": "any"})

        self._name = QtWidgets.QLineEdit(str(getattr(port, "name", "") or ""))
        self._name.setClearButtonEnabled(True)
        self._required = QtWidgets.QCheckBox()
        self._required.setChecked(bool(getattr(port, "required", True)))
        self._desc = QtWidgets.QPlainTextEdit(str(getattr(port, "description", "") or ""))

        self._schema_summary = QtWidgets.QLabel("")
        self._schema_summary.setStyleSheet("color: #888;")
        self._refresh_schema_summary()

        schema_btn = QtWidgets.QPushButton("Edit Schema…")
        schema_btn.clicked.connect(self._edit_schema)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self._name)
        form.addRow("Required", self._required)
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
        desc = str(self._desc.toPlainText() or "").strip() or None
        return F8DataPortSpec(name=name, required=required, description=desc, valueSchema=self._schema)


class _F8EditStateFieldDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title: str, field: F8StateSpec):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._schema = getattr(field, "valueSchema", None) or _schema_from_json_obj({"type": "any"})

        self._name = QtWidgets.QLineEdit(str(getattr(field, "name", "") or ""))
        self._name.setClearButtonEnabled(True)

        self._access = QtWidgets.QComboBox()
        self._access.addItems([e.value for e in F8StateAccess])
        self._access.setCurrentText(str(getattr(getattr(field, "access", None), "value", "rw") or "rw"))

        self._required = QtWidgets.QCheckBox()
        self._required.setChecked(bool(getattr(field, "required", False)))

        self._show_on_node = QtWidgets.QCheckBox()
        self._show_on_node.setChecked(bool(getattr(field, "showOnNode", False)))

        self._label = QtWidgets.QLineEdit(str(getattr(field, "label", "") or ""))
        self._label.setClearButtonEnabled(True)
        self._desc = QtWidgets.QPlainTextEdit(str(getattr(field, "description", "") or ""))
        self._ui_control = QtWidgets.QLineEdit(str(getattr(field, "uiControl", "") or ""))
        self._ui_control.setClearButtonEnabled(True)
        self._ui_lang = QtWidgets.QLineEdit(str(getattr(field, "uiLanguage", "") or ""))
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


class _F8SpecStateFieldEditor(QtWidgets.QWidget):
    spec_applied = QtCore.Signal()

    def __init__(self, parent=None, node=None, on_apply: Callable[[], None] | None = None):
        super().__init__(parent)
        self._node = node
        self._on_apply = on_apply

        self._sec = _F8SpecListSection(title="State Fields")
        self._sec.add_clicked.connect(self._add_field)

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

        self._load_from_spec()

    def _load_from_spec(self) -> None:
        self._sec.clear()
        spec = getattr(self._node, "spec", None)
        if spec is None:
            return
        for f in list(getattr(spec, "stateFields", None) or []):
            self._sec.add_row(self._make_row(f))

    def _make_row(self, field: F8StateSpec) -> _F8SpecNameRow:
        name = str(getattr(field, "name", "") or "")
        row = _F8SpecNameRow(name=name, placeholder="state name")
        row.setProperty("_field", field)
        row.edit_clicked.connect(lambda: self._edit_field(row))
        row.delete_clicked.connect(lambda: self._delete_row(row))
        row.name_committed.connect(lambda v: self._rename_field(row, v))
        row.setToolTip(self._field_tooltip(field))
        return row

    def _field_tooltip(self, field: F8StateSpec) -> str:
        access = str(getattr(getattr(field, "access", None), "value", "") or "")
        req = bool(getattr(field, "required", False))
        show = bool(getattr(field, "showOnNode", False))
        desc = str(getattr(field, "description", "") or "").strip()
        t = _schema_type(getattr(field, "valueSchema", None))
        parts = [f"access={access or 'rw'}", f"required={req}", f"showOnNode={show}", f"type={t or 'unknown'}"]
        if desc:
            parts.append(desc)
        return "\n".join(parts)

    def _edit_field(self, row: _F8SpecNameRow) -> None:
        field = row.property("_field")
        if not isinstance(field, F8StateSpec):
            field = F8StateSpec(
                name=row.name_edit.text(),
                valueSchema=_schema_from_json_obj({"type": "any"}),
                access=F8StateAccess.rw,
            )
        dlg = _F8EditStateFieldDialog(self, title="Edit state field", field=field)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_field = dlg.field()
        row.setProperty("_field", new_field)
        row.name_edit.setText(str(getattr(new_field, "name", "") or ""))
        row.setToolTip(self._field_tooltip(new_field))
        self._commit()

    def _rename_field(self, row: _F8SpecNameRow, name: str) -> None:
        field = row.property("_field")
        if not isinstance(field, F8StateSpec):
            field = F8StateSpec(name=name, valueSchema=_schema_from_json_obj({"type": "any"}), access=F8StateAccess.rw)
        else:
            field = field.model_copy(deep=True)
            field.name = name
        row.setProperty("_field", field)
        row.setToolTip(self._field_tooltip(field))
        self._commit()

    def _delete_row(self, row: QtWidgets.QWidget) -> None:
        row.setParent(None)
        row.deleteLater()
        self._commit()

    def _add_field(self) -> None:
        field = F8StateSpec(name="", valueSchema=_schema_from_json_obj({"type": "any"}), access=F8StateAccess.rw)
        row = self._make_row(field)
        self._sec.add_row(row)
        self._edit_field(row)

    def _commit(self) -> None:
        spec = getattr(self._node, "spec", None)
        if spec is None:
            return
        fields: list[F8StateSpec] = []
        for r in self._sec.rows():
            f = r.property("_field")
            if isinstance(f, F8StateSpec) and str(getattr(f, "name", "") or "").strip():
                fields.append(f)
        try:
            spec.stateFields = fields
        except Exception:
            spec2 = spec.model_copy(deep=True)
            spec2.stateFields = fields
            self._node.spec = spec2

        if self._on_apply:
            self._on_apply()
        self.spec_applied.emit()


class _F8SpecPortEditor(QtWidgets.QWidget):
    """
    Narrow-sidebar friendly spec ports editor.
    """

    spec_applied = QtCore.Signal()

    def __init__(self, parent=None, node=None, on_apply: Callable[[], None] | None = None):
        super().__init__(parent)
        self._node = node
        self._on_apply = on_apply

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
        spec = getattr(self._node, "spec", None)
        is_operator = isinstance(spec, F8OperatorSpec)
        self._sec_exec_in.setVisible(is_operator)
        self._sec_exec_out.setVisible(is_operator)

        self._sec_exec_in.clear()
        self._sec_exec_out.clear()
        self._sec_data_in.clear()
        self._sec_data_out.clear()

        if spec is None:
            return

        if is_operator:
            for name in list(getattr(spec, "execInPorts", None) or []):
                self._sec_exec_in.add_row(self._make_exec_row(str(name)))
            for name in list(getattr(spec, "execOutPorts", None) or []):
                self._sec_exec_out.add_row(self._make_exec_row(str(name)))

        for p in list(getattr(spec, "dataInPorts", None) or []):
            self._sec_data_in.add_row(self._make_data_row(p))
        for p in list(getattr(spec, "dataOutPorts", None) or []):
            self._sec_data_out.add_row(self._make_data_row(p))

    def _make_exec_row(self, name: str) -> _F8SpecNameRow:
        row = _F8SpecNameRow(name=name, placeholder="port name")
        row.edit_clicked.connect(lambda: self._edit_exec(row))
        row.delete_clicked.connect(lambda: self._delete_row(row))
        row.name_committed.connect(lambda _v: self._commit())
        return row

    def _make_data_row(self, port: F8DataPortSpec) -> _F8SpecNameRow:
        row = _F8SpecNameRow(name=str(getattr(port, "name", "") or ""), placeholder="port name")
        row.setProperty("_port", port)
        row.edit_clicked.connect(lambda: self._edit_data(row))
        row.delete_clicked.connect(lambda: self._delete_row(row))
        row.name_committed.connect(lambda v: self._rename_data(row, v))
        row.setToolTip(self._data_tooltip(port))
        return row

    def _data_tooltip(self, port: F8DataPortSpec) -> str:
        req = bool(getattr(port, "required", True))
        desc = str(getattr(port, "description", "") or "").strip()
        t = _schema_type(getattr(port, "valueSchema", None))
        parts = [f"required={req}", f"type={t or 'unknown'}"]
        if desc:
            parts.append(desc)
        return "\n".join(parts)

    def _edit_exec(self, row: _F8SpecNameRow) -> None:
        dlg = _F8EditExecPortDialog(self, title="Edit exec port", name=row.name_edit.text())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        row.name_edit.setText(dlg.name())
        self._commit()

    def _edit_data(self, row: _F8SpecNameRow) -> None:
        port = row.property("_port")
        if not isinstance(port, F8DataPortSpec):
            port = F8DataPortSpec(
                name=row.name_edit.text(), required=True, valueSchema=_schema_from_json_obj({"type": "any"})
            )
        dlg = _F8EditDataPortDialog(self, title="Edit data port", port=port)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_port = dlg.port()
        row.setProperty("_port", new_port)
        row.name_edit.setText(str(getattr(new_port, "name", "") or ""))
        row.setToolTip(self._data_tooltip(new_port))
        self._commit()

    def _rename_data(self, row: _F8SpecNameRow, name: str) -> None:
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
        row.setParent(None)
        row.deleteLater()
        self._commit()

    def _add_exec(self, is_in: bool) -> None:
        row = self._make_exec_row("")
        (self._sec_exec_in if is_in else self._sec_exec_out).add_row(row)
        row.name_edit.setFocus()

    def _add_data(self, is_in: bool) -> None:
        port = F8DataPortSpec(
            name="", required=True, description=None, valueSchema=_schema_from_json_obj({"type": "any"})
        )
        row = self._make_data_row(port)
        (self._sec_data_in if is_in else self._sec_data_out).add_row(row)
        self._edit_data(row)

    def _commit(self) -> None:
        spec = getattr(self._node, "spec", None)
        if spec is None:
            return

        exec_in: list[str] = []
        exec_out: list[str] = []
        if isinstance(spec, F8OperatorSpec):
            for r in self._sec_exec_in.rows():
                name = str(getattr(r, "name_edit").text() or "").strip()
                if name:
                    exec_in.append(name)
            for r in self._sec_exec_out.rows():
                name = str(getattr(r, "name_edit").text() or "").strip()
                if name:
                    exec_out.append(name)

        data_in: list[F8DataPortSpec] = []
        data_out: list[F8DataPortSpec] = []
        for r in self._sec_data_in.rows():
            port = r.property("_port")
            if isinstance(port, F8DataPortSpec) and str(getattr(port, "name", "") or "").strip():
                data_in.append(port)
        for r in self._sec_data_out.rows():
            port = r.property("_port")
            if isinstance(port, F8DataPortSpec) and str(getattr(port, "name", "") or "").strip():
                data_out.append(port)

        try:
            spec.dataInPorts = data_in
            spec.dataOutPorts = data_out
            if isinstance(spec, F8OperatorSpec):
                spec.execInPorts = exec_in
                spec.execOutPorts = exec_out
        except Exception:
            spec2 = spec.model_copy(deep=True)
            spec2.dataInPorts = data_in
            spec2.dataOutPorts = data_out
            if isinstance(spec2, F8OperatorSpec):
                spec2.execInPorts = exec_in
                spec2.execOutPorts = exec_out
            self._node.spec = spec2

        if self._on_apply:
            self._on_apply()
        self.spec_applied.emit()


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
        node_extra_props: list[tuple[str, Any]] = []
        for prop_name, prop_val in model.custom_properties.items():
            # Put svcId into the built-in "Node" tab (avoid wasting a whole tab).
            if prop_name == "svcId":
                node_extra_props.append((prop_name, prop_val))
                continue
            tab_name = model.get_tab_name(prop_name)
            tab_mapping[tab_name].append((prop_name, prop_val))

        # add tabs.
        reserved_tabs = ["Node", "Port", "StateField"]
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
            for prop_name, value in tab_mapping[tab]:
                wid_type = model.get_widget_type(prop_name)
                if wid_type == 0:
                    continue

                schema = _state_field_schema(node, prop_name)
                schema_t = _schema_type(schema) if schema is not None else ""
                if schema is not None and schema_t in {"integer", "number"}:
                    widget = _F8NumberPropLineEdit(data_type=int if schema_t == "integer" else float)
                    widget.set_name(prop_name)
                else:
                    widget = widget_factory.get_widget(wid_type)
                    widget.set_name(prop_name)

                tooltip = None
                if prop_name in common_props.keys():
                    if "items" in common_props[prop_name].keys():
                        widget.set_items(common_props[prop_name]["items"])
                    if "range" in common_props[prop_name].keys():
                        prop_range = common_props[prop_name]["range"]
                        if hasattr(widget, "set_min") and hasattr(widget, "set_max"):
                            widget.set_min(prop_range[0])
                            widget.set_max(prop_range[1])
                        else:
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
        if node_extra_props:
            for prop_name, prop_value in node_extra_props:
                wid_type = model.get_widget_type(prop_name)
                widget = widget_factory.get_widget(wid_type) or widget_factory.get_widget(
                    NodePropWidgetEnum.QLABEL.value
                )
                widget.set_name(prop_name)
                prop_window.add_widget(
                    name=prop_name,
                    widget=widget,
                    value=prop_value,
                    label=prop_name.replace("_", " "),
                    tooltip="Bound service container id.",
                )
                widget.value_changed.connect(self._on_property_changed)

        self.type_wgt.setText(model.get_property("type_") or "")

        # built-in spec editors (if node has F8 spec).
        spec = getattr(node, "spec", None)
        if isinstance(spec, (F8OperatorSpec, F8ServiceSpec)):
            spec_ports = _F8SpecPortEditor(self, node=node, on_apply=self._on_spec_applied)
            self.__tab.addTab(spec_ports, "Port")
            spec_state = _F8SpecStateFieldEditor(self, node=node, on_apply=self._on_spec_applied)
            self.__tab.addTab(spec_state, "StateField")

        # hide/remove empty tabs with no property widgets.
        tab_index = {self.__tab.tabText(x): x for x in range(self.__tab.count())}
        current_idx = None
        for tab_name, prop_window in self.__tab_windows.items():
            prop_widgets = prop_window.get_all_widgets()
            if not prop_widgets:
                # I prefer to hide the tab but in older version of pyside this
                # attribute doesn't exist we'll just remove.
                if hasattr(self.__tab, "setTabVisible"):
                    self.__tab.setTabVisible(tab_index[tab_name], False)
                else:
                    self.__tab.removeTab(tab_index[tab_name])
                continue
            if current_idx is None:
                current_idx = tab_index[tab_name]

        # Order: State, Port, StateField, Node (Node last).
        _reorder_tabs(self.__tab, ["State", "Port", "StateField", "Node"])

        # Default tab: first existing among preferred, else 0.
        preferred_default = None
        for t in ["State", "Port", "StateField", "Node"]:
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

    def reload(self) -> None:
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
        while self.__tab.count():
            self.__tab.removeTab(0)
        self.__tab_windows = {}
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
        self.__tab_windows[name] = _F8StateContainer(self)
        self.__tab.addTab(self.__tab_windows[name], name)
        return self.__tab_windows[name]

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


def _is_json_state_value(node: Any, prop_name: str) -> bool:
    """
    True if the property is a state field whose schema is object/array/any.
    """
    spec = getattr(node, "spec", None)
    fields = list(getattr(spec, "stateFields", None) or [])
    for f in fields:
        if str(getattr(f, "name", "") or "").strip() != prop_name:
            continue
        return _schema_type(getattr(f, "valueSchema", None)) in {"object", "array", "any"}
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
