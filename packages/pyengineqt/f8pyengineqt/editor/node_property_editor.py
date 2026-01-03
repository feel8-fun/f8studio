from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from qtpy import QtCore, QtWidgets

from f8pysdk import (
    F8DataPortSpec,
    F8DataTypeSchema,
    F8OperatorSpec,
    F8PrimitiveTypeEnum,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    array_schema,
    boolean_schema,
    complex_object_schema,
    integer_schema,
    number_schema,
    schema_default,
    schema_type,
    string_schema,
)

from ..renderers.generic import GenericNode


def _as_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)


def _schema_type_label(schema: F8DataTypeSchema) -> str:
    schema_t = schema_type(schema)
    if isinstance(schema_t, F8PrimitiveTypeEnum):
        return schema_t.value
    return str(schema_t)


def _make_schema_by_type(type_name: str) -> F8DataTypeSchema:
    if type_name == F8PrimitiveTypeEnum.string.value:
        return F8DataTypeSchema.model_validate(string_schema().model_dump(mode="json"))
    if type_name == F8PrimitiveTypeEnum.number.value:
        return F8DataTypeSchema.model_validate(number_schema().model_dump(mode="json"))
    if type_name == F8PrimitiveTypeEnum.integer.value:
        return F8DataTypeSchema.model_validate(integer_schema().model_dump(mode="json"))
    if type_name == F8PrimitiveTypeEnum.boolean.value:
        return F8DataTypeSchema.model_validate(boolean_schema().model_dump(mode="json"))
    if type_name == "object":
        schema = complex_object_schema(properties={}).model_dump(mode="json")
        return F8DataTypeSchema.model_validate(schema)
    if type_name == "array":
        schema = array_schema(items=string_schema()).model_dump(mode="json")
        return F8DataTypeSchema.model_validate(schema)
    return F8DataTypeSchema.model_validate(any_schema().model_dump(mode="json"))


class JsonTextDialog(QtWidgets.QDialog):
    def __init__(self, *, title: str, initial_json: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle(title)

        self._editor = QtWidgets.QPlainTextEdit()
        self._editor.setPlainText(initial_json)

        self._error = QtWidgets.QLabel()
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #c44;")

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._editor)
        layout.addWidget(self._error)
        layout.addWidget(buttons)

    def json_text(self) -> str:
        return self._editor.toPlainText()

    def _on_accept(self) -> None:
        try:
            json.loads(self.json_text())
        except Exception as exc:
            self._error.setText(str(exc))
            return
        self.accept()


class SchemaJsonDialog(JsonTextDialog):
    def __init__(
        self,
        *,
        title: str,
        initial_schema: F8DataTypeSchema,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(title=title, initial_json=_as_json(initial_schema.model_dump(mode="json")), parent=parent)
        self._schema: F8DataTypeSchema | None = None

    def schema(self) -> F8DataTypeSchema:
        if self._schema is None:
            raise RuntimeError("Schema not validated yet.")
        return self._schema

    def _on_accept(self) -> None:
        try:
            payload = json.loads(self.json_text())
            self._schema = F8DataTypeSchema.model_validate(payload)
        except Exception as exc:
            self._error.setText(str(exc))
            return
        self.accept()


class SchemaValueEditor(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(object)

    def __init__(
        self,
        schema: F8DataTypeSchema,
        *,
        language: str | None = None,
        read_only: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._schema = schema
        self._language = language
        self._read_only = read_only
        self._updating = False

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._build()

    def set_schema(self, schema: F8DataTypeSchema) -> None:
        self._schema = schema
        self._clear()
        self._build()

    def set_value(self, value: Any) -> None:
        self._updating = True
        try:
            self._set_value_impl(value)
        finally:
            self._updating = False

    def value(self) -> Any:
        schema_t = schema_type(self._schema)
        if isinstance(schema_t, F8PrimitiveTypeEnum):
            return self._primitive_value(schema_t)
        if schema_t == "object":
            data: dict[str, Any] = {k: w.value() for k, w in self._object_props.items()}
            if self._extra_json is not None:
                extra_raw = self._extra_json.toPlainText().strip()
                if extra_raw:
                    try:
                        extra = json.loads(extra_raw)
                        if isinstance(extra, dict):
                            for k, v in extra.items():
                                if k not in data:
                                    data[k] = v
                    except Exception:
                        pass
            return data
        if schema_t == "array":
            return [w.value() for w in self._array_items]

        if hasattr(self, "_json") and isinstance(self._json, QtWidgets.QPlainTextEdit):
            try:
                return json.loads(self._json.toPlainText() or "null")
            except Exception:
                return None
        return None

    def _emit(self, value: Any) -> None:
        if self._updating:
            return
        self.valueChanged.emit(value)

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _build(self) -> None:
        schema_t = schema_type(self._schema)
        if isinstance(schema_t, F8PrimitiveTypeEnum):
            self._build_primitive(schema_t)
        elif schema_t == "object":
            self._build_object()
        elif schema_t == "array":
            self._build_array()
        else:
            self._build_any()

    def _build_any(self) -> None:
        self._json = QtWidgets.QPlainTextEdit()
        self._json.setReadOnly(self._read_only)

        self._apply = QtWidgets.QPushButton("Apply JSON")
        self._apply.setEnabled(not self._read_only)
        self._apply.clicked.connect(self._apply_json)

        self._error = QtWidgets.QLabel()
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #c44;")

        self._layout.addWidget(self._json)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._apply)
        self._layout.addLayout(row)
        self._layout.addWidget(self._error)

    def _apply_json(self) -> None:
        try:
            value = json.loads(self._json.toPlainText() or "null")
        except Exception as exc:
            self._error.setText(str(exc))
            return
        self._error.setText("")
        self._emit(value)

    def _build_primitive(self, schema_t: F8PrimitiveTypeEnum) -> None:
        schema = self._schema
        if schema_t == F8PrimitiveTypeEnum.boolean:
            widget = QtWidgets.QCheckBox()
            widget.setEnabled(not self._read_only)
            widget.stateChanged.connect(lambda _: self._emit(bool(widget.isChecked())))
            self._widget = widget
            self._layout.addWidget(widget)
            return

        if schema_t == F8PrimitiveTypeEnum.integer:
            widget = QtWidgets.QSpinBox()
            minimum = getattr(schema, "minimum", None)
            maximum = getattr(schema, "maximum", None)
            min_value = int(minimum) if minimum is not None else -(2**31)
            max_value = int(maximum) if maximum is not None else 2**31 - 1
            widget.setRange(min_value, max_value)
            widget.setEnabled(not self._read_only)
            widget.valueChanged.connect(lambda v: self._emit(int(v)))
            self._widget = widget
            self._layout.addWidget(widget)
            return

        if schema_t == F8PrimitiveTypeEnum.number:
            widget = QtWidgets.QDoubleSpinBox()
            widget.setDecimals(6)
            minimum = getattr(schema, "minimum", None)
            maximum = getattr(schema, "maximum", None)
            min_value = float(minimum) if minimum is not None else -1e18
            max_value = float(maximum) if maximum is not None else 1e18
            widget.setRange(min_value, max_value)
            step = getattr(schema, "multipleOf", None)
            if step:
                try:
                    widget.setSingleStep(float(step))
                except Exception:
                    pass
            widget.setEnabled(not self._read_only)
            widget.valueChanged.connect(lambda v: self._emit(float(v)))
            self._widget = widget
            self._layout.addWidget(widget)
            return

        if schema_t == F8PrimitiveTypeEnum.string:
            enum_values = getattr(schema, "enum", None)
            if enum_values:
                widget = QtWidgets.QComboBox()
                widget.addItems([str(v) for v in enum_values])
                widget.setEnabled(not self._read_only)
                widget.currentTextChanged.connect(lambda v: self._emit(v))
                self._widget = widget
                self._layout.addWidget(widget)
                return

            if self._language:
                widget = QtWidgets.QPlainTextEdit()
                widget.setTabStopDistance(28)
                widget.setReadOnly(self._read_only)
                widget.textChanged.connect(lambda: self._emit(widget.toPlainText()))
                self._widget = widget
                self._layout.addWidget(widget)
                return

            widget = QtWidgets.QLineEdit()
            widget.setReadOnly(self._read_only)
            widget.textChanged.connect(lambda v: self._emit(v))
            self._widget = widget
            self._layout.addWidget(widget)
            return

        widget = QtWidgets.QLineEdit()
        widget.setReadOnly(True)
        widget.setText("null")
        self._widget = widget
        self._layout.addWidget(widget)

    def _primitive_value(self, schema_t: F8PrimitiveTypeEnum) -> Any:
        if schema_t == F8PrimitiveTypeEnum.boolean:
            return bool(self._widget.isChecked())
        if schema_t == F8PrimitiveTypeEnum.integer:
            return int(self._widget.value())
        if schema_t == F8PrimitiveTypeEnum.number:
            return float(self._widget.value())
        if schema_t == F8PrimitiveTypeEnum.string:
            if isinstance(self._widget, QtWidgets.QComboBox):
                return self._widget.currentText()
            if isinstance(self._widget, QtWidgets.QPlainTextEdit):
                return self._widget.toPlainText()
            return self._widget.text()
        return None

    def _build_object(self) -> None:
        schema = self._schema
        props = getattr(schema, "properties", None) or {}
        required_keys = set(getattr(schema, "required", None) or [])
        additional = bool(getattr(schema, "additionalProperties", False))

        self._object_props: dict[str, SchemaValueEditor] = {}
        group = QtWidgets.QGroupBox()
        form = QtWidgets.QFormLayout(group)

        for key, subschema in props.items():
            child = SchemaValueEditor(subschema, read_only=self._read_only, parent=group)
            child.valueChanged.connect(lambda _v, _key=key: self._emit(self.value()))
            label = f"{key}*" if key in required_keys else key
            form.addRow(label, child)
            self._object_props[key] = child

        self._extra_json: QtWidgets.QPlainTextEdit | None = None
        if additional:
            self._extra_json = QtWidgets.QPlainTextEdit()
            self._extra_json.setReadOnly(self._read_only)
            extra_apply = QtWidgets.QPushButton("Apply Extras JSON")
            extra_apply.setEnabled(not self._read_only)
            extra_apply.clicked.connect(lambda: self._emit(self.value()))
            extra_btns = QtWidgets.QHBoxLayout()
            extra_btns.addStretch(1)
            extra_btns.addWidget(extra_apply)
            form.addRow("extras (dict)", self._extra_json)
            form.addRow("", extra_btns)

        self._layout.addWidget(group)

    def _build_array(self) -> None:
        schema = self._schema
        self._array_item_schema = getattr(schema, "items", any_schema())

        self._array_container = QtWidgets.QWidget()
        self._array_layout = QtWidgets.QVBoxLayout(self._array_container)
        self._array_layout.setContentsMargins(0, 0, 0, 0)
        self._array_layout.setSpacing(4)
        self._array_items: list[SchemaValueEditor] = []

        self._add_item_btn = QtWidgets.QPushButton("Add Item")
        self._add_item_btn.setEnabled(not self._read_only)
        self._add_item_btn.clicked.connect(self._add_array_item)

        self._layout.addWidget(self._array_container)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._add_item_btn)
        self._layout.addLayout(row)

    def _add_array_item(self, *, initial: Any = None) -> None:
        if self._read_only:
            return

        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        editor = SchemaValueEditor(self._array_item_schema, read_only=False, parent=row_widget)
        editor.valueChanged.connect(lambda _v: self._emit(self.value()))
        if initial is not None:
            editor.set_value(initial)

        remove_btn = QtWidgets.QToolButton()
        remove_btn.setText("Remove")
        remove_btn.clicked.connect(lambda: self._remove_array_item(editor, row_widget))

        row_layout.addWidget(editor, 1)
        row_layout.addWidget(remove_btn, 0)

        self._array_items.append(editor)
        self._array_layout.addWidget(row_widget)

    def _remove_array_item(self, editor: "SchemaValueEditor", row_widget: QtWidgets.QWidget) -> None:
        try:
            self._array_items.remove(editor)
        except ValueError:
            pass
        row_widget.deleteLater()
        self._emit(self.value())

    def _set_value_impl(self, value: Any) -> None:
        schema_t = schema_type(self._schema)
        if isinstance(schema_t, F8PrimitiveTypeEnum):
            if schema_t == F8PrimitiveTypeEnum.boolean and isinstance(self._widget, QtWidgets.QCheckBox):
                self._widget.setChecked(bool(value))
            elif schema_t == F8PrimitiveTypeEnum.integer and isinstance(self._widget, QtWidgets.QSpinBox):
                self._widget.setValue(int(value or 0))
            elif schema_t == F8PrimitiveTypeEnum.number and isinstance(self._widget, QtWidgets.QDoubleSpinBox):
                self._widget.setValue(float(value or 0.0))
            elif schema_t == F8PrimitiveTypeEnum.string:
                if isinstance(self._widget, QtWidgets.QComboBox):
                    idx = self._widget.findText("" if value is None else str(value))
                    if idx >= 0:
                        self._widget.setCurrentIndex(idx)
                elif isinstance(self._widget, QtWidgets.QPlainTextEdit):
                    self._widget.setPlainText("" if value is None else str(value))
                elif isinstance(self._widget, QtWidgets.QLineEdit):
                    self._widget.setText("" if value is None else str(value))
            return

        if schema_t == "object":
            data = value if isinstance(value, dict) else {}
            for key, child in self._object_props.items():
                child.set_value(data.get(key))
            if self._extra_json is not None:
                extras = {k: v for k, v in data.items() if k not in self._object_props}
                self._extra_json.setPlainText(_as_json(extras) if extras else "")
            return

        if schema_t == "array":
            items = value if isinstance(value, list) else []
            for i in reversed(range(self._array_layout.count())):
                item = self._array_layout.itemAt(i)
                if item and item.widget():
                    item.widget().deleteLater()
            self._array_items.clear()
            for item_value in items:
                self._add_array_item(initial=item_value)
            return

        if hasattr(self, "_json") and isinstance(self._json, QtWidgets.QPlainTextEdit):
            self._json.setPlainText(_as_json(value))


@dataclass(frozen=True)
class _NodeSelection:
    node: GenericNode | None


class NodeStateEditorWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._selection = _NodeSelection(node=None)
        self._updating = False

        self._empty = QtWidgets.QLabel("Select a node to edit state.")
        self._empty.setAlignment(QtCore.Qt.AlignCenter)

        self._form_widget = QtWidgets.QWidget()
        self._form_layout = QtWidgets.QVBoxLayout(self._form_widget)
        self._form_layout.setContentsMargins(10, 10, 10, 10)
        self._form_layout.setSpacing(8)
        self._form_layout.addStretch(1)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._form_widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._empty)
        layout.addWidget(self._scroll)
        self._set_empty_visible(True)

    def set_node(self, node: GenericNode | None) -> None:
        self._selection = _NodeSelection(node=node)
        self._rebuild()

    def _set_empty_visible(self, is_empty: bool) -> None:
        self._empty.setVisible(is_empty)
        self._scroll.setVisible(not is_empty)

    def _clear_form(self) -> None:
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._form_layout.addStretch(1)

    def _rebuild(self) -> None:
        self._clear_form()
        node = self._selection.node
        if node is None:
            self._set_empty_visible(True)
            return

        self._set_empty_visible(False)

        header = QtWidgets.QLabel(f"{node.name()}  ({getattr(node.spec, 'operatorClass', '')})")
        header.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        header.setStyleSheet("font-weight: 600;")
        self._form_layout.insertWidget(0, header)

        for field in node.spec.states or []:
            self._form_layout.insertWidget(self._form_layout.count() - 1, self._build_field_editor(node, field))

    def _build_field_editor(self, node: GenericNode, field: F8StateSpec) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox(field.label or field.name)
        layout = QtWidgets.QVBoxLayout(box)

        meta_parts = [field.name, _schema_type_label(field.valueSchema), field.access.value]
        if field.language:
            meta_parts.append(f"lang={field.language}")
        meta = QtWidgets.QLabel(" · ".join(meta_parts))
        meta.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        meta.setStyleSheet("color: #666;")
        layout.addWidget(meta)

        if field.description:
            desc = QtWidgets.QLabel(field.description)
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #666;")
            layout.addWidget(desc)

        read_only = field.access == F8StateAccess.ro
        editor = SchemaValueEditor(field.valueSchema, language=field.language, read_only=read_only)

        try:
            value = node.get_property(field.name)
        except Exception:
            value = None
        if value is None:
            value = schema_default(field.valueSchema)
        editor.set_value(value)

        def apply_value(v: Any) -> None:
            if read_only or self._updating:
                return
            try:
                node.set_property(field.name, v, push_undo=False)
            except Exception:
                pass

        editor.valueChanged.connect(apply_value)
        layout.addWidget(editor)
        return box


class EditableStringList(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, *, label: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._label = QtWidgets.QLabel(label)
        self._list = QtWidgets.QListWidget()
        self._add = QtWidgets.QPushButton("Add")
        self._remove = QtWidgets.QPushButton("Remove")
        self._add.clicked.connect(self._on_add)
        self._remove.clicked.connect(self._on_remove)

        btns = QtWidgets.QVBoxLayout()
        btns.addWidget(self._add)
        btns.addWidget(self._remove)
        btns.addStretch(1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._list, 1)
        row.addLayout(btns, 0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addLayout(row)

    def set_items(self, items: Iterable[str]) -> None:
        self._list.clear()
        for item in items:
            self._list.addItem(str(item))

    def items(self) -> list[str]:
        return [self._list.item(i).text() for i in range(self._list.count())]

    def _on_add(self) -> None:
        text, ok = QtWidgets.QInputDialog.getText(self, "Add", "Name:")
        if not ok:
            return
        name = (text or "").strip()
        if not name:
            return
        if name in self.items():
            return
        self._list.addItem(name)
        self.changed.emit()

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if row < 0:
            return
        self._list.takeItem(row)
        self.changed.emit()


class DataPortTable(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, *, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._title = QtWidgets.QLabel(title)
        self._table = QtWidgets.QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["name", "type", "required", "description", "schema"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self._add = QtWidgets.QPushButton("Add")
        self._remove = QtWidgets.QPushButton("Remove")
        self._add.clicked.connect(self._on_add)
        self._remove.clicked.connect(self._on_remove)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self._add)
        btns.addWidget(self._remove)
        btns.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._title)
        layout.addWidget(self._table)
        layout.addLayout(btns)

    def set_ports(self, ports: list[F8DataPortSpec]) -> None:
        self._table.setRowCount(0)
        for port in ports:
            self._append_row(port)

    def ports(self) -> list[F8DataPortSpec]:
        ports: list[F8DataPortSpec] = []
        for row in range(self._table.rowCount()):
            name_widget = self._table.cellWidget(row, 0)
            type_widget = self._table.cellWidget(row, 1)
            req_widget = self._table.cellWidget(row, 2)
            desc_widget = self._table.cellWidget(row, 3)
            schema_btn = self._table.cellWidget(row, 4)

            if not isinstance(name_widget, QtWidgets.QLineEdit):
                continue
            if not isinstance(type_widget, QtWidgets.QComboBox):
                continue
            if not isinstance(req_widget, QtWidgets.QCheckBox):
                continue
            if not isinstance(desc_widget, QtWidgets.QLineEdit):
                continue
            if not isinstance(schema_btn, QtWidgets.QPushButton):
                continue

            schema = schema_btn.property("schema")
            if not isinstance(schema, F8DataTypeSchema):
                schema = _make_schema_by_type(type_widget.currentText())

            ports.append(
                F8DataPortSpec(
                    name=name_widget.text().strip(),
                    valueSchema=schema,
                    required=bool(req_widget.isChecked()),
                    description=desc_widget.text() or None,
                )
            )
        return ports

    def _append_row(self, port: F8DataPortSpec) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        name = QtWidgets.QLineEdit(port.name)
        name.textChanged.connect(lambda _: self.changed.emit())

        type_box = QtWidgets.QComboBox()
        type_box.addItems(
            [
                F8PrimitiveTypeEnum.string.value,
                F8PrimitiveTypeEnum.number.value,
                F8PrimitiveTypeEnum.integer.value,
                F8PrimitiveTypeEnum.boolean.value,
                "object",
                "array",
                "any",
            ]
        )
        type_box.setCurrentText(_schema_type_label(port.valueSchema))
        type_box.currentTextChanged.connect(lambda _: self._on_type_changed(row))

        required = QtWidgets.QCheckBox()
        required.setChecked(bool(port.required))
        required.stateChanged.connect(lambda _: self.changed.emit())

        desc = QtWidgets.QLineEdit(port.description or "")
        desc.textChanged.connect(lambda _: self.changed.emit())

        schema_btn = QtWidgets.QPushButton("Edit…")
        schema_btn.setProperty("schema", port.valueSchema)
        schema_btn.clicked.connect(lambda _=False, r=row: self._edit_schema(r))

        self._table.setCellWidget(row, 0, name)
        self._table.setCellWidget(row, 1, type_box)
        self._table.setCellWidget(row, 2, required)
        self._table.setCellWidget(row, 3, desc)
        self._table.setCellWidget(row, 4, schema_btn)

    def _edit_schema(self, row: int) -> None:
        btn = self._table.cellWidget(row, 4)
        if not isinstance(btn, QtWidgets.QPushButton):
            return
        current = btn.property("schema")
        if not isinstance(current, F8DataTypeSchema):
            current = _make_schema_by_type("any")
        dlg = SchemaJsonDialog(title="Edit Port Schema", initial_schema=current, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        btn.setProperty("schema", dlg.schema())
        type_box = self._table.cellWidget(row, 1)
        if isinstance(type_box, QtWidgets.QComboBox):
            type_box.setCurrentText(_schema_type_label(dlg.schema()))
        self.changed.emit()

    def _on_type_changed(self, row: int) -> None:
        type_box = self._table.cellWidget(row, 1)
        btn = self._table.cellWidget(row, 4)
        if not isinstance(type_box, QtWidgets.QComboBox) or not isinstance(btn, QtWidgets.QPushButton):
            return
        btn.setProperty("schema", _make_schema_by_type(type_box.currentText()))
        self.changed.emit()

    def _on_add(self) -> None:
        used = {p.name for p in self.ports() if p.name}
        idx = 1
        candidate = "port"
        while candidate in used:
            idx += 1
            candidate = f"port{idx}"
        self._append_row(F8DataPortSpec(name=candidate, valueSchema=string_schema(), required=True))
        self.changed.emit()

    def _on_remove(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        self._table.removeRow(row)
        self.changed.emit()


class StateDefTable(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, *, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._title = QtWidgets.QLabel(title)
        self._table = QtWidgets.QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(["name", "label", "access", "type", "required", "language", "schema"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self._add = QtWidgets.QPushButton("Add")
        self._remove = QtWidgets.QPushButton("Remove")
        self._add.clicked.connect(self._on_add)
        self._remove.clicked.connect(self._on_remove)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self._add)
        btns.addWidget(self._remove)
        btns.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._title)
        layout.addWidget(self._table)
        layout.addLayout(btns)

    def set_fields(self, fields: list[F8StateSpec]) -> None:
        self._table.setRowCount(0)
        for field in fields:
            self._append_row(field)

    def fields(self) -> list[F8StateSpec]:
        fields: list[F8StateSpec] = []
        for row in range(self._table.rowCount()):
            name = self._cell(row, 0, QtWidgets.QLineEdit)
            label = self._cell(row, 1, QtWidgets.QLineEdit)
            access = self._cell(row, 2, QtWidgets.QComboBox)
            type_box = self._cell(row, 3, QtWidgets.QComboBox)
            required = self._cell(row, 4, QtWidgets.QCheckBox)
            language = self._cell(row, 5, QtWidgets.QLineEdit)
            schema_btn = self._cell(row, 6, QtWidgets.QPushButton)
            if not all([name, label, access, type_box, required, language, schema_btn]):
                continue

            schema = schema_btn.property("schema")
            if not isinstance(schema, F8DataTypeSchema):
                schema = _make_schema_by_type(type_box.currentText())

            fields.append(
                F8StateSpec(
                    name=name.text().strip(),
                    label=(label.text().strip() or None),
                    valueSchema=schema,
                    access=F8StateAccess(access.currentText()),
                    required=bool(required.isChecked()),
                    language=(language.text().strip() or None),
                )
            )
        return fields

    def _cell(self, row: int, col: int, t: type[QtWidgets.QWidget]) -> QtWidgets.QWidget | None:
        w = self._table.cellWidget(row, col)
        return w if isinstance(w, t) else None

    def _append_row(self, field: F8StateSpec) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        name = QtWidgets.QLineEdit(field.name)
        name.textChanged.connect(lambda _: self.changed.emit())

        label = QtWidgets.QLineEdit(field.label or "")
        label.textChanged.connect(lambda _: self.changed.emit())

        access = QtWidgets.QComboBox()
        access.addItems([a.value for a in F8StateAccess])
        access.setCurrentText(field.access.value)
        access.currentTextChanged.connect(lambda _: self.changed.emit())

        type_box = QtWidgets.QComboBox()
        type_box.addItems(
            [
                F8PrimitiveTypeEnum.string.value,
                F8PrimitiveTypeEnum.number.value,
                F8PrimitiveTypeEnum.integer.value,
                F8PrimitiveTypeEnum.boolean.value,
                "object",
                "array",
                "any",
            ]
        )
        type_box.setCurrentText(_schema_type_label(field.valueSchema))
        type_box.currentTextChanged.connect(lambda _: self._on_type_changed(row))

        required = QtWidgets.QCheckBox()
        required.setChecked(bool(field.required))
        required.stateChanged.connect(lambda _: self.changed.emit())

        language = QtWidgets.QLineEdit(field.language or "")
        language.textChanged.connect(lambda _: self.changed.emit())

        schema_btn = QtWidgets.QPushButton("Edit…")
        schema_btn.setProperty("schema", field.valueSchema)
        schema_btn.clicked.connect(lambda _=False, r=row: self._edit_schema(r))

        self._table.setCellWidget(row, 0, name)
        self._table.setCellWidget(row, 1, label)
        self._table.setCellWidget(row, 2, access)
        self._table.setCellWidget(row, 3, type_box)
        self._table.setCellWidget(row, 4, required)
        self._table.setCellWidget(row, 5, language)
        self._table.setCellWidget(row, 6, schema_btn)

    def _edit_schema(self, row: int) -> None:
        btn = self._table.cellWidget(row, 6)
        if not isinstance(btn, QtWidgets.QPushButton):
            return
        current = btn.property("schema")
        if not isinstance(current, F8DataTypeSchema):
            current = _make_schema_by_type("any")
        dlg = SchemaJsonDialog(title="Edit State Schema", initial_schema=current, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        btn.setProperty("schema", dlg.schema())
        type_box = self._table.cellWidget(row, 3)
        if isinstance(type_box, QtWidgets.QComboBox):
            type_box.setCurrentText(_schema_type_label(dlg.schema()))
        self.changed.emit()

    def _on_type_changed(self, row: int) -> None:
        type_box = self._table.cellWidget(row, 3)
        btn = self._table.cellWidget(row, 6)
        if not isinstance(type_box, QtWidgets.QComboBox) or not isinstance(btn, QtWidgets.QPushButton):
            return
        btn.setProperty("schema", _make_schema_by_type(type_box.currentText()))
        self.changed.emit()

    def _on_add(self) -> None:
        used = {f.name for f in self.fields() if f.name}
        idx = 1
        candidate = "state"
        while candidate in used:
            idx += 1
            candidate = f"state{idx}"
        self._append_row(
            F8StateSpec(name=candidate, label=None, valueSchema=string_schema(), access=F8StateAccess.rw)
        )
        self.changed.emit()

    def _on_remove(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        self._table.removeRow(row)
        self.changed.emit()


class NodeSpecEditorWidget(QtWidgets.QWidget):
    specApplied = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._node: GenericNode | None = None
        self._spec: F8OperatorSpec | None = None

        self._empty = QtWidgets.QLabel("Select a node to edit spec.")
        self._empty.setAlignment(QtCore.Qt.AlignCenter)

        self._exec_in = EditableStringList(label="execInPorts")
        self._exec_out = EditableStringList(label="execOutPorts")
        self._data_in = DataPortTable(title="dataInPorts")
        self._data_out = DataPortTable(title="dataOutPorts")
        self._states = StateDefTable(title="states")

        self._apply = QtWidgets.QPushButton("Apply Spec Changes")
        self._apply.clicked.connect(self._apply_to_node)

        self._raw_json = QtWidgets.QPushButton("Edit Full Spec JSON...")
        self._raw_json.clicked.connect(self._edit_full_spec_json)

        splitter = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(self._exec_in)
        left_layout.addWidget(self._exec_out)
        left_layout.addStretch(1)
        splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(self._data_in)
        right_layout.addWidget(self._data_out)
        right_layout.addWidget(self._states)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self._raw_json)
        btns.addStretch(1)
        btns.addWidget(self._apply)

        self._content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(self._content)
        content_layout.addWidget(splitter)
        content_layout.addLayout(btns)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._empty)
        layout.addWidget(self._content)
        self._set_empty_visible(True)

    def set_node(self, node: GenericNode | None) -> None:
        self._node = node
        if node is None or not isinstance(node.spec, F8OperatorSpec):
            self._spec = None
            self._set_empty_visible(True)
            return
        self._spec = node.spec.model_copy(deep=True)
        self._sync_ui_from_spec()
        self._set_empty_visible(False)

    def _set_empty_visible(self, is_empty: bool) -> None:
        self._empty.setVisible(is_empty)
        self._content.setVisible(not is_empty)

    def _sync_ui_from_spec(self) -> None:
        if self._spec is None:
            return
        self._exec_in.set_items(self._spec.execInPorts or [])
        self._exec_out.set_items(self._spec.execOutPorts or [])
        self._data_in.set_ports(self._spec.dataInPorts or [])
        self._data_out.set_ports(self._spec.dataOutPorts or [])
        self._states.set_fields(self._spec.states or [])

    def _collect_spec_from_ui(self) -> F8OperatorSpec | None:
        if self._spec is None:
            return None
        spec = self._spec.model_copy(deep=True)
        spec.execInPorts = self._exec_in.items()
        spec.execOutPorts = self._exec_out.items()
        spec.dataInPorts = self._data_in.ports()
        spec.dataOutPorts = self._data_out.ports()
        spec.states = self._states.fields()
        try:
            validated = F8OperatorSpec.model_validate(spec.model_dump(mode="json"))
        except Exception:
            return None
        errors = self._validate_operator_spec(validated)
        if errors:
            QtWidgets.QMessageBox.warning(self, "Invalid Spec", "\n".join(errors))
            return None
        return validated

    @staticmethod
    def _find_duplicates(values: list[str]) -> list[str]:
        seen: set[str] = set()
        dupes: set[str] = set()
        for value in values:
            if value in seen:
                dupes.add(value)
            else:
                seen.add(value)
        return sorted(dupes)

    def _validate_operator_spec(self, spec: F8OperatorSpec) -> list[str]:
        errors: list[str] = []

        exec_in = [p.strip() for p in (spec.execInPorts or []) if p.strip()]
        exec_out = [p.strip() for p in (spec.execOutPorts or []) if p.strip()]
        data_in = [p.name.strip() for p in (spec.dataInPorts or []) if p.name and p.name.strip()]
        data_out = [p.name.strip() for p in (spec.dataOutPorts or []) if p.name and p.name.strip()]
        states = [s.name.strip() for s in (spec.states or []) if s.name and s.name.strip()]

        if any(not p.strip() for p in (spec.execInPorts or [])):
            errors.append("execInPorts contains an empty name.")
        if any(not p.strip() for p in (spec.execOutPorts or [])):
            errors.append("execOutPorts contains an empty name.")
        if any(not p.name.strip() for p in (spec.dataInPorts or []) if p.name is not None):
            errors.append("dataInPorts contains an empty name.")
        if any(not p.name.strip() for p in (spec.dataOutPorts or []) if p.name is not None):
            errors.append("dataOutPorts contains an empty name.")
        if any(not s.name.strip() for s in (spec.states or []) if s.name is not None):
            errors.append("states contains an empty name.")

        dup = self._find_duplicates(exec_in)
        if dup:
            errors.append(f"Duplicate execInPorts: {', '.join(dup)}")
        dup = self._find_duplicates(exec_out)
        if dup:
            errors.append(f"Duplicate execOutPorts: {', '.join(dup)}")
        dup = self._find_duplicates(data_in)
        if dup:
            errors.append(f"Duplicate dataInPorts: {', '.join(dup)}")
        dup = self._find_duplicates(data_out)
        if dup:
            errors.append(f"Duplicate dataOutPorts: {', '.join(dup)}")
        dup = self._find_duplicates(states)
        if dup:
            errors.append(f"Duplicate states: {', '.join(dup)}")

        return errors

    def _confirm_rebuild(self) -> bool:
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("Rebuild Node Ports?")
        msg.setText("Applying spec changes will rebuild ports and may disconnect existing links.")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        return msg.exec_() == QtWidgets.QMessageBox.Ok

    def _apply_to_node(self) -> None:
        if self._node is None or self._spec is None:
            return
        next_spec = self._collect_spec_from_ui()
        if next_spec is None:
            return
        if next_spec.operatorClass != self._node.spec.operatorClass:
            QtWidgets.QMessageBox.warning(self, "Unsupported", "Changing operatorClass on an instance is not supported.")
            return
        if not self._confirm_rebuild():
            return
        try:
            self._node.apply_spec(next_spec)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply Failed", str(exc))
            return
        self._spec = next_spec.model_copy(deep=True)
        self._sync_ui_from_spec()
        self.specApplied.emit(self._node)

    def _edit_full_spec_json(self) -> None:
        if self._spec is None:
            return
        dlg = JsonTextDialog(
            title="Edit Full Spec (JSON)",
            initial_json=_as_json(self._spec.model_dump(mode="json")),
            parent=self,
        )
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            payload = json.loads(dlg.json_text())
            next_spec = F8OperatorSpec.model_validate(payload)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Spec", str(exc))
            return
        if self._node and next_spec.operatorClass != self._node.spec.operatorClass:
            QtWidgets.QMessageBox.warning(self, "Unsupported", "Changing operatorClass on an instance is not supported.")
            return
        self._spec = next_spec.model_copy(deep=True)
        self._sync_ui_from_spec()


class NodePropertyEditorWidget(QtWidgets.QWidget):
    """
    Custom inspector replacing NodeGraphQt's PropertiesBinWidget.

    - schema-driven state editor (nested objects/arrays)
    - access enforcement (ro/rw/wo/init)
    - per-node spec editing (ports + state definitions)
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._header = QtWidgets.QLabel("No selection")
        self._header.setStyleSheet("font-weight: 600;")
        self._header.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self._tabs = QtWidgets.QTabWidget()
        self._state = NodeStateEditorWidget()
        self._spec = NodeSpecEditorWidget()
        self._spec.specApplied.connect(self._on_spec_applied)
        self._tabs.addTab(self._state, "State")
        self._tabs.addTab(self._spec, "Spec")

        self._multi = QtWidgets.QLabel("Multi-selection editing is not supported yet.")
        self._multi.setAlignment(QtCore.Qt.AlignCenter)
        self._multi.setVisible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._header)
        layout.addWidget(self._multi)
        layout.addWidget(self._tabs, 1)

    def _on_spec_applied(self, node: object) -> None:
        if isinstance(node, GenericNode):
            self._state.set_node(node)

    def set_selected_nodes(self, nodes: list[Any]) -> None:
        generic_nodes = [n for n in nodes if isinstance(n, GenericNode)]
        if len(generic_nodes) != 1:
            self._multi.setVisible(len(generic_nodes) > 1)
            if not generic_nodes:
                self._header.setText("No selection")
            else:
                self._header.setText(f"{len(generic_nodes)} nodes selected")
            self._state.set_node(None)
            self._spec.set_node(None)
            return

        node = generic_nodes[0]
        self._multi.setVisible(False)
        self._header.setText(f"{node.name()}  ({node.spec.operatorClass})")
        try:
            node.ensure_state_properties()
        except Exception:
            pass
        self._state.set_node(node)
        self._spec.set_node(node)
