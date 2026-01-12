from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from qtpy import QtCore, QtWidgets

from f8pysdk import F8PrimitiveTypeEnum, F8ServiceSpec, F8StateAccess, F8StateSpec
from f8pysdk.schema_helpers import schema_default, schema_type


def _schema_type_label(schema: Any) -> str:
    try:
        t = schema_type(schema)
    except Exception:
        return "any"
    if isinstance(t, F8PrimitiveTypeEnum):
        return t.value
    return str(t)


def _to_json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return "null"


@dataclass(frozen=True)
class _NodeSelection:
    node: Any | None


class _JsonDialog(QtWidgets.QDialog):
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

        self._value: Any | None = None

    def value(self) -> Any:
        if self._value is None:
            raise RuntimeError("Value not validated yet.")
        return self._value

    def _on_accept(self) -> None:
        try:
            self._value = json.loads(self._editor.toPlainText())
        except Exception as exc:
            self._error.setText(str(exc))
            return
        self.accept()


class F8NodePropertyEditorWidget(QtWidgets.QWidget):
    """
    Minimal schema-driven property editor for f8pystudio.

    Replaces NodeGraphQt's built-in property editor.
    """

    def __init__(self, *, node_graph: Any, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._graph = node_graph
        self._selection = _NodeSelection(node=None)
        self._updating = False

        self._header = QtWidgets.QLabel("No selection")
        self._header.setStyleSheet("font-weight: 600;")
        self._header.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

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
        layout.addWidget(self._header)
        layout.addWidget(self._empty)
        layout.addWidget(self._scroll, 1)

        self._set_empty_visible(True)
        self._wire_graph()
        self._sync_from_graph()

    def _wire_graph(self) -> None:
        for sig_name in ("node_selection_changed", "node_selected"):
            try:
                sig = getattr(self._graph, sig_name, None)
                if sig is not None:
                    sig.connect(lambda *_a: self._sync_from_graph())
            except Exception:
                pass
        try:
            sig = getattr(self._graph, "property_changed", None)
            if sig is not None:
                sig.connect(self._on_property_changed)
        except Exception:
            pass

    def _sync_from_graph(self) -> None:
        try:
            nodes = list(self._graph.selected_nodes() or [])
        except Exception:
            nodes = []
        node = nodes[0] if len(nodes) == 1 else None
        self.set_node(node)

    def set_node(self, node: Any | None) -> None:
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
            self._header.setText("No selection")
            self._set_empty_visible(True)
            return

        spec = getattr(node, "spec", None)
        spec_key = getattr(spec, "operatorClass", None) or getattr(spec, "serviceClass", None) or ""
        try:
            node_name = node.name()
        except Exception:
            node_name = str(getattr(node, "NODE_NAME", "") or "Node")

        self._header.setText(f"{node_name}\n{node.id}\n({spec_key})")

        fields = list(getattr(spec, "stateFields", None) or [])
        if not fields:
            self._set_empty_visible(True)
            return

        self._set_empty_visible(False)

        for field in fields:
            self._form_layout.insertWidget(self._form_layout.count() - 1, self._build_field_editor(node, field))

    def _on_property_changed(self, node: Any, name: str, value: Any) -> None:
        if self._updating:
            return
        if node is None or self._selection.node is None:
            return
        if node is not self._selection.node:
            return
        try:
            spec = getattr(node, "spec", None)
            fields = {str(f.name): f for f in (getattr(spec, "stateFields", None) or [])}
        except Exception:
            fields = {}
        if str(name) not in fields:
            return
        # Rebuild is simplest and robust; can be optimized later.
        self._rebuild()

    def _build_field_editor(self, node: Any, field: F8StateSpec) -> QtWidgets.QWidget:
        title = str(field.label or field.name)
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)

        meta_parts = [str(field.name), _schema_type_label(field.valueSchema), str(getattr(field.access, "value", field.access))]
        meta = QtWidgets.QLabel(" • ".join(meta_parts))
        meta.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        meta.setStyleSheet("color: #666;")
        layout.addWidget(meta)

        if field.description:
            desc = QtWidgets.QLabel(str(field.description))
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #666;")
            layout.addWidget(desc)

        editor = self._make_value_editor(node, field)
        layout.addWidget(editor)
        return box

    def _make_value_editor(self, node: Any, field: F8StateSpec) -> QtWidgets.QWidget:
        name = str(field.name)
        read_only = field.access == F8StateAccess.ro

        try:
            value = node.get_property(name)
        except Exception:
            try:
                value = schema_default(field.valueSchema)
            except Exception:
                value = None

        t = _schema_type_label(field.valueSchema)
        inner = getattr(field.valueSchema, "root", None)
        enum_values = getattr(inner, "enum", None) if inner is not None else None

        if enum_values:
            combo = QtWidgets.QComboBox()
            combo.addItems([str(v) for v in enum_values])
            try:
                combo.setCurrentText("" if value is None else str(value))
            except Exception:
                pass

            def _on_change(_idx: int) -> None:
                self._set_node_property(node, name, combo.currentText())

            combo.currentIndexChanged.connect(_on_change)
            combo.setEnabled(not read_only)
            return combo

        if t == F8PrimitiveTypeEnum.boolean.value:
            cb = QtWidgets.QCheckBox("Enabled")
            cb.setChecked(bool(value))

            def _on_toggle(state: int) -> None:
                self._set_node_property(node, name, bool(state))

            cb.stateChanged.connect(_on_toggle)
            cb.setEnabled(not read_only)
            return cb

        if t == F8PrimitiveTypeEnum.integer.value:
            spin = QtWidgets.QSpinBox()
            minimum = getattr(inner, "minimum", None) if inner is not None else None
            maximum = getattr(inner, "maximum", None) if inner is not None else None
            try:
                if minimum is not None:
                    spin.setMinimum(int(minimum))
                else:
                    spin.setMinimum(-2147483648)
                if maximum is not None:
                    spin.setMaximum(int(maximum))
                else:
                    spin.setMaximum(2147483647)
            except Exception:
                pass
            try:
                spin.setValue(int(value) if value is not None else int(schema_default(field.valueSchema) or 0))
            except Exception:
                spin.setValue(0)

            def _on_change(v: int) -> None:
                self._set_node_property(node, name, int(v))

            spin.valueChanged.connect(_on_change)
            spin.setEnabled(not read_only)
            return spin

        if t == F8PrimitiveTypeEnum.number.value:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            minimum = getattr(inner, "minimum", None) if inner is not None else None
            maximum = getattr(inner, "maximum", None) if inner is not None else None
            step = getattr(inner, "multipleOf", None) if inner is not None else None
            try:
                if minimum is not None:
                    spin.setMinimum(float(minimum))
                else:
                    spin.setMinimum(-1e18)
                if maximum is not None:
                    spin.setMaximum(float(maximum))
                else:
                    spin.setMaximum(1e18)
                if step is not None:
                    spin.setSingleStep(float(step))
            except Exception:
                pass
            try:
                spin.setValue(float(value) if value is not None else float(schema_default(field.valueSchema) or 0.0))
            except Exception:
                spin.setValue(0.0)

            def _on_change(v: float) -> None:
                self._set_node_property(node, name, float(v))

            spin.valueChanged.connect(_on_change)
            spin.setEnabled(not read_only)
            return spin

        if t == F8PrimitiveTypeEnum.string.value:
            edit = QtWidgets.QLineEdit("" if value is None else str(value))

            def _on_edit() -> None:
                self._set_node_property(node, name, edit.text())

            edit.editingFinished.connect(_on_edit)
            edit.setEnabled(not read_only)
            return edit

        # Fallback: JSON edit.
        btn = QtWidgets.QPushButton("Edit JSON…")
        btn.setEnabled(not read_only)

        preview = QtWidgets.QPlainTextEdit(_to_json_text(value))
        preview.setReadOnly(True)
        preview.setMaximumHeight(90)

        def _open() -> None:
            dlg = _JsonDialog(title=f"{name} (JSON)", initial_json=_to_json_text(value), parent=self)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            self._set_node_property(node, name, dlg.value())
            try:
                preview.setPlainText(_to_json_text(node.get_property(name)))
            except Exception:
                preview.setPlainText(_to_json_text(dlg.value()))

        btn.clicked.connect(_open)

        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(btn)
        layout.addWidget(preview)
        return box

    def _set_node_property(self, node: Any, name: str, value: Any) -> None:
        if node is None:
            return
        self._updating = True
        try:
            try:
                node.set_property(name, value)
            except Exception:
                # Some NodeGraphQt versions use `set_property(name, value, push_undo=False)`
                node.set_property(name, value, push_undo=True)  # type: ignore[call-arg]
        finally:
            self._updating = False

