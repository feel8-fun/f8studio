from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from qtpy import QtCore, QtWidgets

from f8pysdk import F8OperatorSpec, F8ServiceSpec

from ..ui_notifications import show_info, show_warning
from ..variants.variant_compose import build_variant_record_from_node
from ..variants.variant_ids import build_variant_node_type
from ..variants.variant_models import F8NodeVariantRecord
from ..variants.variant_repository import (
    delete_variant,
    export_to_json,
    import_from_json,
    is_variant_name_conflict,
    list_variants_for_base,
    normalize_variant_name,
    upsert_variant,
)
from ..variants.variant_events import subscribe_variants_changed


class _VariantMetaDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget | None,
        title: str,
        name: str,
        description: str,
        tags: list[str],
        name_validator: Callable[[str], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 220)
        self._name_validator = name_validator
        self._name = QtWidgets.QLineEdit(name, self)
        self._description = QtWidgets.QLineEdit(description, self)
        self._tags = QtWidgets.QLineEdit(", ".join(tags), self)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self._name)
        form.addRow("Description", self._description)
        form.addRow("Tags (comma-separated)", self._tags)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_accept_clicked)  # type: ignore[attr-defined]
        buttons.rejected.connect(self.reject)  # type: ignore[attr-defined]

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _on_accept_clicked(self) -> None:
        name = str(self._name.text() or "").strip()
        if not name:
            show_warning(self, "Invalid name", "Variant name cannot be empty.")
            return
        validator = self._name_validator
        if validator is not None:
            message = validator(name)
            if message:
                show_warning(self, "Invalid name", message)
                return
        self.accept()

    def values(self) -> tuple[str, str, list[str]]:
        tags = [s.strip() for s in str(self._tags.text() or "").split(",")]
        return (
            str(self._name.text() or "").strip(),
            str(self._description.text() or "").strip(),
            [t for t in tags if t],
        )


class NodeVariantManagerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget | None,
        base_node_type: str,
        base_node_name: str,
        node_graph: Any,
    ) -> None:
        super().__init__(parent)
        self._base_node_type = str(base_node_type or "").strip()
        self._base_node_name = str(base_node_name or "").strip() or self._base_node_type
        self._graph = node_graph
        self._variants: list[F8NodeVariantRecord] = []
        self._unsubscribe_variants_changed: Any | None = subscribe_variants_changed(self._on_variants_changed)
        self.setWindowTitle(f"Variants - {self._base_node_name}")
        self.resize(980, 620)

        self._list = QtWidgets.QListWidget(self)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)  # type: ignore[attr-defined]
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)  # type: ignore[attr-defined]
        self._raw = QtWidgets.QPlainTextEdit(self)
        self._raw.setReadOnly(True)
        self._raw.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        btn_add = QtWidgets.QPushButton("Save From Selected Node", self)
        btn_edit = QtWidgets.QPushButton("Edit Metadata", self)
        btn_delete = QtWidgets.QPushButton("Delete", self)
        btn_create = QtWidgets.QPushButton("Create On Canvas", self)
        btn_import = QtWidgets.QPushButton("Import...", self)
        btn_export = QtWidgets.QPushButton("Export...", self)
        btn_close = QtWidgets.QPushButton("Close", self)

        btn_add.clicked.connect(self._on_add_clicked)  # type: ignore[attr-defined]
        btn_edit.clicked.connect(self._on_edit_clicked)  # type: ignore[attr-defined]
        btn_delete.clicked.connect(self._on_delete_clicked)  # type: ignore[attr-defined]
        btn_create.clicked.connect(self._on_create_clicked)  # type: ignore[attr-defined]
        btn_import.clicked.connect(self._on_import_clicked)  # type: ignore[attr-defined]
        btn_export.clicked.connect(self._on_export_clicked)  # type: ignore[attr-defined]
        btn_close.clicked.connect(self.accept)  # type: ignore[attr-defined]

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_edit)
        btn_row.addWidget(btn_delete)
        btn_row.addWidget(btn_create)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_import)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        split.addWidget(self._list)
        split.addWidget(self._raw)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 6)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(btn_row)
        layout.addWidget(split, 1)

        self.destroyed.connect(self._on_destroyed)  # type: ignore[attr-defined]
        self._reload()

    def _on_destroyed(self, _obj: Any) -> None:
        unsubscribe = self._unsubscribe_variants_changed
        self._unsubscribe_variants_changed = None
        if unsubscribe is not None:
            unsubscribe()

    def _on_variants_changed(self) -> None:
        self._reload()

    def _reload(self) -> None:
        self._variants = list_variants_for_base(self._base_node_type)
        self._list.clear()
        for v in self._variants:
            item = QtWidgets.QListWidgetItem(f"{v.name}    [{', '.join(v.tags) if v.tags else 'no-tags'}]")
            item.setToolTip(v.description or v.name)
            item.setData(QtCore.Qt.UserRole, v.variantId)
            self._list.addItem(item)
        self._on_selection_changed()

    def _selected_variant(self) -> F8NodeVariantRecord | None:
        item = self._list.currentItem()
        if item is None:
            return None
        variant_id = str(item.data(QtCore.Qt.UserRole) or "").strip()
        if not variant_id:
            return None
        for v in self._variants:
            if str(v.variantId) == variant_id:
                return v
        return None

    def _on_selection_changed(self) -> None:
        selected = self._selected_variant()
        if selected is None:
            self._raw.setPlainText("")
            return
        self._raw.setPlainText(json.dumps(selected.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str))

    def _on_item_double_clicked(self, _item: QtWidgets.QListWidgetItem) -> None:
        self._on_create_clicked()

    def _find_selected_base_node(self) -> Any | None:
        graph = self._graph
        if graph is None:
            return None
        for n in list(graph.selected_nodes() or []):
            if str(n.type_ or "").strip() == self._base_node_type:
                return n
        return None

    def _on_add_clicked(self) -> None:
        node = self._find_selected_base_node()
        if node is None:
            show_info(
                self,
                "No matching selected node",
                f"Please select a node of type:\n{self._base_node_type}\nthen try again.",
            )
            return
        spec = node.spec
        if not isinstance(spec, (F8OperatorSpec, F8ServiceSpec)):
            show_warning(self, "Unsupported node", "Selected node has no typed spec.")
            return
        node_display_name = ""
        try:
            node_display_name = str(node.name() or "").strip()
        except (AttributeError, RuntimeError, TypeError):
            node_display_name = ""
        dlg = _VariantMetaDialog(
            parent=self,
            title="Save Variant",
            name=str(node_display_name or node.NODE_NAME or spec.label or self._base_node_name),
            description=str(spec.description or ""),
            tags=[str(t) for t in list(spec.tags or [])],
            name_validator=self._validate_new_variant_name,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        name, description, tags = dlg.values()
        record = build_variant_record_from_node(node=node, name=name, description=description, tags=tags)
        try:
            upsert_variant(record)
        except ValueError as exc:
            show_warning(self, "Invalid name", str(exc))
            return

    def _on_edit_clicked(self) -> None:
        selected = self._selected_variant()
        if selected is None:
            return
        dlg = _VariantMetaDialog(
            parent=self,
            title="Edit Variant Metadata",
            name=selected.name,
            description=selected.description,
            tags=list(selected.tags or []),
            name_validator=lambda candidate: self._validate_edit_variant_name(candidate, selected.variantId),
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        name, description, tags = dlg.values()
        payload = selected.model_dump(mode="json")
        payload["name"] = name
        payload["description"] = description
        payload["tags"] = tags
        payload["updatedAt"] = F8NodeVariantRecord.now_iso()
        try:
            upsert_variant(F8NodeVariantRecord.model_validate(payload))
        except ValueError as exc:
            show_warning(self, "Invalid name", str(exc))
            return

    def _validate_new_variant_name(self, candidate: str) -> str | None:
        normalized_name = normalize_variant_name(candidate)
        if is_variant_name_conflict(self._base_node_type, normalized_name):
            return f"Variant name '{normalized_name}' already exists. Please rename."
        return None

    def _validate_edit_variant_name(self, candidate: str, variant_id: str) -> str | None:
        normalized_name = normalize_variant_name(candidate)
        if is_variant_name_conflict(
            self._base_node_type,
            normalized_name,
            exclude_variant_id=variant_id,
        ):
            return f"Variant name '{normalized_name}' already exists. Please rename."
        return None

    def _on_delete_clicked(self) -> None:
        selected = self._selected_variant()
        if selected is None:
            return
        reply = QtWidgets.QMessageBox.question(self, "Delete variant", f"Delete variant '{selected.name}'?")
        if reply != QtWidgets.QMessageBox.Yes:
            return
        delete_variant(selected.variantId)

    def _on_create_clicked(self) -> None:
        selected = self._selected_variant()
        if selected is None:
            return
        graph = self._graph
        if graph is None:
            return
        variant_node_type = build_variant_node_type(str(selected.variantId))
        placement_label = f"{self._base_node_name}\n - {selected.name}"
        graph.begin_node_placement(variant_node_type, placement_label)

    def _on_import_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Variant Library JSON",
            "",
            "JSON (*.json);;All Files (*)",
        )
        p = str(path or "").strip()
        if not p:
            return
        mode = QtWidgets.QMessageBox.question(
            self,
            "Import mode",
            "Merge into existing library?\n\nYes = Merge\nNo = Replace",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
        )
        if mode == QtWidgets.QMessageBox.Cancel:
            return
        try:
            import_from_json(p, mode="merge" if mode == QtWidgets.QMessageBox.Yes else "replace")
        except Exception as exc:
            show_warning(self, "Import failed", str(exc))
            return

    def _on_export_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Variant Library JSON",
            "nodeVariants.json",
            "JSON (*.json);;All Files (*)",
        )
        p = str(path or "").strip()
        if not p:
            return
        try:
            out = export_to_json(p)
        except Exception as exc:
            show_warning(self, "Export failed", str(exc))
            return
        show_info(self, "Exported", f"Saved:\n{out}")
