from __future__ import annotations

from collections import defaultdict
from typing import Any

import json

from qtpy import QtCore, QtWidgets, QtGui
from NodeGraphQt import NodesTreeWidget
from NodeGraphQt.custom_widgets.nodes_tree import _BaseNodeTreeItem, TYPE_CATEGORY, TYPE_NODE

from ..nodegraph.spec_visibility import is_hidden_spec_node_class, typed_spec_template_or_none
from f8pysdk import F8OperatorSpec, F8ServiceSpec
from ..variants.variant_ids import build_variant_node_type
from ..variants.variant_repository import list_variants_for_base


class _F8StudioNodesTreeWidget(NodesTreeWidget):
    _ROLE_NODE_ID = int(QtCore.Qt.UserRole + 1)
    _ROLE_NODE_NAME = int(QtCore.Qt.UserRole + 2)
    _ROLE_BASE_NODE_ID = int(QtCore.Qt.UserRole + 3)
    _ROLE_VARIANT_ID = int(QtCore.Qt.UserRole + 4)
    _ROLE_IS_VARIANT = int(QtCore.Qt.UserRole + 5)
    _ROLE_VARIANT_NAME = int(QtCore.Qt.UserRole + 6)

    def __init__(self, parent: QtWidgets.QWidget | None = None, node_graph: Any | None = None) -> None:
        self._search_text = ""
        self._search_variants_enabled = False
        self._on_open_variant_manager: Any | None = None
        self._node_graph = node_graph
        self._variant_manager_dialogs: list[QtWidgets.QDialog] = []
        super().__init__(parent=parent, node_graph=node_graph)
        self.setColumnCount(1)
        self.setHeaderHidden(True)
        self.setIndentation(12)
        self.setUniformRowHeights(True)
        self.setDragEnabled(False)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.setStyleSheet(
            "QTreeView::item {"
            "  margin: 0px;"
            "  padding-top: 0px;"
            "  padding-bottom: 0px;"
            "}"
            "QTreeView::branch {"
            "  margin: 0px;"
            "  padding: 0px;"
            "}"
        )
        self.customContextMenuRequested.connect(self._on_context_menu_requested)  # type: ignore[attr-defined]
        self.itemClicked.connect(self._on_item_clicked)  # type: ignore[attr-defined]

    def set_search_text(self, text: str) -> None:
        value = str(text or "").strip().lower()
        if value == self._search_text:
            return
        self._search_text = value
        self.update()

    def set_search_variants_enabled(self, enabled: bool) -> None:
        val = bool(enabled)
        if val == self._search_variants_enabled:
            return
        self._search_variants_enabled = val
        self.update()

    def set_open_variant_manager_callback(self, callback: Any | None) -> None:
        self._on_open_variant_manager = callback

    def _build_search_blob(self, *, node_cls: Any, node_name: str, node_id: str) -> str:
        parts: list[str] = [str(node_name), str(node_id)]
        parts.append(str(node_cls.NODE_NAME))
        spec = typed_spec_template_or_none(node_cls)
        if spec is not None:
            if spec.label:
                parts.append(str(spec.label))
            if spec.description:
                parts.append(str(spec.description))
            parts.extend(str(tag) for tag in list(spec.tags or []))
            if isinstance(spec, F8OperatorSpec):
                parts.append(str(spec.serviceClass))
                parts.append(str(spec.operatorClass))
            else:
                parts.append(str(spec.serviceClass))
        return " ".join(parts).lower()

    def _matches_search(self, *, node_cls: Any, node_name: str, node_id: str) -> bool:
        query = self._search_text
        if not query:
            return True
        haystack = self._build_search_blob(node_cls=node_cls, node_name=node_name, node_id=node_id)
        for token in query.split():
            if token not in haystack:
                return False
        return True

    @staticmethod
    def _variant_search_blob(variant: Any) -> str:
        parts = [
            str(variant.name or ""),
            str(variant.description or ""),
            " ".join(str(t) for t in list(variant.tags or [])),
        ]
        spec = variant.spec if isinstance(variant.spec, dict) else {}
        if isinstance(spec, dict):
            parts.append(str(spec.get("label") or ""))
            parts.append(str(spec.get("description") or ""))
            parts.append(" ".join(str(t) for t in list(spec.get("tags") or [])))
        return " ".join(parts).lower()

    def _variant_matches_search(self, variant: Any) -> bool:
        query = self._search_text
        if not query:
            return True
        haystack = self._variant_search_blob(variant)
        for token in query.split():
            if token not in haystack:
                return False
        return True

    @staticmethod
    def _spec_description(spec: F8OperatorSpec | F8ServiceSpec) -> str:
        desc = str(spec.description or "").strip()
        if desc:
            return desc
        label = str(spec.label or "").strip()
        if label:
            return label
        return ""

    @staticmethod
    def _schema_to_json_text(schema_obj: Any) -> str:
        if schema_obj is None:
            return "{}"
        try:
            payload = schema_obj.model_dump(mode="json", by_alias=True)
        except Exception:
            payload = schema_obj
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(payload)

    @staticmethod
    def _md_code_block(text: str) -> str:
        body = str(text or "").strip()
        if not body:
            body = "{}"
        return f"```json\n{body}\n```"

    def _render_data_ports_md(self, title: str, ports: list[Any]) -> str:
        lines: list[str] = [f"## {title}"]
        if not ports:
            lines.append("_None_")
            return "\n".join(lines)
        for port in ports:
            lines.append(f"### `{port.name}`")
            lines.append(f"- **Description**: {port.description or ''}")
            lines.append(f"- **Required**: `{bool(port.required)}`")
            lines.append("**Schema**")
            lines.append(self._md_code_block(self._schema_to_json_text(port.valueSchema)))
        return "\n".join(lines)

    def _render_state_fields_md(self, fields: list[Any]) -> str:
        lines: list[str] = ["## State Fields"]
        if not fields:
            lines.append("_None_")
            return "\n".join(lines)
        for field in fields:
            lines.append(f"### `{field.name}`")
            lines.append(f"- **Label**: {field.label or ''}")
            lines.append(f"- **Access**: `{field.access}`")
            lines.append(f"- **Required**: `{bool(field.required)}`")
            lines.append(f"- **Show On Node**: `{bool(field.showOnNode)}`")
            lines.append(f"- **Description**: {field.description or ''}")
            lines.append("- **Schema**:")
            lines.append(self._md_code_block(self._schema_to_json_text(field.valueSchema)))
        return "\n".join(lines)

    def _render_operator_doc(self, spec: F8OperatorSpec) -> str:
        lines: list[str] = [f"# {spec.label or spec.operatorClass}"]
        lines.append(f"**Operator Class**: `{spec.operatorClass}`  ")
        lines.append(f"**Service Class**: `{spec.serviceClass}`  ")
        lines.append(f"**Version**: `{spec.version or ''}`")
        lines.append("")
        lines.append(spec.description or "_No description._")
        lines.append("")
        tags = ", ".join(str(t) for t in list(spec.tags or []))
        lines.append(f"**Tags**: {tags or '_none_'}")
        lines.append("")

        lines.append("## Exec Ports")
        lines.append(f"- **In**: {', '.join(str(x) for x in list(spec.execInPorts or [])) or '_none_'}")
        lines.append(f"- **Out**: {', '.join(str(x) for x in list(spec.execOutPorts or [])) or '_none_'}")
        lines.append("")
        lines.append("")
        lines.append(self._render_data_ports_md("Data In Ports", list(spec.dataInPorts or [])))
        lines.append("")
        lines.append(self._render_data_ports_md("Data Out Ports", list(spec.dataOutPorts or [])))
        lines.append("")
        lines.append(self._render_state_fields_md(list(spec.stateFields or [])))
        return "\n".join(lines)

    def _render_service_doc(self, spec: F8ServiceSpec) -> str:
        lines: list[str] = [f"# {spec.label or spec.serviceClass}"]
        lines.append(f"**Service Class**: `{spec.serviceClass}`  ")
        lines.append(f"**Version**: `{spec.version or ''}`")
        lines.append("")
        lines.append(spec.description or "_No description._")
        lines.append("")
        tags = ", ".join(str(t) for t in list(spec.tags or []))
        lines.append(f"**Tags**: {tags or '_none_'}")
        lines.append("")
        lines.append(self._render_data_ports_md("Data In Ports", list(spec.dataInPorts or [])))
        lines.append("")
        lines.append(self._render_data_ports_md("Data Out Ports", list(spec.dataOutPorts or [])))
        lines.append("")
        lines.append(self._render_state_fields_md(list(spec.stateFields or [])))
        lines.append("")
        lines.append("## Commands")
        commands = list(spec.commands or [])
        if not commands:
            lines.append("_None_")
        else:
            for c in commands:
                lines.append(f"### `{c.name}`")
                lines.append(f"- **Description**: {c.description or ''}")

        return "\n".join(lines)

    def _show_spec_dialog(self, *, node_id: str, node_name: str) -> None:
        node_cls = self._factory.nodes.get(node_id) if self._factory is not None else None
        if node_cls is None:
            return
        spec = typed_spec_template_or_none(node_cls)
        if spec is None:
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Node Info - {node_name}")
        dialog.resize(860, 620)

        title = QtWidgets.QLabel(f"{node_name}  ({node_id})", dialog)
        title.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        tabs = QtWidgets.QTabWidget(dialog)
        overview = QtWidgets.QTextBrowser(dialog)
        overview.setOpenExternalLinks(False)
        overview.setOpenLinks(False)
        overview.setStyleSheet(
            "QTextBrowser {"
            "  background: #1f2329;"
            "  border: 1px solid #2d333b;"
            "  color: #e6edf3;"
            "  font-size: 12px;"
            "}"
        )
        raw = QtWidgets.QPlainTextEdit(dialog)
        raw.setReadOnly(True)
        raw.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        if isinstance(spec, F8OperatorSpec):
            overview.setMarkdown(self._render_operator_doc(spec))
        else:
            overview.setMarkdown(self._render_service_doc(spec))

        raw.setPlainText(json.dumps(spec.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2, default=str))

        tabs.addTab(overview, "Overview")
        tabs.addTab(raw, "Raw JSON")

        close_btn = QtWidgets.QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)  # type: ignore[attr-defined]

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(title)
        layout.addWidget(tabs, 1)
        layout.addWidget(close_btn, 0, QtCore.Qt.AlignRight)
        dialog.exec()

    def _open_variant_manager(self, *, base_node_type: str, base_node_name: str) -> None:
        callback = self._on_open_variant_manager
        if callable(callback):
            callback(base_node_type=base_node_type, base_node_name=base_node_name)
            return
        try:
            from .node_variant_manager_dialog import NodeVariantManagerDialog

            dlg = NodeVariantManagerDialog(
                parent=self.window(),
                base_node_type=base_node_type,
                base_node_name=base_node_name,
                node_graph=self._node_graph,
            )
            dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            dlg.destroyed.connect(self._on_variant_manager_destroyed)  # type: ignore[attr-defined]
            self._variant_manager_dialogs.append(dlg)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Open variant manager failed", str(exc))

    def _on_variant_manager_destroyed(self, obj: Any) -> None:
        self._variant_manager_dialogs = [dialog for dialog in self._variant_manager_dialogs if dialog is not obj]

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        if item.type() != TYPE_NODE:
            return
        _ = column
        graph = self._node_graph
        if graph is None:
            return
        node_id = str(item.data(0, self._ROLE_NODE_ID) or "")
        if not node_id:
            return
        node_name = str(item.data(0, self._ROLE_NODE_NAME) or item.text(0))
        is_variant = bool(item.data(0, self._ROLE_IS_VARIANT))
        if is_variant:
            variant_name = str(item.data(0, self._ROLE_VARIANT_NAME) or "").strip()
            if variant_name:
                node_name = f"{node_name}\n - {variant_name}"
        graph.begin_node_placement(node_id, node_name)

    def _on_context_menu_requested(self, pos: QtCore.QPoint) -> None:
        item = self.itemAt(pos)
        if item is None or item.type() != TYPE_NODE:
            return
        menu = QtWidgets.QMenu(self)
        graph = self._node_graph
        if graph is None:
            return
        base_node_id = str(item.data(0, self._ROLE_BASE_NODE_ID) or item.data(0, self._ROLE_NODE_ID) or "")
        base_node_name = str(item.data(0, self._ROLE_NODE_NAME) or item.text(0))
        if not base_node_id:
            return

        action_info = menu.addAction("Show Details")
        action_manage = menu.addAction("Manage Variants...")

        variants = list_variants_for_base(base_node_id)
        if variants:
            variants_menu = menu.addMenu("Variants")
            variant_actions: dict[QtGui.QAction, tuple[str, str]] = {}
            for v in variants:
                act = variants_menu.addAction(str(v.name or v.variantId))
                variant_actions[act] = (str(v.variantId), str(v.name or ""))
        else:
            variant_actions = {}

        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        if chosen == action_info:
            self._show_spec_dialog(node_id=base_node_id, node_name=base_node_name)
            return
        if chosen == action_manage:
            self._open_variant_manager(base_node_type=base_node_id, base_node_name=base_node_name)
            return
        if chosen in variant_actions:
            variant_id, variant_name = variant_actions[chosen]
            variant_node_type = build_variant_node_type(variant_id)
            placement_label = str(base_node_name)
            if variant_name:
                placement_label = f"{base_node_name}\n - {variant_name}"
            graph.begin_node_placement(variant_node_type, placement_label)

    def _build_tree(self) -> None:
        self.clear()
        if self._factory is None:
            return

        show_variant_children = self._search_variants_enabled
        node_types_by_category: dict[str, list[tuple[str, str, list[Any]]]] = defaultdict(list)
        for node_name, node_ids in self._factory.names.items():
            for node_id_any in list(node_ids or []):
                node_id = str(node_id_any)
                node_cls = self._factory.nodes.get(node_id)
                if node_cls is None:
                    continue
                if is_hidden_spec_node_class(node_cls):
                    continue
                base_match = self._matches_search(node_cls=node_cls, node_name=str(node_name), node_id=node_id)
                matched_variants: list[Any] = []
                if show_variant_children:
                    variants = list_variants_for_base(node_id)
                    if not self._search_text:
                        matched_variants.extend(variants)
                    else:
                        for v in variants:
                            if self._variant_matches_search(v):
                                matched_variants.append(v)
                if not base_match and not matched_variants:
                    continue
                category = str(node_cls.__identifier__ or "")
                if not category:
                    category = ".".join(node_id.split(".")[:-1])
                node_types_by_category[category].append((node_id, str(node_name), matched_variants))

        self._category_items = {}
        for category in sorted(node_types_by_category.keys()):
            label = str(self._custom_labels.get(category, category))
            cat_item = _BaseNodeTreeItem(self, [label], type=TYPE_CATEGORY)
            cat_item.setFirstColumnSpanned(True)
            cat_item.setFlags(QtCore.Qt.ItemIsEnabled)
            cat_item.setSizeHint(0, QtCore.QSize(100, 22))
            self.addTopLevelItem(cat_item)
            cat_item.setExpanded(True)
            self._category_items[category] = cat_item

        for category, nodes_list in node_types_by_category.items():
            category_item = self._category_items.get(category)
            if category_item is None:
                continue
            for node_id, node_name, matched_variants in nodes_list:
                item = _BaseNodeTreeItem(category_item, [node_name], type=TYPE_NODE)
                item.setToolTip(0, node_id)
                item.setSizeHint(0, QtCore.QSize(100, 22))
                item.setData(0, self._ROLE_NODE_ID, node_id)
                item.setData(0, self._ROLE_BASE_NODE_ID, node_id)
                item.setData(0, self._ROLE_NODE_NAME, node_name)
                item.setData(0, self._ROLE_IS_VARIANT, False)
                item.setExpanded(False)
                category_item.addChild(item)

                node_cls = self._factory.nodes.get(node_id)
                if node_cls is not None:
                    spec = typed_spec_template_or_none(node_cls)
                    if spec is not None:
                        desc = self._spec_description(spec)
                        if desc:
                            item.setToolTip(0, f"{node_id}\n\n{desc}")

                for variant in matched_variants:
                    variant_node_type = build_variant_node_type(str(variant.variantId))
                    variant_text = f"|{variant.name}|"
                    variant_item = _BaseNodeTreeItem(item, [variant_text], type=TYPE_NODE)
                    variant_item.setToolTip(0, variant_node_type)
                    variant_item.setSizeHint(0, QtCore.QSize(100, 22))
                    variant_item.setData(0, self._ROLE_NODE_ID, variant_node_type)
                    variant_item.setData(0, self._ROLE_BASE_NODE_ID, node_id)
                    variant_item.setData(0, self._ROLE_NODE_NAME, node_name)
                    variant_item.setData(0, self._ROLE_VARIANT_ID, str(variant.variantId))
                    variant_item.setData(0, self._ROLE_IS_VARIANT, True)
                    variant_item.setData(0, self._ROLE_VARIANT_NAME, str(variant.name or ""))
                    item.addChild(variant_item)


class F8StudioNodeLibraryWidget(QtWidgets.QWidget):
    """
    Tree-based nodes browser with keyword search for Studio.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None, node_graph: Any | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Node Library")
        self._node_graph = node_graph

        self._search = QtWidgets.QLineEdit(self)
        self._search.setPlaceholderText("Search nodes (name, tags, description)")
        self._search_variants = QtWidgets.QCheckBox("Search Variants", self)
        self._search_variants.setChecked(False)
        self._tree = _F8StudioNodesTreeWidget(self, node_graph=node_graph)

        search_row = QtWidgets.QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 0)
        search_row.setSpacing(6)
        search_row.addWidget(self._search, 1)
        search_row.addWidget(self._search_variants, 0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(search_row)
        layout.addWidget(self._tree)

        self._search.textChanged.connect(self._on_search_text_changed)  # type: ignore[attr-defined]
        self._search_variants.toggled.connect(self._on_search_variants_toggled)  # type: ignore[attr-defined]
        if node_graph is not None:
            node_graph.nodes_registered.connect(self._on_nodes_registered)  # type: ignore[attr-defined]
            node_graph.node_placement_changed.connect(self._on_node_placement_changed)  # type: ignore[attr-defined]

    def _on_search_text_changed(self, text: str) -> None:
        self._tree.set_search_text(str(text or ""))

    def _on_search_variants_toggled(self, enabled: bool) -> None:
        self._tree.set_search_variants_enabled(bool(enabled))

    def _on_nodes_registered(self, _nodes: list[Any]) -> None:
        self._tree.update()

    def _on_node_placement_changed(self, active: bool, label: str) -> None:
        _ = label
        if bool(active):
            return
        self._tree.clearSelection()
        self._tree.setCurrentItem(None)

    def set_category_label(self, category: str, label: str) -> None:
        self._tree.set_category_label(category, label)

    def set_open_variant_manager_callback(self, callback: Any | None) -> None:
        self._tree.set_open_variant_manager_callback(callback)

    def update(self) -> None:
        self._tree.update()
