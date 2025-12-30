from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
import dearpygui.dearpygui as dpg

from .operator import Access, OperatorGraph, OperatorInstance, OperatorRegistry
from .renderer import RendererRegistry, BaseOpRenderer, NodeAttrUserData, LinkUserData


class OperatorGraphEditor:
    """
    DearPyGui-based editor for a single operator graph.

    - Palette lists registry operators; double-click inside the node editor to spawn via picker, or click palette buttons to spawn.
    - Enforces unique node ids.
    - Separate link handling for exec/data/state with cardinality checks delegated to OperatorGraph.
    """

    def __init__(
        self,
        *,
        operatorCls_registry: OperatorRegistry | None = None,
        graph: OperatorGraph | None = None,
        rendererCls_registry: RendererRegistry | None = None,
    ) -> None:
        self.graph = graph or OperatorGraph()
        self.operatorCls_registry = operatorCls_registry or OperatorRegistry()
        self.rendererCls_registry = rendererCls_registry or RendererRegistry()

        self._renderers = {}

        self._node_editor_id: int | None = None
        self._node_counter = 0
        self._operator_menu_id: int | None = None
        self._pending_spawn_pos: tuple[float, float] | None = None
        self._save_dialog_id: int | None = None
        self._load_dialog_id: int | None = None
        # Ensure default renderer is present.
        self.rendererCls_registry.register("default", BaseOpRenderer, overwrite=True)

    def run(self) -> None:
        """Start the DearPyGui event loop."""
        dpg.create_context()

        with dpg.window(label="PyEngine Graph UI", width=1280, height=720) as main_win:
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Load graph...", callback=self._on_click_load)
                    dpg.add_menu_item(label="Save graph...", callback=self._on_click_save)
            self._node_editor_id = dpg.add_node_editor(
                callback=self._on_link,
                delink_callback=self._on_delink,
                minimap=True,
                minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
            )
            with dpg.handler_registry():
                dpg.add_mouse_double_click_handler(callback=self._on_double_click)
                dpg.add_key_press_handler(key=dpg.mvKey_Delete, callback=self._on_delete_selected)
                dpg.add_key_press_handler(key=dpg.mvKey_Back, callback=self._on_delete_selected)

        # with dpg.window(label='Operators', width=200, height=400):
        #     self._build_palette()

        self._build_operator_menu()
        self._build_file_dialogs()
        self._rebuild_ui_from_graph()

        dpg.create_viewport(title="PyEngine Graph UI", width=1280, height=720)

        # with dpg.theme() as theme:
        #     with dpg.theme_component():
        #         dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, 8, 8, category=dpg.mvThemeCat_Nodes)

        # dpg.show_debug()
        dpg.show_item_registry()

        # dpg.bind_theme(theme)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(main_win, True)
        dpg.start_dearpygui()
        dpg.destroy_context()

    # UI construction helpers -------------------------------------------------
    def _build_palette(self) -> None:
        for spec in self.operatorCls_registry.all():
            label = spec.label or spec.operatorClass
            dpg.add_button(
                label=label,
                width=-1,
                callback=self._on_pick_operator,
                user_data={"operatorClass": spec.operatorClass, "pos": None},
            )
            dpg.add_text(spec.operatorClass, indent=12)

    def _build_operator_menu(self) -> None:
        """Popup menu shown on node-editor double click to choose an operator."""
        self._operator_menu_id = dpg.add_window(
            label="Add operator",
            show=False,
            no_title_bar=False,
            modal=True,
            no_resize=True,
            autosize=True,
            no_move=True,
        )
        for spec in self.operatorCls_registry.all():
            label = spec.label or spec.operatorClass
            dpg.add_button(
                label=f"Add {label}",
                width=-1,
                parent=self._operator_menu_id,
                callback=self._on_pick_operator,
                user_data={"operatorClass": spec.operatorClass, "pos": None},
            )
        dpg.add_button(
            label="Cancel",
            width=-1,
            parent=self._operator_menu_id,
            callback=lambda: dpg.configure_item(self._operator_menu_id, show=False),
        )

    def _create_node(self, operator_class: str, pos: tuple[float, float]) -> None:
        template = self.operatorCls_registry.get(operator_class)
        node_id = self._unique_node_id(operator_class)
        instance = OperatorInstance.from_spec(template, id=node_id)
        self.graph.add_node(instance)
        self._build_node_ui(instance, pos)

    def _set_node_pos(self, node_tag: int, pos: tuple[float, float]) -> None:
        """Set node position using available DPG API (set_item_pos is supported)."""
        dpg.set_item_pos(node_tag, pos)

    def _build_node_ui(self, instance: OperatorInstance, pos: tuple[float, float]) -> None:
        """Create DearPyGui node widgets for an existing operator instance."""
        assert self._node_editor_id is not None
        with dpg.node(label=instance.spec.label, parent=self._node_editor_id, tag=instance.id) as node_tag:
            renderer_key = instance.spec.rendererClass or "default"
            rendererClass = self.rendererCls_registry.get(renderer_key)
            self._renderers[instance.id] = rendererClass(dpg.get_alias_id(node_tag), instance)

        self._set_node_pos(node_tag, pos)

    # Event handlers ----------------------------------------------------------
    def _on_double_click(self, sender: int, app_data: Any) -> None:
        if not self._node_editor_id or not dpg.is_item_hovered(self._node_editor_id):
            return
        self._pending_spawn_pos = dpg.get_mouse_pos(local=False)
        if self._operator_menu_id:
            dpg.configure_item(self._operator_menu_id, pos=self._pending_spawn_pos, show=True)

    def _on_pick_operator(self, sender: int, app_data: Any, user_data: dict[str, Any]) -> None:
        operator_class = user_data.get("operatorClass")
        pos = user_data.get("pos") or self._pending_spawn_pos or (50, 50)
        self._create_node(operator_class, pos)
        if self._operator_menu_id:
            dpg.configure_item(self._operator_menu_id, show=False)

    def _on_link(self, sender: int, app_data: Any, user_data: Any) -> None:
        from_attr, to_attr = app_data
        from_meta: NodeAttrUserData = dpg.get_item_user_data(from_attr)
        to_meta: NodeAttrUserData = dpg.get_item_user_data(to_attr)
        assert from_meta.direction == "out" or from_meta.direction == "in", "Invalid from_attr direction"

        if from_meta.kind != to_meta.kind:
            print("Link rejected: mismatched kinds")
            return

        kind = from_meta.kind
        try:
            if kind == "exec":
                edge = self.graph.connect_exec(from_meta.instance_key, from_meta.port, to_meta.instance_key, to_meta.port)
            elif kind == "data":
                edge = self.graph.connect_data(from_meta.instance_key, from_meta.port, to_meta.instance_key, to_meta.port)
            elif kind == "state":
                edge = self.graph.connect_state(from_meta.instance_key, from_meta.port, to_meta.instance_key, to_meta.port)
            else:
                return
        except Exception as exc:  # noqa: BLE001
            print(f"Link rejected: {exc}")
            return

        link_tag = dpg.generate_uuid()
        dpg.add_node_link(
            from_attr,
            to_attr,
            parent=self._node_editor_id,
            tag=link_tag,
            user_data=LinkUserData(kind=kind, edge=edge, source_attr=from_attr, target_attr=to_attr),
        )

    def _on_delink(self, sender: int, app_data: Any, user_data: Any) -> None:
        link_tag = app_data
        self._remove_link(link_tag, delete_item=True)

    def _on_delete_selected(self, sender: int, app_data: Any) -> None:
        if not self._node_editor_id:
            return
        selected_links = dpg.get_selected_links(self._node_editor_id) or []
        for link_tag in list(selected_links):
            self._remove_link(link_tag, delete_item=True)
        selected_nodes = dpg.get_selected_nodes(self._node_editor_id) or []
        for node_tag in list(selected_nodes):
            self._remove_node(node_tag)

    # Helpers -----------------------------------------------------------------
    def _unique_node_id(self, operator_class: str) -> str:
        base = operator_class.split(".")[-1]
        while True:
            self._node_counter += 1
            candidate = f"{base}_{self._node_counter}"
            if candidate not in self.graph.nodes:
                return candidate

    def _attr_tag(self, node_id: str, *, kind: str, direction: str, port: str) -> str:
        prefix = "out" if direction == "out" else "in"
        return f"{node_id}_{prefix}_{kind}_{port}"

    def _link_tags(self) -> list[int | str]:
        if not self._node_editor_id or not dpg.does_item_exist(self._node_editor_id):
            return []
        return list(dpg.get_item_children(self._node_editor_id, 0) or [])

    def _get_link_meta(self, link_tag: int | str) -> LinkUserData | None:
        if not dpg.does_item_exist(link_tag):
            return None
        return dpg.get_item_user_data(link_tag)

    def _reset_editor_state(self) -> None:
        self._renderers.clear()
        if self._node_editor_id:
            dpg.delete_item(self._node_editor_id, children_only=True)

    def _reseed_node_counter(self) -> None:
        max_seen = self._node_counter
        for node_id in self.graph.nodes:
            parts = node_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                max_seen = max(max_seen, int(parts[1]))
        self._node_counter = max_seen

    def _node_pos_from_layout(self, layout_positions: dict[str, Any] | None, node_id: str) -> tuple[float, float]:
        positions = layout_positions or {}
        raw = positions.get(node_id)
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                return float(raw[0]), float(raw[1])
            except (TypeError, ValueError):
                pass
        return 50.0, 50.0

    def _rebuild_ui_from_graph(self, layout_positions: dict[str, Any] | None = None) -> None:
        if not self._node_editor_id:
            return
        self._reset_editor_state()
        positions_lookup = layout_positions or {}
        for index, instance in enumerate(self.graph.nodes.values()):
            pos = self._node_pos_from_layout(positions_lookup, instance.id)
            if instance.id not in positions_lookup:
                pos = (pos[0] + index * 30.0, pos[1] + index * 30.0)
            self._build_node_ui(instance, pos)
        self._reseed_node_counter()
        self._rebuild_links_from_graph()

    def _rebuild_links_from_graph(self) -> None:
        if not self._node_editor_id:
            return
        for edge in self.graph.exec_edges:
            self._add_link_from_edge(
                kind="exec",
                source_id=edge.source_id,
                target_id=edge.target_id,
                source_port=edge.out_port,
                target_port=edge.in_port,
                edge_obj=edge,
            )
        for edge in self.graph.data_edges:
            self._add_link_from_edge(
                kind="data",
                source_id=edge.source_id,
                target_id=edge.target_id,
                source_port=edge.out_port,
                target_port=edge.in_port,
                edge_obj=edge,
            )
        for edge in self.graph.state_edges:
            self._add_link_from_edge(
                kind="state",
                source_id=edge.source_id,
                target_id=edge.target_id,
                source_port=edge.source_field,
                target_port=edge.target_field,
                edge_obj=edge,
            )

    def _add_link_from_edge(
        self,
        *,
        kind: str,
        source_id: str,
        target_id: str,
        source_port: str,
        target_port: str,
        edge_obj: Any,
    ) -> None:
        src_attr = self._attr_tag(source_id, kind=kind, direction="out", port=source_port)
        dst_attr = self._attr_tag(target_id, kind=kind, direction="in", port=target_port)
        if not dpg.does_item_exist(src_attr) or not dpg.does_item_exist(dst_attr):
            print(f"Skip rebuilding {kind} link {source_id} -> {target_id}: missing pin widgets")
            return
        link_tag = dpg.generate_uuid()
        dpg.add_node_link(
            src_attr,
            dst_attr,
            parent=self._node_editor_id,
            tag=link_tag,
            user_data=LinkUserData(kind=kind, edge=edge_obj, source_attr=src_attr, target_attr=dst_attr),
        )

    def _remove_link(self, link_tag: int | str, *, delete_item: bool) -> None:
        link_meta = self._get_link_meta(link_tag)
        if link_meta:
            if link_meta.kind == "exec":
                self.graph.disconnect_exec(link_meta.edge)
            elif link_meta.kind == "data":
                self.graph.disconnect_data(link_meta.edge)
            elif link_meta.kind == "state":
                self.graph.disconnect_state(link_meta.edge)
        if delete_item and dpg.does_item_exist(link_tag):
            dpg.delete_item(link_tag)

    def _remove_node(self, node_id: int | str) -> None:
        node_id_str = str(node_id)
        # Remove any links touching this node first.
        for link_tag in self._link_tags():
            link_meta = self._get_link_meta(link_tag)
            if not link_meta:
                continue
            src_meta = (
                dpg.get_item_user_data(link_meta.source_attr)
                if dpg.does_item_exist(link_meta.source_attr)
                else None
            )
            dst_meta = (
                dpg.get_item_user_data(link_meta.target_attr)
                if dpg.does_item_exist(link_meta.target_attr)
                else None
            )
            src_node = getattr(src_meta, "node_id", None) if src_meta is not None else None
            dst_node = getattr(dst_meta, "node_id", None) if dst_meta is not None else None
            if (src_node and str(src_node) == node_id_str) or (dst_node and str(dst_node) == node_id_str):
                self._remove_link(link_tag, delete_item=True)

        # Drop node from graph/model.
        self.graph.remove_node(node_id_str)

        # Cleanup renderer/meta.
        self._renderers.pop(node_id_str, None)

        # Remove UI item.
        if dpg.does_item_exist(node_id):
            dpg.delete_item(node_id)

    def _build_file_dialogs(self) -> None:
        self._save_dialog_id = dpg.generate_uuid()
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_save_file_selected,
            tag=self._save_dialog_id,
            default_filename="graph",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".json", color=(0, 255, 255, 255))
            dpg.add_file_extension(".*")

        self._load_dialog_id = dpg.generate_uuid()
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_load_file_selected,
            tag=self._load_dialog_id,
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".json", color=(0, 255, 255, 255))
            dpg.add_file_extension(".*")

    def _on_click_save(self, sender: int, app_data: Any, user_data: Any) -> None:
        if self._save_dialog_id:
            dpg.configure_item(self._save_dialog_id, show=True)

    def _on_click_load(self, sender: int, app_data: Any, user_data: Any) -> None:
        if self._load_dialog_id:
            dpg.configure_item(self._load_dialog_id, show=True)

    def _on_save_file_selected(self, sender: int, app_data: Any, user_data: Any) -> None:
        path = app_data.get("file_path_name")
        if not path:
            return
        try:
            self._save_graph_to_path(path)
            print(f"Saved graph to {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save graph: {exc}")

    def _on_load_file_selected(self, sender: int, app_data: Any, user_data: Any) -> None:
        path = app_data.get("file_path_name")
        if not path:
            return
        try:
            self._load_graph_from_file(path)
            print(f"Loaded graph from {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load graph: {exc}")

    def _save_graph_to_path(self, path: str) -> None:
        payload = self.graph.to_dict()
        positions: dict[str, list[float]] = {}
        for node_id in self.graph.nodes:
            if dpg.does_item_exist(node_id):
                pos = dpg.get_item_pos(node_id)
                positions[node_id] = [float(pos[0]), float(pos[1])]
        snapshot = {"graph": payload, "layout": {"positions": positions}}
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(snapshot, fp, indent=2)

    def _load_graph_from_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        graph_payload = payload.get("graph", payload)
        layout = payload.get("layout", {})
        positions = layout.get("positions", {})
        self._apply_graph_payload(graph_payload, layout_positions=positions)

    def _apply_graph_payload(
        self, graph_payload: dict[str, Any], *, layout_positions: dict[str, Any] | None = None
    ) -> None:
        self.graph.load_dict(graph_payload, registry=self.operatorCls_registry)
        self._rebuild_ui_from_graph(layout_positions)
