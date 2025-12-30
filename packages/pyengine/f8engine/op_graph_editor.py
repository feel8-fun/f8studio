from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import dearpygui.dearpygui as dpg

from .operator import OperatorGraph, OperatorInstance, OperatorRegistry, StateEdge
from .generated.operator_spec import StateField
from .renderer.registry import RendererRegistry, BaseOpRenderer


@dataclass
class LinkMeta:
    kind: str  # exec | data | state
    edge: Any
    source_attr: int
    target_attr: int


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

        self._attr_meta: dict[int, dict[str, str]] = {}
        self._link_meta: dict[int, LinkMeta] = {}
        self._node_editor_id: int | None = None
        self._node_counter = 0
        self._operator_menu_id: int | None = None
        self._pending_spawn_pos: tuple[float, float] | None = None
        # Ensure default renderer is present.
        self.rendererCls_registry.register('default', BaseOpRenderer, overwrite=True)

    def run(self) -> None:
        """Start the DearPyGui event loop."""
        dpg.create_context()

        with dpg.window(label='PyEngine Graph UI', width=1280, height=720) as main_win:
            self._node_editor_id = dpg.add_node_editor(
                callback=self._on_link,
                delink_callback=self._on_delink,
                minimap=True,
                minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
            )
            with dpg.handler_registry():
                dpg.add_mouse_double_click_handler(callback=self._on_double_click)

        # with dpg.window(label='Operators', width=200, height=400):
        #     self._build_palette()

        self._build_operator_menu()

        dpg.create_viewport(title='PyEngine Graph UI', width=1280, height=720)

        with dpg.theme() as theme:
            with dpg.theme_component():
                dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding, 0, 8, category=dpg.mvThemeCat_Nodes)

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
                user_data={'operatorClass': spec.operatorClass, 'pos': None},
            )
            dpg.add_text(spec.operatorClass, indent=12)

    def _build_operator_menu(self) -> None:
        """Popup menu shown on node-editor double click to choose an operator."""
        self._operator_menu_id = dpg.add_window(
            label='Add operator',
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
                label=f'Add {label}',
                width=-1,
                parent=self._operator_menu_id,
                callback=self._on_pick_operator,
                user_data={'operatorClass': spec.operatorClass, 'pos': None},
            )
        dpg.add_button(
            label='Cancel',
            width=-1,
            parent=self._operator_menu_id,
            callback=lambda: dpg.configure_item(self._operator_menu_id, show=False),
        )

    def _create_node(self, operator_class: str, pos: tuple[float, float]) -> None:
        template = self.operatorCls_registry.get(operator_class)
        node_id = self._unique_node_id(operator_class)
        instance = OperatorInstance.from_spec(template, id=node_id)
        self.graph.add_node(instance)

        assert self._node_editor_id is not None
        with dpg.node(label=instance.spec.label, parent=self._node_editor_id, tag=node_id) as node_tag:
            renderer_key = instance.spec.rendererClass or 'default'
            rendererClass = self.rendererCls_registry.get(renderer_key)
            self._renderers[node_id] = rendererClass(node_id, instance)

        self._set_node_pos(node_tag, pos)

    def _set_node_pos(self, node_tag: int, pos: tuple[float, float]) -> None:
        """Set node position using available DPG API (set_item_pos is supported)."""
        dpg.set_item_pos(node_tag, pos)

    # Event handlers ----------------------------------------------------------
    def _on_double_click(self, sender: int, app_data: Any) -> None:
        if not self._node_editor_id or not dpg.is_item_hovered(self._node_editor_id):
            return
        self._pending_spawn_pos = dpg.get_mouse_pos(local=False)
        if self._operator_menu_id:
            dpg.configure_item(self._operator_menu_id, pos=self._pending_spawn_pos, show=True)

    def _on_pick_operator(self, sender: int, app_data: Any, user_data: dict[str, Any]) -> None:
        operator_class = user_data.get('operatorClass')
        pos = user_data.get('pos') or self._pending_spawn_pos or (50, 50)
        self._create_node(operator_class, pos)
        if self._operator_menu_id:
            dpg.configure_item(self._operator_menu_id, show=False)


    def _on_link(self, sender: int, app_data: Any, user_data: Any) -> None:
        from_attr, to_attr = app_data
        meta_a = self._attr_meta.get(from_attr)
        meta_b = self._attr_meta.get(to_attr)
        if not meta_a or not meta_b:
            return

        # Normalize so that source is always the out pin.
        if meta_a['direction'] == 'out':
            src_meta, dst_meta = meta_a, meta_b
            src_attr, dst_attr = from_attr, to_attr
        else:
            src_meta, dst_meta = meta_b, meta_a
            src_attr, dst_attr = to_attr, from_attr

        if src_meta['direction'] != 'out' or dst_meta['direction'] != 'in':
            return
        if src_meta['kind'] != dst_meta['kind']:
            return

        kind = src_meta['kind']
        try:
            if kind == 'exec':
                edge = self.graph.connect_exec(
                    src_meta['node_id'], src_meta['port'], dst_meta['node_id'], dst_meta['port']
                )
            elif kind == 'data':
                edge = self.graph.connect_data(
                    src_meta['node_id'], src_meta['port'], dst_meta['node_id'], dst_meta['port']
                )
            elif kind == 'state':
                edge = self.graph.connect_state(
                    src_meta['node_id'], src_meta['port'], dst_meta['node_id'], dst_meta['port']
                )
            else:
                return
        except Exception as exc:  # noqa: BLE001
            print(f'Link rejected: {exc}')
            return

        link_tag = dpg.generate_uuid()
        dpg.add_node_link(src_attr, dst_attr, parent=self._node_editor_id, tag=link_tag)
        self._link_meta[link_tag] = LinkMeta(
            kind=kind, edge=edge, source_attr=src_attr, target_attr=dst_attr
        )

    def _on_delink(self, sender: int, app_data: Any, user_data: Any) -> None:
        link_tag = app_data
        link_meta = self._link_meta.pop(link_tag, None)
        if not link_meta:
            return
        if link_meta.kind == 'exec':
            self.graph.disconnect_exec(link_meta.edge)
        elif link_meta.kind == 'data':
            self.graph.disconnect_data(link_meta.edge)
        elif link_meta.kind == 'state':
            self.graph.disconnect_state(link_meta.edge)

    # Helpers -----------------------------------------------------------------
    def _unique_node_id(self, operator_class: str) -> str:
        base = operator_class.split('.')[-1]
        while True:
            self._node_counter += 1
            candidate = f'{base}_{self._node_counter}'
            if candidate not in self.graph.nodes:
                return candidate
