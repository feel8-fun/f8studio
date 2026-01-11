from NodeGraphQt import NodesPaletteWidget

from collections import defaultdict


class F8NodesPaletteWidget(NodesPaletteWidget):
    """Customized NodesPaletteWidget for F8PyStudio."""

    def __init__(self, parent=None, node_graph=None):
        super().__init__(parent, node_graph)

    def _on_nodes_registered(self, nodes):
        """
        Slot function when a new node has been registered into the node graph.

        Args:
            nodes (list[NodeObject]): node objects.
        """
        node_types = defaultdict(list)
        for node in nodes:
            name = node.NODE_NAME
            nid = node.type_
            category = node.__identifier__
            node_types[category].append((nid, name))

        update_tabs = False
        for category, nodes_list in node_types.items():
            if not update_tabs and category not in self._category_tabs:
                update_tabs = True
            grid_view = self._add_category_tab(category)
            for node_id, node_name in nodes_list:
                grid_view.add_item(node_name, node_id)

        if update_tabs:
            self._update_tab_labels()

    def _build_ui(self):
        """
        populate the ui
        """
        node_types = defaultdict(list)
        for nid, node_cls in self._factory.nodes.items():
            category = node_cls.__identifier__
            node_types[category].append((nid, node_cls.NODE_NAME))

        for category, nodes_list in node_types.items():
            grid_view = self._add_category_tab(category)
            for node_id, node_name in nodes_list:
                grid_view.add_item(node_name, node_id)

    def update(self):
        """
        Update and refresh the node palette widget.
        """
        for category, grid_view in self._category_tabs.items():
            grid_view.clear()

        node_types = defaultdict(list)

        for nid, node_cls in self._factory.nodes.items():
            category = node_cls.__identifier__
            node_types[category].append((nid, node_cls.NODE_NAME))
            
        for category, nodes_list in node_types.items():
            grid_view = self._category_tabs.get(category)
            if not grid_view:
                grid_view = self._add_category_tab(category)

            for node_id, node_name in nodes_list:
                grid_view.add_item(node_name, node_id)

        self._update_tab_labels()
