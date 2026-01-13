from __future__ import annotations

from typing import Any

from Qt import QtCore
from NodeGraphQt.widgets.viewer import NodeViewer


class F8NodeViewer(NodeViewer):
    """
    Studio viewer with basic keyboard shortcuts.

    - `Delete` / `Backspace`: delete selected nodes
    - `Tab`: open node search
    """

    def __init__(self, parent=None, undo_stack=None):
        super().__init__(parent=parent, undo_stack=undo_stack)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._f8_graph: Any | None = None

    def set_graph(self, graph: Any) -> None:
        self._f8_graph = graph

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()

    def mousePressEvent(self, event):
        self.setFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        graph = self._f8_graph

        if graph is not None and event.modifiers() == QtCore.Qt.NoModifier:
            if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                nodes = graph.selected_nodes()
                if nodes:
                    graph.delete_nodes(nodes)
                    event.accept()
                    return

            if event.key() == QtCore.Qt.Key_Tab:
                graph.toggle_node_search()
                event.accept()
                return

        super().keyPressEvent(event)

