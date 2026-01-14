from __future__ import annotations

from typing import Any

from Qt import QtCore, QtGui, QtWidgets
from NodeGraphQt.widgets.viewer import NodeViewer


class F8StudioNodeViewer(NodeViewer):
    """
    Studio viewer with basic keyboard shortcuts.

    - `Delete` / `Backspace`: delete selected nodes
    - `Tab`: open node search
    """

    def __init__(self, parent=None, undo_stack=None):
        super().__init__(parent=parent, undo_stack=undo_stack)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._f8_graph: Any | None = None

        self._shortcut_search = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Tab), self)
        self._shortcut_search.setContext(QtCore.Qt.WidgetShortcut)
        self._shortcut_search.activated.connect(self._open_node_search)  # type: ignore[attr-defined]

        self._shortcut_delete = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self)
        self._shortcut_delete.setContext(QtCore.Qt.WidgetShortcut)
        self._shortcut_delete.activated.connect(self._delete_selected_nodes)  # type: ignore[attr-defined]

        self._shortcut_backspace = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self)
        self._shortcut_backspace.setContext(QtCore.Qt.WidgetShortcut)
        self._shortcut_backspace.activated.connect(self._delete_selected_nodes)  # type: ignore[attr-defined]

    def set_graph(self, graph: Any) -> None:
        self._f8_graph = graph

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()

    def mousePressEvent(self, event):
        self.setFocus()
        super().mousePressEvent(event)

    def _delete_selected_nodes(self) -> None:
        graph = self._f8_graph
        if graph is None:
            return
        nodes = graph.selected_nodes()
        if nodes:
            graph.delete_nodes(nodes)

    def _open_node_search(self) -> None:
        graph = self._f8_graph
        if graph is None:
            return
        graph.toggle_node_search()
