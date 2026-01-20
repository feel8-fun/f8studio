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
        # Always reserve MMB for canvas pan, regardless of what's under cursor.
        # NodeGraphQt disables MMB pan when clicking on nodes; we want consistent
        # navigation behavior.
        try:
            if event.button() == QtCore.Qt.MiddleButton:
                self.MMB_state = True
                self._origin_pos = event.pos()
                self._previous_pos = event.pos()
                if self._search_widget.isVisible():
                    self.tab_search_toggle()
                event.accept()
                return
        except Exception:
            pass
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Force MMB to pan only (no ALT+MMB zoom).
        try:
            if getattr(self, "MMB_state", False):
                previous_pos = self.mapToScene(self._previous_pos)
                current_pos = self.mapToScene(event.pos())
                delta = previous_pos - current_pos
                self._set_viewer_pan(delta.x(), delta.y())
                self._previous_pos = event.pos()
                QtWidgets.QGraphicsView.mouseMoveEvent(self, event)
                return
        except Exception:
            pass
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == QtCore.Qt.MiddleButton:
                self.MMB_state = False
                event.accept()
                return
        except Exception:
            pass
        super().mouseReleaseEvent(event)

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
