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

    _PAN_STEP_PX: int = 50
    _PAN_STEP_PX_FAST: int = 150
    _ZOOM_TICKS: int = 1
    _ZOOM_TICKS_FAST: int = 3

    def __init__(self, parent=None, undo_stack=None):
        super().__init__(parent=parent, undo_stack=undo_stack)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._f8_graph: Any | None = None
        # NOTE: NodeGraphQt's NodeViewer already uses internal attributes like
        # `MMB_state`, `_origin_pos`, `_previous_pos` for selection, tab-search,
        # and other interactions. Do not overwrite them here.
        self._f8_mmb_panning: bool = False
        self._f8_mmb_prev_pos: QtCore.QPoint | None = None

        self._shortcut_search = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Tab), self)
        self._shortcut_search.setContext(QtCore.Qt.WidgetShortcut)
        self._shortcut_search.activated.connect(self._open_node_search)  # type: ignore[attr-defined]

        self._shortcut_delete = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self)
        self._shortcut_delete.setContext(QtCore.Qt.WidgetShortcut)
        self._shortcut_delete.activated.connect(self._delete_selected_nodes)  # type: ignore[attr-defined]

        self._shortcut_backspace = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self)
        self._shortcut_backspace.setContext(QtCore.Qt.WidgetShortcut)
        self._shortcut_backspace.activated.connect(self._delete_selected_nodes)  # type: ignore[attr-defined]

    @property
    def f8_graph(self) -> Any | None:
        return self._f8_graph

    def set_graph(self, graph: Any) -> None:
        self._f8_graph = graph

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()

    def focusOutEvent(self, event):  # type: ignore[override]
        self._cancel_f8_mmb_pan()
        return super().focusOutEvent(event)

    def leaveEvent(self, event):  # type: ignore[override]
        # If mouse is released outside the viewport, ensure we don't get stuck.
        self._cancel_f8_mmb_pan()
        return super().leaveEvent(event)

    def _cancel_f8_mmb_pan(self) -> None:
        if not self._f8_mmb_panning:
            return
        self._f8_mmb_panning = False
        self._f8_mmb_prev_pos = None
        try:
            self.unsetCursor()
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _pan_by_pixels(self, dx_px: int, dy_px: int) -> None:
        center_view = QtCore.QPoint(int(self.viewport().width() / 2), int(self.viewport().height() / 2))
        scene_center = self.mapToScene(center_view)
        scene_offset = self.mapToScene(center_view + QtCore.QPoint(dx_px, dy_px))
        delta = scene_offset - scene_center
        self._set_viewer_pan(delta.x(), delta.y())

    def _zoom_by_ticks(self, ticks: int) -> None:
        center_view = QtCore.QPoint(int(self.viewport().width() / 2), int(self.viewport().height() / 2))
        for _ in range(abs(ticks)):
            self._set_viewer_zoom(1.0 if ticks > 0 else -1.0, 0.0, center_view)

    @staticmethod
    def _is_text_input_focus(widget: QtWidgets.QWidget | None) -> bool:
        if widget is None:
            return False
        return isinstance(
            widget,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QAbstractSpinBox,
            ),
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        event.setAccepted(False)
        super().keyPressEvent(event)

        # Respect keys already consumed by child/proxy widgets.
        if event.isAccepted():
            return

        focus_widget = QtWidgets.QApplication.focusWidget()
        if self._is_text_input_focus(focus_widget):
            return

        # Avoid interfering when Tab search is active (it should generally hold focus,
        # but be defensive in case focus is still on the viewer).
        if self._search_widget.isVisible():
            return

        key = event.key()
        mods = event.modifiers()
        if bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.AltModifier | QtCore.Qt.MetaModifier)):
            return
        fast = bool(mods & QtCore.Qt.ShiftModifier)
        pan_step = self._PAN_STEP_PX_FAST if fast else self._PAN_STEP_PX
        zoom_ticks = self._ZOOM_TICKS_FAST if fast else self._ZOOM_TICKS

        # Pan: arrow keys or WASD (grab canvas style).
        if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_A):
            self._pan_by_pixels(-pan_step, 0)
            event.accept()
            return
        if key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_D):
            self._pan_by_pixels(pan_step, 0)
            event.accept()
            return
        if key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_W):
            self._pan_by_pixels(0, -pan_step)
            event.accept()
            return
        if key in (QtCore.Qt.Key_Down, QtCore.Qt.Key_S):
            self._pan_by_pixels(0, pan_step)
            event.accept()
            return

        # Zoom: Q/E or PageUp/PageDown.
        if key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_PageUp):
            self._zoom_by_ticks(zoom_ticks)
            event.accept()
            return
        if key in (QtCore.Qt.Key_E, QtCore.Qt.Key_PageDown):
            self._zoom_by_ticks(-zoom_ticks)
            event.accept()
            return

    def mousePressEvent(self, event):
        self.setFocus()
        # Always reserve MMB for canvas pan, regardless of what's under cursor.
        # NodeGraphQt disables MMB pan when clicking on nodes; we want consistent
        # navigation behavior.
        if event.button() == QtCore.Qt.MiddleButton:
            self._f8_mmb_panning = True
            self._f8_mmb_prev_pos = event.pos()
            if self._search_widget.isVisible():
                self.tab_search_toggle()
            try:
                self.setCursor(QtCore.Qt.ClosedHandCursor)
            except (AttributeError, RuntimeError, TypeError):
                pass
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Force MMB to pan only (no ALT+MMB zoom).
        if self._f8_mmb_panning and self._f8_mmb_prev_pos is not None:
            previous_pos = self.mapToScene(self._f8_mmb_prev_pos)
            current_pos = self.mapToScene(event.pos())
            delta = previous_pos - current_pos
            self._set_viewer_pan(delta.x(), delta.y())
            self._f8_mmb_prev_pos = event.pos()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self._cancel_f8_mmb_pan()
            event.accept()
            return
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

    @staticmethod
    def acyclic_check(start_port, end_port) -> bool:
        # Disable Acyclic checking
        return True
