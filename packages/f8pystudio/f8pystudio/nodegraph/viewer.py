from __future__ import annotations

import logging
from typing import Any

from Qt import QtCore, QtGui, QtWidgets
from NodeGraphQt.constants import PortTypeEnum
from NodeGraphQt.widgets.viewer import NodeViewer

from .edge_rules import (
    EDGE_KIND_DATA,
    EDGE_KIND_EXEC,
    EDGE_KIND_STATE,
    connection_kind,
    normalize_edge_kind,
    port_view_name,
    validate_runtime_connection,
)
from .pipe_item import F8StudioPipeItem

logger = logging.getLogger(__name__)


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
    _PREVIEW_PADDING_X: float = 10.0
    _PREVIEW_PADDING_Y: float = 6.0
    _PREVIEW_OFFSET_X: float = 14.0
    _PREVIEW_OFFSET_Y: float = 14.0
    node_placement_changed = QtCore.Signal(bool, str)

    def __init__(self, parent=None, undo_stack=None):
        super().__init__(parent=parent, undo_stack=undo_stack)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._f8_graph: Any | None = None
        # NOTE: NodeGraphQt's NodeViewer already uses internal attributes like
        # `MMB_state`, `_origin_pos`, `_previous_pos` for selection, tab-search,
        # and other interactions. Do not overwrite them here.
        self._f8_mmb_panning: bool = False
        self._f8_mmb_prev_pos: QtCore.QPoint | None = None
        self._pending_node_type: str | None = None
        self._pending_node_label: str = ""
        self._pending_graph_insert_request: Any | None = None
        self._pending_graph_label: str = ""
        self._pending_graph_size: tuple[float, float] | None = None
        self._placement_preview_rect: QtWidgets.QGraphicsRectItem | None = None
        self._placement_preview_label: QtWidgets.QGraphicsSimpleTextItem | None = None
        self._edge_kind_visibility: dict[str, bool] = {
            EDGE_KIND_EXEC: True,
            EDGE_KIND_DATA: True,
            EDGE_KIND_STATE: True,
        }

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

    def set_edge_kind_visible(self, kind: str, visible: bool) -> None:
        normalized = normalize_edge_kind(kind)
        if normalized is None:
            raise ValueError(f"unknown edge kind: {kind}")
        self._edge_kind_visibility[normalized] = bool(visible)
        self.refresh_edge_visibility()

    def edge_kind_visible(self, kind: str) -> bool:
        normalized = normalize_edge_kind(kind)
        if normalized is None:
            raise ValueError(f"unknown edge kind: {kind}")
        return bool(self._edge_kind_visibility.get(normalized, True))

    def refresh_edge_visibility(self) -> None:
        for pipe in list(self.all_pipes() or []):
            if isinstance(pipe, F8StudioPipeItem):
                try:
                    pipe.draw_path(pipe.input_port, pipe.output_port)
                except (AttributeError, RuntimeError, TypeError):
                    continue
                continue
            try:
                out_name = port_view_name(pipe.output_port)
                in_name = port_view_name(pipe.input_port)
                kind = connection_kind(out_name, in_name)
                if kind is None:
                    pipe.draw_path(pipe.input_port, pipe.output_port)
                    continue
                pipe_visible = bool(self.edge_kind_visible(kind))
                if pipe_visible:
                    pipe_visible = all(
                        (
                            pipe.input_port.isVisible(),
                            pipe.output_port.isVisible(),
                            pipe.input_port.node.isVisible(),
                            pipe.output_port.node.isVisible(),
                        )
                    )
                pipe.setVisible(bool(pipe_visible))
                pipe.draw_path(pipe.input_port, pipe.output_port)
            except (AttributeError, RuntimeError, TypeError):
                continue

    def begin_node_placement(self, node_type: str, node_label: str) -> None:
        pending_type = str(node_type or "").strip()
        if not pending_type:
            self.cancel_node_placement()
            return
        self.cancel_graph_placement()
        self._pending_node_type = pending_type
        self._pending_node_label = str(node_label or "").strip()
        self._ensure_placement_preview_items()
        self._update_placement_preview_at(self.mapFromGlobal(QtGui.QCursor.pos()))
        self._update_cursor_state()
        self.node_placement_changed.emit(True, self._pending_node_label)

    def cancel_node_placement(self) -> None:
        if self._pending_node_type is None:
            return
        self._pending_node_type = None
        self._pending_node_label = ""
        if self._pending_graph_insert_request is None:
            self._set_placement_preview_visible(False)
        self._update_cursor_state()
        self.node_placement_changed.emit(bool(self._pending_graph_insert_request is not None), self._pending_graph_label)

    def is_node_placement_active(self) -> bool:
        return self._pending_node_type is not None

    def begin_graph_placement(self, request: Any, label: str) -> None:
        if request is None:
            self.cancel_graph_placement()
            return
        self.cancel_node_placement()
        self._pending_graph_insert_request = request
        self._pending_graph_label = str(label or "").strip()
        bbox = request.source_bbox
        width = max(1.0, float(bbox.width))
        height = max(1.0, float(bbox.height))
        self._pending_graph_size = (width, height)
        self._ensure_placement_preview_items()
        self._update_placement_preview_at(self.mapFromGlobal(QtGui.QCursor.pos()))
        self._update_cursor_state()
        shown_label = self._pending_graph_label or "Insert Graph"
        self.node_placement_changed.emit(True, shown_label)

    def cancel_graph_placement(self) -> None:
        if self._pending_graph_insert_request is None:
            return
        self._pending_graph_insert_request = None
        self._pending_graph_label = ""
        self._pending_graph_size = None
        if self._pending_node_type is None:
            self._set_placement_preview_visible(False)
        self._update_cursor_state()
        self.node_placement_changed.emit(bool(self._pending_node_type is not None), self._pending_node_label)

    def is_graph_placement_active(self) -> bool:
        return self._pending_graph_insert_request is not None

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()

    def enterEvent(self, event):  # type: ignore[override]
        if self.is_node_placement_active() or self.is_graph_placement_active():
            self._update_placement_preview_at(self.mapFromGlobal(QtGui.QCursor.pos()))
        return super().enterEvent(event)

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
        self._update_cursor_state()

    def _update_cursor_state(self) -> None:
        try:
            if self._f8_mmb_panning:
                self.setCursor(QtCore.Qt.ClosedHandCursor)
            elif self._pending_node_type is not None or self._pending_graph_insert_request is not None:
                self.setCursor(QtCore.Qt.CrossCursor)
            else:
                self.unsetCursor()
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _ensure_placement_preview_items(self) -> None:
        if self._placement_preview_rect is not None and self._placement_preview_label is not None:
            return
        scene = self.scene()
        if scene is None:
            return
        rect_item = QtWidgets.QGraphicsRectItem()
        rect_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        rect_item.setZValue(10_000.0)
        rect_item.setPen(QtGui.QPen(QtGui.QColor(132, 190, 255, 220), 1.0))
        rect_item.setBrush(QtGui.QBrush(QtGui.QColor(32, 43, 56, 155)))
        rect_item.setVisible(False)
        scene.addItem(rect_item)

        label_item = QtWidgets.QGraphicsSimpleTextItem()
        label_item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        label_item.setBrush(QtGui.QBrush(QtGui.QColor(240, 246, 252, 230)))
        font = label_item.font()
        font.setPointSize(10)
        label_item.setFont(font)
        label_item.setZValue(10_001.0)
        label_item.setVisible(False)
        scene.addItem(label_item)

        self._placement_preview_rect = rect_item
        self._placement_preview_label = label_item

    def _set_placement_preview_visible(self, visible: bool) -> None:
        rect_item = self._placement_preview_rect
        label_item = self._placement_preview_label
        if rect_item is None or label_item is None:
            return
        rect_item.setVisible(visible)
        label_item.setVisible(visible)

    def _update_placement_preview_at(self, view_pos: QtCore.QPoint) -> None:
        if not self.is_node_placement_active() and not self.is_graph_placement_active():
            return
        self._ensure_placement_preview_items()
        rect_item = self._placement_preview_rect
        label_item = self._placement_preview_label
        if rect_item is None or label_item is None:
            return

        label_text = self._pending_node_label or (self._pending_node_type or "")
        top_left = self.mapToScene(view_pos)
        x = float(top_left.x()) + self._PREVIEW_OFFSET_X
        y = float(top_left.y()) + self._PREVIEW_OFFSET_Y
        if self.is_graph_placement_active() and self._pending_graph_size is not None:
            graph_width, graph_height = self._pending_graph_size
            label_text = self._pending_graph_label or "Insert Graph"
            label_item.setText(label_text)
            rect_item.setRect(float(top_left.x()), float(top_left.y()), float(graph_width), float(graph_height))
            label_x = float(top_left.x()) + self._PREVIEW_PADDING_X
            label_y = float(top_left.y()) + self._PREVIEW_PADDING_Y
            label_item.setPos(label_x, label_y)
            self._set_placement_preview_visible(True)
            return

        label_item.setText(label_text)
        text_rect = label_item.boundingRect()
        width = float(text_rect.width()) + (self._PREVIEW_PADDING_X * 2.0)
        height = float(text_rect.height()) + (self._PREVIEW_PADDING_Y * 2.0)
        rect_item.setRect(x, y, width, height)
        label_item.setPos(x + self._PREVIEW_PADDING_X, y + self._PREVIEW_PADDING_Y)
        self._set_placement_preview_visible(True)

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
        if event.key() == QtCore.Qt.Key_Escape:
            if self.is_graph_placement_active():
                self.cancel_graph_placement()
                event.accept()
                return
            if self.is_node_placement_active():
                self.cancel_node_placement()
                event.accept()
                return

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
        if self.is_graph_placement_active():
            if event.button() == QtCore.Qt.RightButton:
                self.cancel_graph_placement()
                event.accept()
                return
            if event.button() == QtCore.Qt.LeftButton:
                graph = self._f8_graph
                request = self._pending_graph_insert_request
                if graph is None or request is None:
                    event.accept()
                    return
                scene_pos = self.mapToScene(event.pos())
                try:
                    graph.apply_insert_graph(
                        request,
                        anchor_x=float(scene_pos.x()),
                        anchor_y=float(scene_pos.y()),
                    )
                    self.cancel_graph_placement()
                except Exception:
                    logger.exception("failed to insert graph from placement request")
                event.accept()
                return

        if self.is_node_placement_active():
            if event.button() == QtCore.Qt.RightButton:
                self.cancel_node_placement()
                event.accept()
                return
            if event.button() == QtCore.Qt.LeftButton:
                pending_type = self._pending_node_type
                graph = self._f8_graph
                if pending_type is None or graph is None:
                    event.accept()
                    return
                scene_pos = self.mapToScene(event.pos())
                try:
                    created = graph.create_node(
                        pending_type,
                        pos=(float(scene_pos.x()), float(scene_pos.y())),
                    )
                except Exception:
                    logger.exception('failed to create pending node "%s"', pending_type)
                    event.accept()
                    return
                if created is not None:
                    self.cancel_node_placement()
                event.accept()
                return

        # Always reserve MMB for canvas pan, regardless of what's under cursor.
        # NodeGraphQt disables MMB pan when clicking on nodes; we want consistent
        # navigation behavior.
        if event.button() == QtCore.Qt.MiddleButton:
            self._f8_mmb_panning = True
            self._f8_mmb_prev_pos = event.pos()
            if self._search_widget.isVisible():
                self.tab_search_toggle()
            self._update_cursor_state()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_node_placement_active() or self.is_graph_placement_active():
            self._update_placement_preview_at(event.pos())
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

    def _validate_accept_connection(self, from_port, to_port):  # type: ignore[override]
        if not super()._validate_accept_connection(from_port, to_port):
            return False

        out_port = None
        in_port = None
        if from_port.port_type == PortTypeEnum.OUT.value and to_port.port_type == PortTypeEnum.IN.value:
            out_port = from_port
            in_port = to_port
        elif from_port.port_type == PortTypeEnum.IN.value and to_port.port_type == PortTypeEnum.OUT.value:
            out_port = to_port
            in_port = from_port
        if out_port is None or in_port is None:
            return False

        graph = self._f8_graph
        if graph is None:
            return False

        out_node_id = str(out_port.node.id or "").strip()
        in_node_id = str(in_port.node.id or "").strip()
        if not out_node_id or not in_node_id:
            return False

        try:
            out_node = graph.get_node_by_id(out_node_id)
            in_node = graph.get_node_by_id(in_node_id)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            return False
        if out_node is None or in_node is None:
            return False

        allowed, _reason = validate_runtime_connection(
            out_port_name=port_view_name(out_port),
            in_port_name=port_view_name(in_port),
            out_node=out_node,
            in_node=in_node,
        )
        return bool(allowed)

    def establish_connection(self, start_port, end_port):  # type: ignore[override]
        pipe = F8StudioPipeItem()
        scene = self.scene()
        if scene is None:
            return
        scene.addItem(pipe)
        pipe.set_connections(start_port, end_port)
        pipe.draw_path(pipe.input_port, pipe.output_port)
        if start_port.node.selected or end_port.node.selected:
            pipe.highlight()
        if not start_port.node.visible or not end_port.node.visible:
            pipe.hide()

    @staticmethod
    def acyclic_check(start_port, end_port) -> bool:
        # Disable Acyclic checking
        return True
