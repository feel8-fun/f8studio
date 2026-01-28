from __future__ import annotations

from collections import OrderedDict
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from NodeGraphQt.constants import Z_VAL_BACKDROP, NodeEnum
from NodeGraphQt.constants import NodePropWidgetEnum
from NodeGraphQt.qgraphics.node_abstract import AbstractNodeItem
from NodeGraphQt.qgraphics.node_backdrop import BackdropSizer

from f8pysdk import F8ServiceSpec
from f8pysdk.schema_helpers import schema_default, schema_type
from .node_base import F8StudioBaseNode
from .service_process_toolbar import ServiceProcessToolbar


class F8StudioContainerBaseNode(F8StudioBaseNode):
    """
    Base class for all container nodes.

    Responsibilities:
    - track child node objects
    - keep container view and child views bound (dragging container moves children)
    """

    _child_nodes: list[F8StudioBaseNode]

    def __init__(self, qgraphics_item=None):
        _nodeitem_cls = qgraphics_item or F8StudioContainerNodeItem
        assert issubclass(
            _nodeitem_cls, F8StudioContainerNodeItem
        ), "F8StudioContainerBaseNode requires a F8StudioContainerNodeItem or subclass."
        super().__init__(qgraphics_item=_nodeitem_cls)
        assert isinstance(self.spec, F8ServiceSpec), "F8StudioContainerBaseNode requires F8ServiceSpec"

        self.model.color = (5, 129, 138, 50)
        self._child_nodes = []
        self._build_state_properties()

    def sync_from_spec(self) -> None:
        self._build_state_properties()

    def _build_state_properties(self) -> None:
        for s in self.effective_state_fields() or []:
            name = str(getattr(s, "name", "") or "").strip()
            if not name:
                continue
            try:
                if self.has_property(name):  # type: ignore[attr-defined]
                    continue
            except Exception:
                pass
            try:
                default_value = schema_default(s.valueSchema)
            except Exception:
                default_value = None
            widget_type, items, prop_range = self._state_widget_for_schema(getattr(s, "valueSchema", None))
            tooltip = str(getattr(s, "description", "") or "").strip() or None
            try:
                self.create_property(
                    name,
                    default_value,
                    items=items,
                    range=prop_range,
                    widget_type=widget_type,
                    widget_tooltip=tooltip,
                    tab="State",
                )
            except Exception:
                continue

    @staticmethod
    def _state_widget_for_schema(value_schema) -> tuple[int, list[str] | None, tuple[float, float] | None]:
        if value_schema is None:
            return NodePropWidgetEnum.QTEXT_EDIT.value, None, None
        try:
            t = schema_type(value_schema)
        except Exception:
            t = ""

        try:
            enum_items = list(getattr(getattr(value_schema, "root", None), "enum", None) or [])
        except Exception:
            enum_items = []
        if enum_items:
            return NodePropWidgetEnum.QCOMBO_BOX.value, [str(x) for x in enum_items], None

        if t == "boolean":
            return NodePropWidgetEnum.QCHECK_BOX.value, None, None
        if t in ("integer", "number", "string"):
            return NodePropWidgetEnum.QLINE_EDIT.value, None, None

        return NodePropWidgetEnum.QTEXT_EDIT.value, None, None

    def add_child(self, node: F8StudioBaseNode) -> None:
        if node in self._child_nodes:
            return
        self._child_nodes.append(node)
        self.view.add_child(node.view)

    def remove_child(self, node: F8StudioBaseNode) -> None:
        if node in self._child_nodes:
            self._child_nodes.remove(node)
        self.view.remove_child(node.view)

    def child_nodes(self) -> list[F8StudioBaseNode]:
        return list(self._child_nodes)


class F8StudioContainerNodeItem(AbstractNodeItem):
    """
    container item.

    This class is intentionally generic so all container types can reuse:
    - child tracking via `add_child` / `remove_child`
    - moving children when the container moves
    """

    def __init__(self, name: str = "container", text: str = "", parent=None):
        super().__init__(name, parent)
        self.setZValue(Z_VAL_BACKDROP)
        # Match `NodeGraphQt.qgraphics.node_base.NodeItem` API so
        # `BaseNode.update_model()` can iterate `self.view.widgets`.
        self._widgets = OrderedDict()
        # Must exist before BackdropSizer calls `itemChange()` which may call `_position_service_toolbar()`.
        self._svc_toolbar_proxy: QtWidgets.QGraphicsProxyWidget | None = None
        self._min_size = 500, 300
        self._sizer = BackdropSizer(self, 26.0)
        self._sizer.set_pos(*self._min_size)
        self._child_views: list[AbstractNodeItem] = []
        self._svc_toolbar_proxy: QtWidgets.QGraphicsProxyWidget | None = None

    def post_init(self, viewer=None, pos=None):
        self._ensure_service_toolbar(viewer)
        self._position_service_toolbar()
        if pos:
            try:
                self.xy_pos = pos
            except Exception:
                pass
            self._position_service_toolbar()

    def _ensure_service_toolbar(self, viewer: Any | None) -> None:
        if self._svc_toolbar_proxy is not None:
            return
        service_id = str(getattr(self, "id", "") or "").strip()
        if not service_id:
            return

        def _get_bridge() -> Any | None:
            try:
                g = getattr(viewer, "_f8_graph", None)
                return getattr(g, "service_bridge", None) if g is not None else None
            except Exception:
                return None

        def _get_service_class() -> str:
            try:
                g = getattr(viewer, "_f8_graph", None)
                if g is None:
                    return ""
                n = g.get_node_by_id(service_id)
                spec = getattr(n, "spec", None)
                return str(getattr(spec, "serviceClass", "") or "")
            except Exception:
                return ""

        try:
            w = ServiceProcessToolbar(
                service_id=service_id, get_bridge=_get_bridge, get_service_class=_get_service_class
            )
            proxy = QtWidgets.QGraphicsProxyWidget(self)
            proxy.setWidget(w)
            proxy.setZValue(10_000)
            proxy.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
            self._svc_toolbar_proxy = proxy
        except Exception:
            self._svc_toolbar_proxy = None

    def _position_service_toolbar(self) -> None:
        proxy = getattr(self, "_svc_toolbar_proxy", None)
        if proxy is None:
            return
        try:
            rect = self.boundingRect()
            w = float(proxy.size().width() or 0.0)
            h = float(proxy.size().height() or 0.0)
        except Exception:
            return
        
        try:
            proxy.setPos(rect.right() - w, rect.top() - h)
        except Exception:
            pass

    def _combined_rect(self, nodes: list[AbstractNodeItem]) -> QtCore.QRectF:
        group = self.scene().createItemGroup(nodes)
        rect = group.boundingRect()
        self.scene().destroyItemGroup(group)
        return rect

    def mouseDoubleClickEvent(self, event):
        viewer = self.viewer()
        if viewer:
            viewer.node_double_clicked.emit(self.id)
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """Moves child nodes along with the container."""
        prev_pos = self.pos()
        super().mouseMoveEvent(event)
        new_pos = self.pos()
        delta = new_pos - prev_pos
        for view in self._child_views:
            p = view.xy_pos
            p[0] += delta.x()
            p[1] += delta.y()
            view.xy_pos = p

    @AbstractNodeItem.xy_pos.setter
    def xy_pos(self, pos=None):
        new_x, new_y = pos or [0.0, 0.0]
        delta_x = new_x - self.pos().x()
        delta_y = new_y - self.pos().y()
        self.setPos(new_x, new_y)
        self._position_service_toolbar()
        for view in self._child_views:
            p = view.xy_pos
            p[0] += delta_x
            p[1] += delta_y
            view.xy_pos = p

    def on_sizer_pos_changed(self, pos):
        self._width = pos.x() + self._sizer.size
        self._height = pos.y() + self._sizer.size
        self._position_service_toolbar()
        self.update()

    def on_sizer_pos_mouse_release(self):
        size = {"pos": self.xy_pos, "width": self._width, "height": self._height}
        self.viewer().node_backdrop_updated.emit(self.id, "sizer_mouse_release", size)

    def on_sizer_double_clicked(self):
        size = self.calc_backdrop_size()
        self.viewer().node_backdrop_updated.emit(self.id, "sizer_double_clicked", size)

    def paint(self, painter, option, widget):
        painter.save()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.NoBrush)

        margin = 1.0
        rect = self.boundingRect()
        rect = QtCore.QRectF(
            rect.left() + margin,
            rect.top() + margin,
            rect.width() - (margin * 2),
            rect.height() - (margin * 2),
        )

        radius = 2.6
        color = (self.color[0], self.color[1], self.color[2], 50)
        painter.setBrush(QtGui.QColor(*color))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(rect, radius, radius)

        top_rect = QtCore.QRectF(rect.x(), rect.y(), rect.width(), 26.0)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(*self.color)))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(top_rect, radius, radius)
        for pos in [top_rect.left(), top_rect.right() - 5.0]:
            painter.drawRect(QtCore.QRectF(pos, top_rect.bottom() - 5.0, 5.0, 5.0))

        if self.selected:
            sel_color = [x for x in NodeEnum.SELECTED_COLOR.value]
            sel_color[-1] = 15
            painter.setBrush(QtGui.QColor(*sel_color))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(rect, radius, radius)

        txt_rect = QtCore.QRectF(top_rect.x(), top_rect.y(), rect.width(), top_rect.height())
        painter.setPen(QtGui.QColor(*self.text_color))
        painter.drawText(txt_rect, QtCore.Qt.AlignCenter, self.name)

        border = 0.8
        border_color = self.color
        if self.selected and NodeEnum.SELECTED_BORDER_COLOR.value:
            border = 1.0
            border_color = NodeEnum.SELECTED_BORDER_COLOR.value
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(*border_color), border))
        painter.drawRoundedRect(rect, radius, radius)

        painter.restore()

    def get_nodes(self, inc_intersects: bool = False) -> list[AbstractNodeItem]:
        mode = {True: QtCore.Qt.IntersectsItemShape, False: QtCore.Qt.ContainsItemShape}
        nodes: list[AbstractNodeItem] = []
        if self.scene():
            polygon = self.mapToScene(self.boundingRect())
            rect = polygon.boundingRect()
            items = self.scene().items(rect, mode=mode[inc_intersects])
            for item in items:
                if item == self or item == self._sizer:
                    continue
                if isinstance(item, AbstractNodeItem):
                    nodes.append(item)
        return nodes

    def calc_backdrop_size(self, nodes: list[AbstractNodeItem] | None = None) -> dict[str, Any]:
        nodes = nodes or self.get_nodes(True)
        if nodes:
            nodes_rect = self._combined_rect(nodes)
        else:
            center = self.mapToScene(self.boundingRect().center())
            nodes_rect = QtCore.QRectF(center.x(), center.y(), self._min_size[0], self._min_size[1])

        padding = 40
        return {
            "pos": [nodes_rect.x() - padding, nodes_rect.y() - padding],
            "width": nodes_rect.width() + (padding * 2),
            "height": nodes_rect.height() + (padding * 2),
        }

    def add_child(self, view: Any) -> None:
        if view not in self._child_views:
            self._child_views.append(view)
        view._container_item = self

    def remove_child(self, view: Any) -> None:
        if view in self._child_views:
            self._child_views.remove(view)
        if view._container_item is self:
            view._container_item = None

    @property
    def minimum_size(self):
        return self._min_size

    @minimum_size.setter
    def minimum_size(self, size=(50, 50)):
        self._min_size = size

    @property
    def widgets(self):
        return self._widgets.copy()

    @AbstractNodeItem.width.setter
    def width(self, width=0.0):
        AbstractNodeItem.width.fset(self, width)
        self._sizer.set_pos(self._width, self._height)

    @AbstractNodeItem.height.setter
    def height(self, height=0.0):
        AbstractNodeItem.height.fset(self, height)
        self._sizer.set_pos(self._width, self._height)
