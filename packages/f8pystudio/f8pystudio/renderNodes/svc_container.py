from __future__ import annotations

from platform import node
from typing import Any, TYPE_CHECKING

from qtpy import QtCore, QtGui, QtWidgets
from .internal.base import F8BaseRenderNode
from NodeGraphQt import NodeObject, BaseNode
from NodeGraphQt.constants import Z_VAL_BACKDROP, NodeEnum
from NodeGraphQt.qgraphics.node_abstract import AbstractNodeItem
from NodeGraphQt.qgraphics.node_backdrop import BackdropSizer

from f8pysdk import F8OperatorSpec, F8ServiceSpec

if TYPE_CHECKING:
    from .op_generic import OperatorNodeItem


class ContainerSvcRenderNode(F8BaseRenderNode):
    """
    Service container node (engine runner).

    This is a Backdrop-based container that can wrap operator nodes for deployment.
    """

    _child_nodes: list[F8BaseRenderNode]

    def __init__(self):
        assert isinstance(self.spec, F8ServiceSpec), "ContainerSvcRenderNode requires F8ServiceSpec"
        super().__init__(qgraphics_item=ContainerNodeItem)
        self.model.color = (5, 129, 138, 255)
        self._child_nodes: list[F8BaseRenderNode] = []

    def add_child(self, node: F8BaseRenderNode) -> None:
        if node not in self._child_nodes:
            self._child_nodes.append(node)
        self.view.add_child(node.view)

    def remove_child(self, node: F8BaseRenderNode) -> None:
        self.view.remove_child(node.view)
        if node in self._child_nodes:
            self._child_nodes.remove(node)

    def contained_nodes(self) -> list[F8BaseRenderNode]:
        """
        Returns operator nodes bound to this runner.

        We intentionally do not rely on NodeGraphQt's `wrap_nodes` parenthood,
        because runner geometry should be user-controlled.
        """
        if self.graph is None:
            return []
        out: list[F8BaseRenderNode] = []
        for nid in list(self._child_node_ids):
            n = self.graph.get_node_by_id(nid)
            if n is None:
                self._child_node_ids.discard(nid)
                continue
            if not hasattr(n, "spec") or not isinstance(n.spec, F8OperatorSpec):  # type: ignore[attr-defined]
                continue
            out.append(n)
        return out


class ContainerNodeItem(AbstractNodeItem):
    """
    Base Backdrop item.

    Args:
        name (str): name displayed on the node.
        text (str): backdrop text.
        parent (QtWidgets.QGraphicsItem): parent item.
    """

    def __init__(self, name="backdrop", text="", parent=None):
        super(ContainerNodeItem, self).__init__(name, parent)
        self.setZValue(Z_VAL_BACKDROP)
        self._properties["backdrop_text"] = text
        self._min_size = 500, 300
        self._sizer = BackdropSizer(self, 26.0)
        self._sizer.set_pos(*self._min_size)
        self._child_views: list[AbstractNodeItem] = []

    def _combined_rect(self, nodes):
        group = self.scene().createItemGroup(nodes)
        rect = group.boundingRect()
        self.scene().destroyItemGroup(group)
        return rect

    def mouseDoubleClickEvent(self, event):
        viewer = self.viewer()
        if viewer:
            viewer.node_double_clicked.emit(self.id)
        super(ContainerNodeItem, self).mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        super(ContainerNodeItem, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super(ContainerNodeItem, self).mouseReleaseEvent(event)

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
        """
        set the item scene postion.
        ("node.pos" conflicted with "QGraphicsItem.pos()"
        so it was refactored to "xy_pos".)

        Args:
            pos (list[float]): x, y scene position.
        """
        new_x, new_y = pos or [0.0, 0.0]
        delta_x = new_x - self.pos().x()
        delta_y = new_y - self.pos().y()
        self.setPos(new_x, new_y)
        # update child positions
        for view in self._child_views:
            p = view.xy_pos
            p[0] += delta_x
            p[1] += delta_y
            view.xy_pos = p

    def on_sizer_pos_changed(self, pos):
        self._width = pos.x() + self._sizer.size
        self._height = pos.y() + self._sizer.size
        self.update()

    def on_sizer_pos_mouse_release(self):
        size = {"pos": self.xy_pos, "width": self._width, "height": self._height}
        self.viewer().node_backdrop_updated.emit(self.id, "sizer_mouse_release", size)

    def on_sizer_double_clicked(self):
        size = self.calc_backdrop_size()
        self.viewer().node_backdrop_updated.emit(self.id, "sizer_double_clicked", size)

    def paint(self, painter, option, widget):
        """
        Draws the backdrop rect.

        Args:
            painter (QtGui.QPainter): painter used for drawing the item.
            option (QtGui.QStyleOptionGraphicsItem):
                used to describe the parameters needed to draw.
            widget (QtWidgets.QWidget): not used.
        """
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

        if self.backdrop_text:
            painter.setPen(QtGui.QColor(*self.text_color))
            txt_rect = QtCore.QRectF(top_rect.x() + 5.0, top_rect.height() + 3.0, rect.width() - 5.0, rect.height())
            painter.setPen(QtGui.QColor(*self.text_color))
            painter.drawText(txt_rect, QtCore.Qt.AlignLeft | QtCore.Qt.TextWordWrap, self.backdrop_text)

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

    def get_nodes(self, inc_intersects=False):
        mode = {True: QtCore.Qt.IntersectsItemShape, False: QtCore.Qt.ContainsItemShape}
        nodes = []
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

    def calc_backdrop_size(self, nodes=None):
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

    def add_child(self, view: OperatorNodeItem) -> None:
        if view not in self._child_views:
            self._child_views.append(view)
        view._container_item = self

    def remove_child(self, view: OperatorNodeItem) -> None:
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
    def backdrop_text(self):
        return self._properties["backdrop_text"]

    @backdrop_text.setter
    def backdrop_text(self, text):
        self._properties["backdrop_text"] = text
        self.update(self.boundingRect())

    @AbstractNodeItem.width.setter
    def width(self, width=0.0):
        AbstractNodeItem.width.fset(self, width)
        self._sizer.set_pos(self._width, self._height)

    @AbstractNodeItem.height.setter
    def height(self, height=0.0):
        AbstractNodeItem.height.fset(self, height)
        self._sizer.set_pos(self._width, self._height)

    def from_dict(self, node_dict):
        super().from_dict(node_dict)
        custom_props = node_dict.get("custom") or {}
        for prop_name, value in custom_props.items():
            if prop_name == "backdrop_text":
                self.backdrop_text = value
