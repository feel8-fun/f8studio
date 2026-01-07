from __future__ import annotations

import uuid
from collections.abc import Iterable

from NodeGraphQt.nodes.backdrop_node import BackdropNode
from NodeGraphQt.qgraphics.node_backdrop import BackdropNodeItem

from Qt import QtCore, QtWidgets  # type: ignore[import-not-found]

from f8pysdk import F8ServiceSpec, schema_default

from ..services.service_registry import ServiceSpecRegistry
from ..services.builtin import ENGINE_SERVICE_CLASS
from .generic import GenericNode


class EngineBackdropNodeItem(BackdropNodeItem):
    """
    Customized backdrop item for EngineServiceNode.

    Goals:
    1) Selecting the backdrop should not also select enclosed nodes.
    2) Dragging the backdrop moves enclosed nodes (without selecting them).
    """

    def __init__(self, name: str = "engine", text: str = "", parent: QtWidgets.QGraphicsItem | None = None) -> None:
        super().__init__(name=name, text=text, parent=parent)
        self._drag_active = False
        self._drag_nodes: list[QtWidgets.QGraphicsItem] = []

        # Required to receive position change notifications reliably.
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsScenePositionChanges, True)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[name-defined]
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.scenePos()
            rect = QtCore.QRectF(pos.x() - 5, pos.y() - 5, 10, 10)
            try:
                item = self.scene().items(rect)[0]  # type: ignore[union-attr]
            except Exception:
                item = None

            # Keep default behavior: if clicking on a pipe or port, don't drag the backdrop.
            from NodeGraphQt.qgraphics.pipe import PipeItem
            from NodeGraphQt.qgraphics.port import PortItem

            if isinstance(item, (PortItem, PipeItem)):
                self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
                return

            viewer = self.viewer()
            if viewer:
                try:
                    [n.setSelected(False) for n in viewer.selected_nodes()]
                except Exception:
                    pass

            self.setSelected(True)
            self._drag_active = True
            try:
                # Include intersecting nodes so "edge-touching" nodes move too.
                self._drag_nodes = [n for n in list(self.get_nodes(True)) if not isinstance(n, BackdropNodeItem)]
            except Exception:
                self._drag_nodes = []

            # Let the base class update selection state.
            QtWidgets.QGraphicsItem.mousePressEvent(self, event)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[name-defined]
        if self._drag_active and self._drag_nodes:
            try:
                delta = event.scenePos() - event.lastScenePos()
            except Exception:
                delta = QtCore.QPointF(0.0, 0.0)
            if not delta.isNull():
                for node_item in list(self._drag_nodes):
                    try:
                        node_item.setPos(node_item.pos() + delta)
                    except Exception:
                        continue
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # type: ignore[name-defined]
        super().mouseReleaseEvent(event)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self._drag_active = False
        self._drag_nodes = []

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value: object) -> object:  # type: ignore[name-defined]
        # Keep the base behavior; we move children in `mouseMoveEvent` so this can be a no-op.
        return super().itemChange(change, value)


class EngineServiceNode(BackdropNode):  # type: ignore[misc]
    """
    Engine service node (Backdrop-style grouping).

    This node is intentionally NOT a SubGraph/GroupNode: it stays in the same
    graph view and visually "boxes" a set of operator nodes to indicate they
    run under the same engine service instance (serviceId == this node id).
    """

    __identifier__ = "feel8.service"
    NODE_NAME = "Engine"

    SPEC_KEY: str = ENGINE_SERVICE_CLASS

    service_spec: F8ServiceSpec
    spec: F8ServiceSpec

    def __init__(self) -> None:
        super().__init__(qgraphics_views=EngineBackdropNodeItem)
        stable_id = uuid.uuid4().hex
        try:
            self.model.id = stable_id  # type: ignore[attr-defined]
            self.view.id = stable_id  # type: ignore[attr-defined]
        except Exception:
            pass
        self.service_spec = ServiceSpecRegistry.instance().get(self.SPEC_KEY)
        self.spec = self.service_spec
        try:
            self.set_name(self.service_spec.label)  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            self.model.color = (70, 90, 130, 255)  # type: ignore[attr-defined]
        except Exception:
            pass

        self._apply_state_properties()

    def _apply_state_properties(self) -> None:
        for field_def in self.service_spec.states or []:
            if self.has_property(field_def.name):  # type: ignore[attr-defined]
                continue
            try:
                default_value = schema_default(field_def.valueSchema)
            except Exception:
                default_value = None
            try:
                self.create_property(field_def.name, default_value)
            except Exception:
                pass

    def ensure_state_properties(self) -> None:
        self._apply_state_properties()

    def operator_nodes(self) -> list[GenericNode]:
        try:
            nodes = self.nodes()  # BackdropNode.nodes()
        except Exception:
            nodes = []
        return [n for n in nodes if isinstance(n, GenericNode)]

    def wrap_operator_nodes(self, nodes: Iterable[GenericNode]) -> None:
        try:
            self.wrap_nodes(list(nodes))  # type: ignore[arg-type]
        except Exception:
            pass
