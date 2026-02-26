from __future__ import annotations

from qtpy import QtWidgets

from f8pysdk import (
    F8ServiceSchemaVersion,
    F8ServiceSpec,
)

from .missing_badge import MissingBadgeMixin
from .service_basenode import F8StudioServiceBaseNode, F8StudioServiceNodeItem

class F8StudioMissingServiceNodeItem(MissingBadgeMixin, F8StudioServiceNodeItem):
    def __init__(self, name: str = "node", parent: QtWidgets.QGraphicsItem | None = None):
        super().__init__(name=name, parent=parent)
        self._init_missing_badge()

    def draw_node(self) -> None:
        super().draw_node()
        self._refresh_missing_badge()


class F8StudioServiceMissingNode(F8StudioServiceBaseNode):
    """
    Placeholder service node used when a session references an unknown type.
    """

    SPEC_TEMPLATE = F8ServiceSpec(
        schemaVersion=F8ServiceSchemaVersion.f8service_1,
        serviceClass="f8.missing",
        version="0.0.1",
        label="Missing Service",
        tags=["__hidden__"],
    )

    def __init__(self, qgraphics_item: type[QtWidgets.QGraphicsItem] | None = None):
        super().__init__(qgraphics_item=qgraphics_item or F8StudioMissingServiceNodeItem)


# Compatibility alias.
F8StudioMissingServiceBaseNode = F8StudioServiceMissingNode
