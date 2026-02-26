from __future__ import annotations

from qtpy import QtWidgets

from f8pysdk import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
)

from .missing_badge import MissingBadgeMixin
from .operator_basenode import F8StudioOperatorBaseNode, F8StudioOperatorNodeItem


class F8StudioMissingOperatorNodeItem(MissingBadgeMixin, F8StudioOperatorNodeItem):
    def __init__(self, name: str = "node", parent: QtWidgets.QGraphicsItem | None = None):
        super().__init__(name=name, parent=parent)
        self._init_missing_badge()

    def draw_node(self) -> None:
        super().draw_node()
        self._refresh_missing_badge()


class F8StudioOperatorMissingNode(F8StudioOperatorBaseNode):
    """
    Placeholder operator node used when a session references an unknown type.
    """

    SPEC_TEMPLATE = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass="f8.missing",
        operatorClass="f8.missing.operator",
        version="0.0.1",
        label="Missing Operator",
        tags=["__hidden__"],
    )

    def __init__(self, qgraphics_item: type[QtWidgets.QGraphicsItem] | None = None):
        super().__init__(qgraphics_item=qgraphics_item or F8StudioMissingOperatorNodeItem)


# Compatibility alias.
F8StudioMissingOperatorBaseNode = F8StudioOperatorMissingNode
