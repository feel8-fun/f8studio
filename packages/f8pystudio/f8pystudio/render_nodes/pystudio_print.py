from __future__ import annotations

import json
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode, F8StudioOperatorNodeItem


class _PrintNodeItem(F8StudioOperatorNodeItem):
    def __init__(self, name: str = "node", parent: QtWidgets.QGraphicsItem | None = None):
        super().__init__(name=name, parent=parent)
        self._preview_item = QtWidgets.QGraphicsTextItem("", self)
        self._preview_item.setDefaultTextColor(QtGui.QColor(180, 180, 180))
        self._preview_item.setTextWidth(220)


    def post_init(self, viewer, pos=None):
        super().post_init(viewer, pos=pos)
        # Position preview text under the title area.
        try:
            self._preview_item.setPos(QtCore.QPointF(8.0, 40.0))
        except Exception:
            pass

    def set_preview_text(self, text: str) -> None:
        try:
            self._preview_item.setPlainText(str(text or ""))
        except Exception:
            pass


class PyStudioPrintNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.print_node_operator`.

    Adds a preview text area that can be updated by the editor refresh loop.
    """

    def __init__(self, qgraphics_item=None):
        super().__init__(qgraphics_item=qgraphics_item or _PrintNodeItem)

    def set_preview(self, value: Any) -> None:
        try:
            txt = json.dumps(value, ensure_ascii=False, indent=1, default=str)
        except Exception:
            txt = str(value)
        try:
            item = self.view
            if hasattr(item, "set_preview_text"):
                item.set_preview_text(txt)
        except Exception:
            return

