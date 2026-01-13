from __future__ import annotations

from typing import Any

from NodeGraphQt import BackdropNode, BaseNode
from NodeGraphQt.qgraphics.node_backdrop import BackdropNodeItem

from f8pysdk import F8OperatorSpec, F8ServiceSpec


class OperatorRunnerBackdropItem(BackdropNodeItem):
    """
    Backdrop item used by OperatorRunner.

    NodeGraphQt's default BackdropNodeItem has grouping selection behavior
    (selecting/moving nodes contained by the rect) and double-click auto resize.
    For OperatorRunner we treat the container as a "soft" boundary and keep all
    containment logic in our graph layer, so we disable those behaviors to avoid
    confusing interactions (eg. moving a child accidentally moving the runner).
    """

    def __init__(self, name='backdrop', text='', parent=None):
        super().__init__(name, text, parent)

        self.minimum_size=(500, 300)

    def mousePressEvent(self, event):
        # Bypass BackdropNodeItem grouping-selection behavior.
        super(BackdropNodeItem, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # Bypass BackdropNodeItem re-select behavior.
        super(BackdropNodeItem, self).mouseReleaseEvent(event)

    def get_nodes(self, inc_intersects=False):
        # NodeGraphQt.viewer selection logic uses `BackdropNodeItem.get_nodes()`
        # to select/move wrapped nodes together. OperatorRunner implements its
        # own containment rules, so we disable this behavior.
        return []

    def on_sizer_double_clicked(self):
        # Disable "auto resize" on double click.
        return


class OperatorRunnerRenderNode(BackdropNode):
    """
    Service container node (engine runner).

    This is a Backdrop-based container that can wrap operator nodes for deployment.
    """

    spec: F8ServiceSpec

    def __init__(self, qgraphics_item=None, qgraphics_views=None):
        super().__init__(qgraphics_views=qgraphics_views or OperatorRunnerBackdropItem)
        
        self._child_node_ids: set[str] = set()
        try:
            self.set_text("engine")  # type: ignore[attr-defined]
        except Exception:
            pass

    def add_child(self, node: BaseNode) -> None:
        self._child_node_ids.add(node.id)

    def remove_child(self, node_id: str) -> None:
        self._child_node_ids.discard(node_id)

    def contained_nodes(self) -> list[BaseNode]:
        """
        Returns operator nodes bound to this runner.

        We intentionally do not rely on NodeGraphQt's `wrap_nodes` parenthood,
        because runner geometry should be user-controlled.
        """
        if self.graph is None:
            return []
        out: list[BaseNode] = []
        for nid in list(self._child_node_ids):
            try:
                n = self.graph.get_node_by_id(nid)
            except Exception:
                n = None
            if n is None:
                self._child_node_ids.discard(nid)
                continue
            if not isinstance(getattr(n, "spec", None), F8OperatorSpec):
                continue
            out.append(n)
        return out
