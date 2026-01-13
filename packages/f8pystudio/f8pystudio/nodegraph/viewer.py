from __future__ import annotations

from NodeGraphQt.widgets.viewer import NodeViewer
from Qt import QtCore


class F8NodeViewer(NodeViewer):
    """
    NodeGraphQt viewer with a "moving_nodes" signal.

    NodeGraphQt only emits "moved_nodes" on mouse release. For container
    constraints we also need continuous updates while dragging.
    """

    moving_nodes = QtCore.Signal(object)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

        # Only emit while dragging nodes (not panning/zooming/rubber band).
        try:
            if not getattr(self, "LMB_state", False):
                return
            if getattr(self, "_rubber_band", None) and getattr(self._rubber_band, "isActive", False):
                return
            if getattr(self, "ALT_state", False):
                return
        except Exception:
            return

        try:
            moved_nodes = {
                n: xy_pos for n, xy_pos in getattr(self, "_node_positions", {}).items()
                if getattr(n, "xy_pos", None) != xy_pos
            }
        except Exception:
            moved_nodes = {}

        try:
            if moved_nodes and not getattr(self, "COLLIDING_state", False):
                self.moving_nodes.emit(moved_nodes)
        except Exception:
            return

