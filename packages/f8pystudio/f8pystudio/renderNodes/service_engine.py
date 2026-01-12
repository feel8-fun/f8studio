from __future__ import annotations

from typing import Any

from NodeGraphQt import BackdropNode


class OperatorRunnerNode(BackdropNode):
    """
    Service container node (engine runner).

    This is a Backdrop-based container that can wrap operator nodes for deployment.
    """

    spec: Any

    def __init__(self, qgraphics_item=None):
        super().__init__(qgraphics_views=qgraphics_item)
        try:
            self.set_text("engine")  # type: ignore[attr-defined]
        except Exception:
            pass

    def contained_nodes(self) -> list[Any]:
        try:
            nodes = list(self.nodes())  # type: ignore[attr-defined]
        except Exception:
            nodes = []
        # Exclude backdrops/groups.
        out: list[Any] = []
        for n in nodes:
            try:
                if hasattr(n, "type_") and getattr(n, "type_", "") == self.type_:
                    continue
            except Exception:
                pass
            out.append(n)
        return out

    def wrap_selected_nodes(self) -> None:
        try:
            graph = self.graph  # type: ignore[attr-defined]
            selected = list(graph.selected_nodes() or []) if graph is not None else []
        except Exception:
            selected = []
        if not selected:
            return
        try:
            self.wrap_nodes(selected)  # type: ignore[attr-defined]
        except Exception:
            return
