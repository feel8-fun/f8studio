from __future__ import annotations

from typing import Any

from Qt import QtGui
from NodeGraphQt.constants import PipeEnum, PortTypeEnum
from NodeGraphQt.qgraphics.pipe import PipeItem

from .edge_rules import (
    EDGE_KIND_DATA,
    EDGE_KIND_EXEC,
    EDGE_KIND_STATE,
    connection_kind,
    normalize_edge_kind,
    port_kind,
    port_view_name,
)

_EDGE_PIPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    EDGE_KIND_EXEC: (245, 245, 245, 255),
    EDGE_KIND_DATA: (150, 150, 150, 255),
    EDGE_KIND_STATE: (246, 210, 64, 255),
}
_EDGE_PIPE_WIDTHS: dict[str, int] = {
    EDGE_KIND_EXEC: 4,
    EDGE_KIND_DATA: 2,
    EDGE_KIND_STATE: 2,
}


class F8StudioPipeItem(PipeItem):
    def __init__(self, input_port: Any = None, output_port: Any = None):
        # NodeGraphQt PipeItem.__init__() calls self.reset(), so initialize
        # subclass fields before super() to keep overridden reset() safe.
        self._edge_kind: str | None = None
        super().__init__(input_port=input_port, output_port=output_port)

    def set_connections(self, port1, port2):
        super().set_connections(port1, port2)
        self._refresh_edge_kind()
        self._apply_edge_style(width_delta=0)

    def draw_path(self, start_port, end_port=None, cursor_pos=None):
        self._refresh_edge_kind(start_port=start_port, end_port=end_port)
        super().draw_path(start_port, end_port=end_port, cursor_pos=cursor_pos)
        self._apply_edge_style(width_delta=1 if self._active or self._highlight else 0)
        self._apply_kind_visibility()

    def activate(self):
        self._active = True
        self._highlight = False
        if not self._apply_edge_style(width_delta=1):
            super().activate()

    def highlight(self):
        self._highlight = True
        self._active = False
        if not self._apply_edge_style(width_delta=1):
            super().highlight()

    def reset(self):
        self._active = False
        self._highlight = False
        if not self._apply_edge_style(width_delta=0):
            super().reset()
        self._draw_direction_pointer()

    def _refresh_edge_kind(self, *, start_port: Any = None, end_port: Any = None) -> None:
        out_name = ""
        in_name = ""

        if start_port is not None and end_port is not None:
            start_kind = port_kind(port_view_name(start_port))
            end_kind = port_kind(port_view_name(end_port))
            if start_kind == end_kind:
                if start_port.port_type == PortTypeEnum.OUT.value:
                    out_name = port_view_name(start_port)
                    in_name = port_view_name(end_port)
                elif start_port.port_type == PortTypeEnum.IN.value:
                    out_name = port_view_name(end_port)
                    in_name = port_view_name(start_port)

        if not out_name or not in_name:
            input_port = self.input_port
            output_port = self.output_port
            if input_port is not None and output_port is not None:
                out_name = port_view_name(output_port)
                in_name = port_view_name(input_port)

        kind = connection_kind(out_name, in_name)
        self._edge_kind = normalize_edge_kind(kind or "")

    def _apply_edge_style(self, *, width_delta: int) -> bool:
        kind = self._edge_kind
        if kind is None:
            return False
        color = _EDGE_PIPE_COLORS.get(kind)
        width = _EDGE_PIPE_WIDTHS.get(kind)
        if color is None or width is None:
            return False
        self.color = color
        self.style = PipeEnum.DRAW_TYPE_DEFAULT.value
        self.set_pipe_styling(
            color=color,
            width=max(1, int(width) + int(width_delta)),
            style=PipeEnum.DRAW_TYPE_DEFAULT.value,
        )
        return True

    def _apply_kind_visibility(self) -> None:
        if self.input_port is None or self.output_port is None:
            return
        kind = self._edge_kind
        if kind is None:
            return
        viewer = self.viewer()
        if viewer is None:
            return
        try:
            visible = bool(viewer.edge_kind_visible(kind))
        except (AttributeError, RuntimeError, TypeError):
            visible = True
        if not visible:
            self.setVisible(False)
            return
        if self.input_port is None or self.output_port is None:
            return
        is_visible = all(
            (
                self.input_port.isVisible(),
                self.output_port.isVisible(),
                self.input_port.node.isVisible(),
                self.output_port.node.isVisible(),
            )
        )
        self.setVisible(bool(is_visible))
        if not bool(is_visible):
            self._dir_pointer.setVisible(False)
            return
        self._dir_pointer.setBrush(QtGui.QColor(*_EDGE_PIPE_COLORS[kind]).darker(200))
