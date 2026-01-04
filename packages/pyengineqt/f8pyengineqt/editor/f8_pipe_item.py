from __future__ import annotations

from typing import Any

from NodeGraphQt.constants import PipeEnum
from NodeGraphQt.qgraphics.pipe import PipeItem


def pipe_style_for_kind(kind: str | None) -> tuple[tuple[int, int, int, int], int, int]:
    """
    Returns (color_rgba, width, style_enum) for the given port kind.
    """
    if kind == "exec":
        return ((230, 230, 230, 220), 5, PipeEnum.DRAW_TYPE_DEFAULT.value)
    if kind == "data":
        return ((150, 150, 150, 210), 2, PipeEnum.DRAW_TYPE_DEFAULT.value)
    if kind == "state":
        return ((200, 200, 80, 230), 2, PipeEnum.DRAW_TYPE_DASHED.value)
    return (tuple(PipeEnum.COLOR.value), 2, PipeEnum.DRAW_TYPE_DEFAULT.value)


class F8PipeItem(PipeItem):
    """
    PipeItem with per-connection base styling (kept across reset/highlight).
    """

    def __init__(self, *, kind: str | None = None) -> None:
        self._base_kind = kind
        self._base_color, self._base_width, self._base_style = pipe_style_for_kind(kind)
        super().__init__()

        self.color = self._base_color
        self.style = self._base_style
        self.set_pipe_styling(color=self._base_color, width=self._base_width, style=self._base_style)

    def set_kind(self, kind: str | None) -> None:
        self._base_kind = kind
        self._base_color, self._base_width, self._base_style = pipe_style_for_kind(kind)
        self.color = self._base_color
        self.style = self._base_style
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._highlight = False
        self.set_pipe_styling(color=self.color, width=self._base_width, style=self.style)
        self._draw_direction_pointer()

    def activate(self) -> None:
        self._active = True
        width = max(int(self._base_width), 3)
        self.set_pipe_styling(
            color=PipeEnum.ACTIVE_COLOR.value,
            width=width,
            style=PipeEnum.DRAW_TYPE_DEFAULT.value,
        )

    def highlight(self) -> None:
        self._highlight = True
        width = max(int(self._base_width), 2)
        self.set_pipe_styling(
            color=PipeEnum.HIGHLIGHT_COLOR.value,
            width=width,
            style=PipeEnum.DRAW_TYPE_DEFAULT.value,
        )
