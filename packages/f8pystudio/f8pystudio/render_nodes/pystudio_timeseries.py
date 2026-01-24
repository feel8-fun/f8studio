from __future__ import annotations

import time
from typing import Any

from qtpy import QtCore, QtWidgets
from NodeGraphQt.nodes.base_node import NodeBaseWidget

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode

import pyqtgraph as pg  # type: ignore[import-not-found]


class _TimeSeriesPane(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if pg is None:
            label = QtWidgets.QLabel("pyqtgraph not installed")
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label)
            self._plot = None
            self._curve = None
            return

        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("bottom", "Time (s)", units="s")
        plot.setLabel("left", "Value")
        curve = plot.plot([], [], pen=pg.mkPen("b", width=2))
        plot.enableAutoRange(axis="y", enable=True)

        layout.addWidget(plot)
        self._plot = plot
        self._curve = curve

        self.setMinimumWidth(260)
        self.setMinimumHeight(160)

    def set_series(self, points: list[list[float] | tuple[int, float]], *, window_ms: int | None = None) -> None:
        if self._plot is None or self._curve is None:
            return
        now_ms = int(time.time() * 1000)
        xs: list[float] = []
        ys: list[float] = []
        for p in points:
            try:
                ts, v = p
                xs.append((float(ts) - float(now_ms)) / 1000.0)
                ys.append(float(v))
            except Exception:
                continue
        self._curve.setData(xs, ys)
        if window_ms is not None and window_ms > 0:
            window_s = float(window_ms) / 1000.0
            self._plot.setXRange(-window_s, 0.0, padding=0)


class _TimeSeriesWidget(NodeBaseWidget):
    def __init__(self, parent=None, name: str = "__timeseries", label: str = "") -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _TimeSeriesPane()
        self.set_custom_widget(self._pane)

    def get_value(self) -> object:
        return {}

    def set_value(self, value: object) -> None:
        return

    def set_series(self, points: list[list[float] | tuple[int, float]], *, window_ms: int | None = None) -> None:
        self._pane.set_series(points, window_ms=window_ms)


class PyStudioTimeSeriesNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.timeseries`.
    """

    def __init__(self):
        super().__init__()
        try:
            self.add_custom_widget(_TimeSeriesWidget(self.view, name="__timeseries", label=""))
        except Exception as e:
            pass

    def apply_ui_command(self, cmd: Any) -> None:
        try:
            if str(getattr(cmd, "command", "")) != "timeseries.set":
                return
            payload = getattr(cmd, "payload", {}) or {}
            points = list(payload.get("points") or [])
            window_ms = payload.get("windowMs")
        except Exception:
            return
        try:
            w = self.get_widget("__timeseries")
            if w and hasattr(w, "set_series"):
                w.set_series(points, window_ms=window_ms)
        except Exception:
            return
