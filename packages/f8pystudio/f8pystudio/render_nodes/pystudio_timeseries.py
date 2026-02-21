from __future__ import annotations

import time
from typing import Any, Callable

from qtpy import QtCore, QtWidgets
from NodeGraphQt.nodes.base_node import NodeBaseWidget

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from ..color_table import series_colors
from ..nodegraph.viz_operator_nodeitem import F8StudioVizOperatorNodeItem
from ..ui_bus import UiCommand

import pyqtgraph as pg  # type: ignore[import-not-found]


class _TimeSeriesPane(QtWidgets.QWidget):
    clear_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._clear = QtWidgets.QToolButton()
        self._clear.setText("CLEAR")
        self._clear.setAutoRaise(True)
        self._clear.setToolTip("Clear accumulated samples.")
        self._clear.setStyleSheet(
            """
            QToolButton {
                color: rgb(225, 225, 225);
                background: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 45);
                padding: 2px 8px;
                border-radius: 3px;
            }
            QToolButton:hover { background: rgba(255, 255, 255, 35); }
            QToolButton:pressed { background: rgba(255, 255, 255, 25); }
            """
        )
        self._update = QtWidgets.QCheckBox("Update")
        self._update.setChecked(True)
        self._update.setStyleSheet(
            """
            QCheckBox { color: rgb(225, 225, 225); }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border: 1px solid rgba(255, 255, 255, 90);
                background: rgba(0, 0, 0, 35);
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                image: none;
                background: rgba(120, 200, 255, 90);
            }
            """
        )
        top.addWidget(self._clear, 0)
        top.addWidget(self._update, 0)
        top.addStretch(1)
        layout.addLayout(top)

        self._pending_payload: dict[str, Any] | None = None
        self._clear.clicked.connect(self._on_clear_clicked)
        self._update.toggled.connect(self._on_update_toggled)

        if pg is None:
            label = QtWidgets.QLabel("pyqtgraph not installed")
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label)
            self._plot = None
            self._curves: dict[str, object] = {}
            self._legend = None
            return

        plot = pg.PlotWidget()
        plot.setBackground((16, 16, 16))
        plot.showGrid(x=True, y=True, alpha=0.25)
        # Keep the plot compact: omit axis captions ("Time/Value") which take
        # space and are usually redundant for viz nodes.
        try:
            plot.getAxis("bottom").setLabel("")
            plot.getAxis("left").setLabel("")
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            # Tighten internal margins to reduce wasted chrome.
            pi = plot.getPlotItem()
            if pi is not None:
                try:
                    pi.layout.setContentsMargins(2, 2, 2, 2)
                except (AttributeError, RuntimeError, TypeError):
                    pass
                try:
                    pi.setDefaultPadding(0.02)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
        except (AttributeError, RuntimeError, TypeError):
            pass
        plot.enableAutoRange(axis="y", enable=True)
        self._legend = None

        layout.addWidget(plot)
        self._plot = plot
        self._curves: dict[str, pg.PlotDataItem] = {}

        # Default footprint for viz nodes should be small; users can resize as needed.
        self.setMinimumWidth(200)
        self.setMinimumHeight(150)
        self.setMaximumWidth(200)
        self.setMaximumHeight(150)

    def update_checkbox(self) -> QtWidgets.QCheckBox:
        return self._update

    def update_enabled(self) -> bool:
        try:
            return bool(self._update.isChecked())
        except (AttributeError, RuntimeError, TypeError):
            return True

    def set_update_enabled(self, enabled: bool) -> None:
        try:
            self._update.setChecked(bool(enabled))
        except (AttributeError, RuntimeError, TypeError):
            pass
        if self.update_enabled():
            self._apply_pending_payload()

    def clear_series(self) -> None:
        self._pending_payload = None
        if self._plot is None:
            return
        for k, c in list(self._curves.items()):
            try:
                self._plot.removeItem(c)
            except (AttributeError, RuntimeError, TypeError):
                pass
            self._curves.pop(k, None)

    def _on_clear_clicked(self) -> None:
        self.clear_series()
        self.clear_requested.emit()

    def _on_update_toggled(self, enabled: bool) -> None:
        if not enabled:
            return
        self._apply_pending_payload()

    def _apply_pending_payload(self) -> None:
        payload = self._pending_payload
        self._pending_payload = None
        if not payload:
            return
        self.set_series_map(
            payload["series"],
            colors=payload["colors"],
            window_ms=payload["window_ms"],
            now_ms=payload["now_ms"],
            show_legend=payload["show_legend"],
            y_min=payload["y_min"],
            y_max=payload["y_max"],
        )

    def set_series_map(
        self,
        series: dict[str, list[list[float] | tuple[int, float]]],
        *,
        colors: dict[str, tuple[int, int, int]] | None = None,
        window_ms: int | None = None,
        now_ms: int | None = None,
        show_legend: bool | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> None:
        if not self.update_enabled():
            self._pending_payload = {
                "series": series,
                "colors": colors,
                "window_ms": window_ms,
                "now_ms": now_ms,
                "show_legend": show_legend,
                "y_min": y_min,
                "y_max": y_max,
            }
            return
        if self._plot is None:
            return
        if not isinstance(series, dict) or not series:
            self.clear_series()
            return

        if now_ms is None:
            now_ms = int(time.time() * 1000)
        colors = colors or {}

        if show_legend is not None:
            try:
                if bool(show_legend):
                    if self._legend is None:
                        self._legend = self._plot.addLegend(offset=(10, 10))
                    self._legend.setVisible(True)
                else:
                    if self._legend is not None:
                        self._legend.setVisible(False)
            except (AttributeError, RuntimeError, TypeError):
                pass

        active = {str(k): v for k, v in series.items() if isinstance(k, str) and isinstance(v, list)}

        # Remove stale curves.
        for k in list(self._curves.keys()):
            if k not in active:
                c = self._curves.pop(k, None)
                if c is not None:
                    try:
                        self._plot.removeItem(c)
                    except (AttributeError, RuntimeError, TypeError):
                        pass

        for name, points in active.items():
            curve = self._curves.get(name)
            if curve is None:
                rgb = colors.get(name) or (180, 180, 180)
                try:
                    pen = pg.mkPen((*rgb, 255), width=2)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pen = pg.mkPen("w", width=2)
                curve = self._plot.plot([], [], pen=pen, name=name)
                self._curves[name] = curve

            xs: list[float] = []
            ys: list[float] = []
            for p in points:
                try:
                    ts, v = p
                    xs.append((float(ts) - float(now_ms)) / 1000.0)
                    ys.append(float(v))
                except (TypeError, ValueError):
                    continue
            try:
                curve.setData(xs, ys)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        # Y range handling:
        # - if both min/max are set and max > min: fixed range
        # - otherwise: auto range based on data
        fixed = False
        try:
            if y_min is not None and y_max is not None and float(y_max) > float(y_min):
                fixed = True
        except (TypeError, ValueError):
            fixed = False
        try:
            if fixed:
                self._plot.enableAutoRange(axis="y", enable=False)
                self._plot.setYRange(float(y_min), float(y_max), padding=0.0)
            else:
                self._plot.enableAutoRange(axis="y", enable=True)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

        if window_ms is not None and window_ms > 0:
            window_s = float(window_ms) / 1000.0
            self._plot.setXRange(-window_s, 0.0, padding=0)


class _TimeSeriesWidget(NodeBaseWidget):
    def __init__(
        self,
        parent=None,
        name: str = "__timeseries",
        label: str = "",
        *,
        on_clear_requested: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _TimeSeriesPane()
        self.set_custom_widget(self._pane)
        self._block = False
        self._on_clear_requested_cb = on_clear_requested
        self._pane.update_checkbox().toggled.connect(self.on_value_changed)  # type: ignore[attr-defined]
        self._pane.clear_requested.connect(self._on_clear_requested)

    def get_value(self) -> object:
        return {"update": bool(self._pane.update_enabled())}

    def set_value(self, value: object) -> None:
        update_enabled = True
        if isinstance(value, dict):
            update_enabled = bool(value.get("update", True))
        try:
            self._block = True
            self._pane.set_update_enabled(update_enabled)
        finally:
            self._block = False

    def on_value_changed(self, *args, **kwargs):
        if self._block:
            return
        return super().on_value_changed(*args, **kwargs)

    def _on_clear_requested(self) -> None:
        cb = self._on_clear_requested_cb
        if cb is None:
            return
        try:
            cb()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return

    def set_series_map(
        self,
        series: dict[str, list[list[float] | tuple[int, float]]],
        *,
        colors: dict[str, tuple[int, int, int]] | None = None,
        window_ms: int | None = None,
        now_ms: int | None = None,
        show_legend: bool | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> None:
        self._pane.set_series_map(
            series,
            colors=colors,
            window_ms=window_ms,
            now_ms=now_ms,
            show_legend=show_legend,
            y_min=y_min,
            y_max=y_max,
        )


class PyStudioTimeSeriesNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.timeseries`.
    """

    def __init__(self):
        super().__init__(qgraphics_item=F8StudioVizOperatorNodeItem)
        self._clear_nonce = 0
        self.add_custom_widget(
            _TimeSeriesWidget(
                self.view,
                name="__timeseries",
                label="",
                on_clear_requested=self._on_clear_requested,
            )
        )

    def _on_clear_requested(self) -> None:
        self._clear_nonce = int(self._clear_nonce) + 1
        try:
            self.set_property("clearNonce", int(self._clear_nonce), push_undo=False)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return

    def sync_from_spec(self) -> None:
        super().sync_from_spec()
        spec = self.spec
        ports = spec.dataInPorts
            
        try:
            colors = series_colors([str(p.name) for p in ports])
        except (AttributeError, TypeError):
            colors = {}
        for i, p in enumerate(ports):
            name = str(p.name)
            if not name:
                continue
            rgb = colors.get(name) or (180, 180, 180)
            try:
                port = self.get_input(f"[D]{name}")
            except (AttributeError, RuntimeError, TypeError):
                port = None

            if port is None:
                continue

            port.view.display_name = False
            port.color = rgb
            port.border_color = rgb
            port.view.setToolTip(str(name))

    def apply_ui_command(self, cmd: UiCommand) -> None:
        if str(cmd.command) != "timeseries.set":
            return
        try:
            payload = dict(cmd.payload or {})
        except (AttributeError, TypeError):
            return

        window_ms = payload.get("windowMs")
        now_ms = payload.get("nowMs")
        show_legend = payload.get("showLegend")
        min_val = payload.get("minVal")
        max_val = payload.get("maxVal")
        series = payload.get("series")
        colors_raw = payload.get("colors")

        if not (isinstance(series, dict) and series):
            # Backwards compatibility (single series).
            points = list(payload.get("points") or [])
            series = {"value": points}

        colors: dict[str, tuple[int, int, int]] = {}
        if isinstance(colors_raw, dict):
            for k, v in colors_raw.items():
                if not isinstance(k, str):
                    continue
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    try:
                        colors[k] = (int(v[0]), int(v[1]), int(v[2]))
                    except (TypeError, ValueError):
                        continue
        else:
            # Derive colors from current spec order when runtime didn't provide them.
            try:
                ports = list(self.spec.dataInPorts or [])
            except (AttributeError, TypeError):
                ports = []
            try:
                colors = series_colors([str(p.name) for p in ports])
            except (AttributeError, TypeError):
                colors = {}

        y_min: float | None = None
        y_max: float | None = None
        try:
            if min_val is not None:
                y_min = float(min_val)
        except (TypeError, ValueError):
            y_min = None
        try:
            if max_val is not None:
                y_max = float(max_val)
        except (TypeError, ValueError):
            y_max = None

        try:
            w = self.get_widget("__timeseries")
        except (AttributeError, RuntimeError, TypeError):
            return
        if not w:
            return
        try:
            w.set_series_map(
                series,
                colors=colors,
                window_ms=window_ms,
                now_ms=now_ms,
                show_legend=bool(show_legend) if show_legend is not None else None,
                y_min=y_min,
                y_max=y_max,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return
