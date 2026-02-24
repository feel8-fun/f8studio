from __future__ import annotations

import colorsys
import math
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets
from NodeGraphQt.nodes.base_node import NodeBaseWidget

from f8pysdk.shm import VideoShmReader

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from ..nodegraph.viz_operator_nodeitem import F8StudioVizOperatorNodeItem
from ..skeleton_protocols import skeleton_edges_for_protocol
from ..ui_bus import UiCommand

import pyqtgraph as pg  # type: ignore[import-not-found]

_STATE_UI_UPDATE = "uiUpdate"
_WIDGET_NAME = "__trackviz"


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    # Stable distinct-ish colors by hue.
    h = (int(track_id) * 0.161803398875) % 1.0  # golden ratio
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _color_for_kind(kind: str) -> tuple[int, int, int] | None:
    """
    Optional override colors (used when `kind` is present in samples).
    - match: yellow
    - track: green
    """
    k = str(kind or "").strip().lower()
    if k == "match":
        return (235, 200, 70)
    if k == "track":
        return (80, 220, 120)
    return None


def _flow_color(mag: float, max_mag: float) -> tuple[int, int, int]:
    den = max(1e-6, max_mag)
    t = max(0.0, min(1.0, float(mag) / den))
    # cyan -> orange/red
    r = int(40 + 215 * t)
    g = int(220 - 120 * t)
    b = int(220 - 200 * t)
    return r, g, b


class _TrackVizCanvas(pg.GraphicsObject):  # type: ignore[misc]
    """
    A single GraphicsObject that paints the whole scene.

    This avoids creating thousands of QGraphicsItems per refresh, which is the
    primary cause of UI jank/lag when history is enabled.
    """

    def __init__(self) -> None:
        super().__init__()
        self._payload: dict[str, Any] | None = None
        self._w: int = 0
        self._h: int = 0
        self._frame: QtGui.QImage | None = None

    def set_canvas_size(self, width: int, height: int) -> None:
        ww = max(0, int(width))
        hh = max(0, int(height))
        if ww == self._w and hh == self._h:
            return
        self._w = ww
        self._h = hh
        self.prepareGeometryChange()
        self.update()

    def set_video_frame(self, image: QtGui.QImage | None) -> None:
        self._frame = image
        self.update()

    def set_payload(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.update()

    def boundingRect(self) -> QtCore.QRectF:  # type: ignore[override]
        w = float(max(0, int(self._w)))
        h = float(max(0, int(self._h)))
        if w <= 0 or h <= 0:
            return QtCore.QRectF(0.0, 0.0, 1.0, 1.0)
        return QtCore.QRectF(0.0, 0.0, w, h)

    def paint(self, p: QtGui.QPainter, *args) -> None:  # type: ignore[override]
        payload = self._payload
        if not payload and not self._frame:
            return

        has_frame = self._frame is not None
        if has_frame and self._frame is not None:
            target_w = max(1.0, float(self._w if self._w > 0 else self._frame.width()))
            target_h = max(1.0, float(self._h if self._h > 0 else self._frame.height()))
            p.drawImage(QtCore.QRectF(0.0, 0.0, target_w, target_h), self._frame)

        if not payload:
            return

        try:
            now_ms = int(payload.get("nowMs") or 0)
        except Exception:
            now_ms = 0
        try:
            history_ms = int(payload.get("historyMs") or 0)
        except Exception:
            history_ms = 0

        tracks = payload.get("tracks") if isinstance(payload.get("tracks"), list) else []
        flow = payload.get("flow") if isinstance(payload.get("flow"), dict) else None

        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

        for t in tracks:
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("id"))
            except (TypeError, ValueError):
                continue
            hist = t.get("history")
            if not isinstance(hist, list) or not hist:
                continue

            r, g, b = _color_for_id(tid)

            centers: list[tuple[float, float, float, int, int, int]] = []

            # Draw fading boxes and compute trail centers.
            for s in hist:
                if not isinstance(s, dict):
                    continue
                try:
                    ts = int(s.get("tsMs") or 0)
                except (TypeError, ValueError):
                    ts = 0

                age = max(0, now_ms - ts) if now_ms > 0 and ts > 0 else 0
                if history_ms > 0:
                    a01 = max(0.05, 1.0 - (float(age) / float(history_ms)))
                else:
                    a01 = 1.0

                bb = s.get("bbox")
                if isinstance(bb, list) and len(bb) == 4:
                    try:
                        x1, y1, x2, y2 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        if w <= 0.5 or h <= 0.5:
                            continue
                        kind = ""
                        try:
                            kind = str(s.get("kind") or "")
                        except (TypeError, ValueError):
                            kind = ""
                        override = _color_for_kind(kind)
                        rr, gg, bb_ = override if override is not None else (r, g, b)
                        pen = QtGui.QPen(QtGui.QColor(rr, gg, bb_, int(a01 * 255)))
                        pen.setWidthF(2.0 if a01 > 0.7 else 1.0)
                        p.setPen(pen)
                        p.setBrush(QtCore.Qt.NoBrush)
                        p.drawRect(QtCore.QRectF(x1, y1, w, h))
                        centers.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5, a01, rr, gg, bb_))
                    except (TypeError, ValueError):
                        pass

            # Trail as fading segments.
            if len(centers) >= 2:
                for (x0, y0, a0, r0, g0, b0), (x1, y1, a1, r1, g1, b1) in zip(centers[:-1], centers[1:], strict=False):
                    a = max(0.05, min(1.0, (a0 + a1) * 0.5))
                    rr = int((r0 + r1) * 0.5)
                    gg = int((g0 + g1) * 0.5)
                    bb_ = int((b0 + b1) * 0.5)
                    pen = QtGui.QPen(QtGui.QColor(rr, gg, bb_, int(a * 255)))
                    pen.setWidthF(2.0)
                    p.setPen(pen)
                    p.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

            # Pose overlay (only latest sample).
            last = hist[-1]
            if not isinstance(last, dict):
                continue
            kps = last.get("keypoints")
            if not (isinstance(kps, list) and kps):
                continue

            pts: list[tuple[float, float]] = []
            for kp in kps:
                if not isinstance(kp, dict):
                    pts.append((float("nan"), float("nan")))
                    continue
                try:
                    x = float(kp.get("x"))
                    y = float(kp.get("y"))
                    pts.append((x, y))
                except (TypeError, ValueError):
                    pts.append((float("nan"), float("nan")))

            skeleton_protocol = ""
            try:
                skeleton_protocol = str(last.get("skeletonProtocol") or "").strip()
            except (AttributeError, TypeError, ValueError):
                skeleton_protocol = ""
            skeleton_edges = skeleton_edges_for_protocol(skeleton_protocol)
            if skeleton_edges is not None:
                pen = QtGui.QPen(QtGui.QColor(r, g, b, 220))
                pen.setWidthF(2.0)
                p.setPen(pen)
                for i, j in skeleton_edges:
                    if i >= len(pts) or j >= len(pts):
                        continue
                    x0, y0 = pts[i]
                    x1, y1 = pts[j]
                    if not (x0 == x0 and y0 == y0 and x1 == x1 and y1 == y1):
                        continue
                    p.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))

            # Keypoint dots.
            brush = QtGui.QBrush(QtGui.QColor(r, g, b, 220))
            p.setBrush(brush)
            p.setPen(QtCore.Qt.NoPen)
            rad = 3.0 if skeleton_edges is not None else 6.0
            for x, y in pts:
                if not (x == x and y == y):
                    continue
                p.drawEllipse(QtCore.QPointF(x, y), rad, rad)

        if isinstance(flow, dict):
            vectors = flow.get("vectors") if isinstance(flow.get("vectors"), list) else []
            try:
                scale = float(payload.get("flowArrowScale") or 1.0)
            except (TypeError, ValueError):
                scale = 1.0
            try:
                min_mag = float(payload.get("flowArrowMinMag") or 0.0)
            except (TypeError, ValueError):
                min_mag = 0.0
            scale = max(0.1, min(20.0, scale))
            min_mag = max(0.0, min(100.0, min_mag))

            mags: list[float] = []
            for item in vectors:
                if not isinstance(item, dict):
                    continue
                try:
                    mags.append(float(item.get("mag") or 0.0))
                except (TypeError, ValueError):
                    continue
            max_mag = max(mags) if mags else 1.0

            for item in vectors:
                if not isinstance(item, dict):
                    continue
                try:
                    x = float(item.get("x"))
                    y = float(item.get("y"))
                    dx = float(item.get("dx"))
                    dy = float(item.get("dy"))
                    mag = float(item.get("mag"))
                except (TypeError, ValueError):
                    continue
                if mag < min_mag:
                    continue

                x2 = x + dx * scale
                y2 = y + dy * scale
                rr, gg, bb_ = _flow_color(mag, max_mag)
                pen = QtGui.QPen(QtGui.QColor(rr, gg, bb_, 190))
                pen.setWidthF(1.2)
                p.setPen(pen)
                p.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x2, y2))

                vx = x2 - x
                vy = y2 - y
                vlen = math.hypot(vx, vy)
                if vlen < 1e-6:
                    continue
                ux = vx / vlen
                uy = vy / vlen
                head_len = max(3.0, min(8.0, 2.0 + 0.3 * vlen))
                # two wings rotated by +/- 30 deg
                c = 0.8660254037844386
                s = 0.5
                lx = c * ux - s * uy
                ly = s * ux + c * uy
                rx = c * ux + s * uy
                ry = -s * ux + c * uy
                p.drawLine(QtCore.QPointF(x2, y2), QtCore.QPointF(x2 - head_len * lx, y2 - head_len * ly))
                p.drawLine(QtCore.QPointF(x2, y2), QtCore.QPointF(x2 - head_len * rx, y2 - head_len * ry))


class _TrackVizPane(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
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
        top.addStretch()
        top.addWidget(self._update)

        layout.addLayout(top)

        if pg is None:
            label = QtWidgets.QLabel("pyqtgraph not installed")
            label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(label, 1)
            self._plot = None
            self._vb = None
            self._pending: dict[str, Any] | None = None
            return

        plot = pg.PlotWidget()
        plot.setBackground((16, 16, 16))
        plot.showGrid(x=False, y=False, alpha=0.2)
        plot.hideAxis("bottom")
        plot.hideAxis("left")
        plot.setMouseEnabled(x=True, y=True)
        plot.setMenuEnabled(False)
        plot.setMinimumSize(200, 120)
        vb = plot.getPlotItem().getViewBox()
        vb.invertY(True)
        vb.setAspectLocked(True)

        layout.addWidget(plot, 1)
        self._plot = plot
        self._vb = vb
        self._canvas = _TrackVizCanvas()
        vb.addItem(self._canvas)

        # self._status = QtWidgets.QLabel("")
        # self._status.setStyleSheet("color: rgb(160, 160, 160);")
        # layout.addWidget(self._status)

        self._pending = None
        self._last_wh: tuple[int, int] | None = None
        self._scene_size: tuple[int, int] | None = None
        self._video_shm_name = ""
        self._video_shm_throttle_ms = 33
        self._video_reader: VideoShmReader | None = None
        self._video_frame_id = 0
        self._video_frame_bytes: bytes | None = None
        self._video_size: tuple[int, int] | None = None
        self._video_timer = QtCore.QTimer(self)
        self._video_timer.timeout.connect(self._tick_video)  # type: ignore[attr-defined]
        self._video_timer.setInterval(33)

        # Smaller default footprint (similar to VideoSHM view).
        self.setMinimumWidth(240)
        self.setMinimumHeight(180)
        self.setMaximumWidth(240)
        self.setMaximumHeight(180)

    def update_checkbox(self) -> QtWidgets.QCheckBox:
        return self._update

    def update_enabled(self) -> bool:
        return bool(self._update.isChecked())

    def set_update_enabled(self, enabled: bool) -> None:
        self._update.setChecked(bool(enabled))
        self._sync_video_timer_with_update_state()
        if self.update_enabled() and self._pending is not None:
            p = self._pending
            self._pending = None
            self.set_scene(p)

    def set_scene(self, payload: dict[str, Any]) -> None:
        if self._plot is None or self._vb is None:
            return
        if not self.update_enabled():
            self._pending = payload
            return

        try:
            video_shm_name = str(payload.get("videoShmName") or "").strip()
        except (AttributeError, TypeError, ValueError):
            video_shm_name = ""
        try:
            video_shm_throttle_ms = int(payload.get("throttleMs") or 33)
        except (AttributeError, TypeError, ValueError):
            video_shm_throttle_ms = 33

        self._set_video_config(
            shm_name=video_shm_name,
            throttle_ms=video_shm_throttle_ms,
        )
        try:
            pw = int(payload.get("width") or 0)
            ph = int(payload.get("height") or 0)
            self._scene_size = (pw, ph) if pw > 0 and ph > 0 else None
        except (AttributeError, TypeError, ValueError):
            self._scene_size = None
        try:
            self._canvas.set_payload(payload)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        self._sync_canvas_geometry()


    def detach(self) -> None:
        try:
            self._video_timer.stop()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._reset_video_reader()

    def _set_video_config(self, *, shm_name: str, throttle_ms: int) -> None:
        next_throttle_ms = max(1, int(throttle_ms))
        if self._video_shm_throttle_ms != next_throttle_ms:
            self._video_shm_throttle_ms = next_throttle_ms
            self._video_timer.setInterval(self._video_shm_throttle_ms)

        next_name = str(shm_name or "").strip()
        if next_name != self._video_shm_name:
            self._video_shm_name = next_name
            self._reset_video_reader()

        self._sync_video_timer_with_update_state()

    def _sync_video_timer_with_update_state(self) -> None:
        if self._video_shm_name and self.update_enabled():
            if not self._video_timer.isActive():
                self._video_timer.start()
            return
        if self._video_timer.isActive():
            self._video_timer.stop()
        if not self._video_shm_name:
            self._canvas.set_video_frame(None)
            self._video_size = None
            self._sync_canvas_geometry()

    def _sync_canvas_geometry(self) -> None:
        if self._vb is None:
            return
        width = 0
        height = 0
        # Priority: VideoSHM frame size first.
        if self._video_size is not None:
            width, height = self._video_size
        elif self._scene_size is not None:
            width, height = self._scene_size
        if width <= 0 or height <= 0:
            return
        self._canvas.set_canvas_size(width, height)
        wh = (int(width), int(height))
        if self._last_wh == wh:
            return
        self._last_wh = wh
        self._vb.setRange(
            xRange=(0.0, float(width)),
            yRange=(0.0, float(height)),
            padding=0.0,
            disableAutoRange=True,
        )

    def _reset_video_reader(self) -> None:
        try:
            if self._video_reader is not None:
                self._video_reader.close()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._video_reader = None
        self._video_frame_id = 0
        self._video_frame_bytes = None
        self._video_size = None
        self._sync_canvas_geometry()

    def _ensure_video_reader(self) -> bool:
        if self._video_reader is not None:
            return True
        if not self._video_shm_name:
            return False
        try:
            reader = VideoShmReader(self._video_shm_name)
            reader.open(use_event=False)
            self._video_reader = reader
            return True
        except Exception as exc:
            if hasattr(self, "_status") and self._status is not None:
                self._status.setText(f"video shm open failed: {exc}")
            self._video_reader = None
            return False

    def _tick_video(self) -> None:
        if not self.update_enabled():
            return
        if not self._ensure_video_reader():
            return
        if self._video_reader is None:
            return
        try:
            header, payload = self._video_reader.read_latest_bgra()
        except Exception as exc:
            if hasattr(self, "_status") and self._status is not None:
                self._status.setText(f"video shm read failed: {exc}")
            return
        if header is None or payload is None:
            return
        frame_id = int(header.frame_id)
        if frame_id == self._video_frame_id:
            return

        w = int(header.width)
        h = int(header.height)
        pitch = int(header.pitch)
        if w <= 0 or h <= 0 or pitch <= 0:
            return

        frame_bytes = bytes(payload)
        self._video_frame_bytes = frame_bytes
        self._video_frame_id = frame_id
        try:
            img = QtGui.QImage(frame_bytes, w, h, pitch, QtGui.QImage.Format_ARGB32)
            safe_img = img.copy()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return
        self._video_size = (w, h)
        self._canvas.set_video_frame(safe_img)
        self._sync_canvas_geometry()


class _TrackVizWidget(NodeBaseWidget):
    def __init__(
        self,
        parent=None,
        name: str = _WIDGET_NAME,
        label: str = "",
        *,
        on_update_toggled: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _TrackVizPane()
        self.set_custom_widget(self._pane)
        self._block = False
        self._on_update_toggled_cb = on_update_toggled
        self._pane.update_checkbox().toggled.connect(self.on_value_changed)  # type: ignore[attr-defined]
        self._pane.update_checkbox().toggled.connect(self._on_update_toggled)

    def get_value(self) -> object:
        return {"update": bool(self._pane.update_enabled())}

    def set_value(self, value: object) -> None:
        _ = value

    def set_update_enabled(self, enabled: bool) -> None:
        try:
            self._block = True
            self._pane.set_update_enabled(enabled)
        finally:
            self._block = False

    def on_value_changed(self, *args, **kwargs):
        if self._block:
            return
        return super().on_value_changed(*args, **kwargs)

    def _on_update_toggled(self, enabled: bool) -> None:
        if self._block:
            return
        cb = self._on_update_toggled_cb
        if cb is None:
            return
        cb(bool(enabled))

    def set_scene(self, payload: dict[str, Any]) -> None:
        self._pane.set_scene(payload)

    def detach(self) -> None:
        self._pane.detach()


class VizTrackRenderNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.viz.track`.
    """

    def __init__(self):
        super().__init__(qgraphics_item=F8StudioVizOperatorNodeItem)
        self.add_ephemeral_widget(
            _TrackVizWidget(
                self.view,
                name=_WIDGET_NAME,
                label="",
                on_update_toggled=self._on_update_toggled,
            )
        )
        self._sync_update_checkbox_from_state(default=True)

    def sync_from_spec(self) -> None:
        super().sync_from_spec()
        self._sync_update_checkbox_from_state(default=True)

    def set_property(self, name, value, push_undo=True):  # type: ignore[override]
        super().set_property(name, value, push_undo=push_undo)
        if str(name or "").strip() == _STATE_UI_UPDATE:
            self._sync_update_checkbox_from_state(default=bool(value))

    def _on_update_toggled(self, enabled: bool) -> None:
        self.set_state_bool(_STATE_UI_UPDATE, bool(enabled))

    def _sync_update_checkbox_from_state(self, *, default: bool) -> None:
        self.sync_bool_state_to_widget(
            state_name=_STATE_UI_UPDATE,
            default=default,
            widget_name=_WIDGET_NAME,
            widget_type=_TrackVizWidget,
            apply_value=_TrackVizWidget.set_update_enabled,
        )

    def _widget(self) -> _TrackVizWidget | None:
        return self.widget_by_name(_WIDGET_NAME, _TrackVizWidget)

    def apply_ui_command(self, cmd: UiCommand) -> None:
        command = str(cmd.command or "")
        if command == "viz.track.detach":
            widget = self._widget()
            if widget is not None:
                widget.detach()
            return
        if command != "viz.track.set":
            return
        try:
            payload = dict(cmd.payload or {})
        except (AttributeError, TypeError, ValueError):
            return
        widget = self._widget()
        if widget is None:
            return
        widget.set_scene(dict(payload))
