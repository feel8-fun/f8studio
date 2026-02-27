from __future__ import annotations

from typing import Any, Callable

import numpy as np
from NodeGraphQt.nodes.base_node import NodeBaseWidget
from qtpy import QtCore, QtGui, QtWidgets

from f8pysdk.shm import VIDEO_FORMAT_FLOW2_F16, VideoShmReader

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from ..nodegraph.viz_operator_nodeitem import F8StudioVizOperatorNodeItem
from ..ui_bus import UiCommand

_STATE_UI_UPDATE = "uiUpdate"
_WIDGET_NAME = "__videoshm"


def _hsv_to_rgb_u8(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    m = i % 6

    r = np.select([m == 0, m == 1, m == 2, m == 3, m == 4], [v, q, p, p, t], default=v)
    g = np.select([m == 0, m == 1, m == 2, m == 3, m == 4], [t, v, v, q, p], default=p)
    b = np.select([m == 0, m == 1, m == 2, m == 3, m == 4], [p, p, t, v, v], default=q)
    rgb = np.stack([r, g, b], axis=2)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


class _VideoShmPane(QtWidgets.QWidget):
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

        self._image = QtWidgets.QLabel()
        self._image.setAlignment(QtCore.Qt.AlignCenter)
        self._image.setMinimumSize(320, 180)
        self._image.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._image.setStyleSheet("background: rgba(0,0,0,60); border: 1px solid rgba(255,255,255,25);")

        layout.addLayout(top)
        layout.addWidget(self._image, 1)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)  # type: ignore[attr-defined]
        self._timer.setInterval(33)

        self._video_reader: VideoShmReader | None = None
        self._flow_reader: VideoShmReader | None = None
        self._video_shm_name = ""
        self._flow_shm_name = ""
        self._flow_display_mode = "off"
        self._flow_mag_scale = 20.0
        self._flow_stride = 12
        self._scale_mode = "native"

        self._last_video_frame_id = 0
        self._last_flow_frame_id = 0
        self._latest_video: QtGui.QImage | None = None
        self._last_pixmap_size: QtCore.QSize | None = None
        self.setMinimumWidth(360)
        self.setMinimumHeight(240)

    def update_checkbox(self) -> QtWidgets.QCheckBox:
        return self._update

    def update_enabled(self) -> bool:
        return bool(self._update.isChecked())

    def set_update_enabled(self, enabled: bool) -> None:
        self._update.setChecked(bool(enabled))
        self._sync_timer_with_update_state()

    def set_config(
        self,
        *,
        shm_name: str,
        throttle_ms: int,
        flow_shm_name: str,
        flow_display_mode: str,
        flow_mag_scale: float,
        flow_stride: int,
        scale_mode: str,
    ) -> None:
        next_video = str(shm_name or "").strip()
        next_flow = str(flow_shm_name or "").strip()
        next_mode = str(flow_display_mode or "off").strip().lower()
        if next_mode not in ("off", "hsv", "arrows"):
            next_mode = "off"

        if next_video != self._video_shm_name:
            self._video_shm_name = next_video
            self._reset_video_reader()
        if next_flow != self._flow_shm_name:
            self._flow_shm_name = next_flow
            self._reset_flow_reader()
        self._flow_display_mode = next_mode
        self._flow_mag_scale = max(0.1, float(flow_mag_scale))
        self._flow_stride = max(2, int(flow_stride))
        next_scale_mode = str(scale_mode or "native").strip().lower()
        self._scale_mode = next_scale_mode if next_scale_mode in ("native", "fit") else "native"
        self._timer.setInterval(max(1, int(throttle_ms)))
        self._sync_timer_with_update_state()

    def detach(self) -> None:
        try:
            self._timer.stop()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._reset_video_reader()
        self._reset_flow_reader()

    def _reset_video_reader(self) -> None:
        try:
            if self._video_reader is not None:
                self._video_reader.close()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._video_reader = None
        self._last_video_frame_id = 0
        self._latest_video = None

    def _reset_flow_reader(self) -> None:
        try:
            if self._flow_reader is not None:
                self._flow_reader.close()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._flow_reader = None
        self._last_flow_frame_id = 0

    def _sync_timer_with_update_state(self) -> None:
        has_source = bool(self._video_shm_name) or bool(self._flow_shm_name)
        if self.update_enabled() and has_source:
            if not self._timer.isActive():
                self._timer.start()
            return
        if self._timer.isActive():
            self._timer.stop()

    def _ensure_video_reader(self) -> bool:
        if self._video_reader is not None:
            return True
        if not self._video_shm_name:
            return False
        try:
            r = VideoShmReader(self._video_shm_name)
            r.open(use_event=False)
            self._video_reader = r
            return True
        except Exception:
            self._video_reader = None
            return False

    def _ensure_flow_reader(self) -> bool:
        if self._flow_reader is not None:
            return True
        if not self._flow_shm_name:
            return False
        try:
            r = VideoShmReader(self._flow_shm_name)
            r.open(use_event=False)
            self._flow_reader = r
            return True
        except Exception:
            self._flow_reader = None
            return False

    def _update_video_cache(self) -> None:
        if not self._ensure_video_reader() or self._video_reader is None:
            return
        try:
            header, payload = self._video_reader.read_latest_bgra()
        except Exception:
            return
        if header is None or payload is None:
            return
        frame_id = int(header.frame_id)
        if frame_id == self._last_video_frame_id:
            return
        w = int(header.width)
        h = int(header.height)
        pitch = int(header.pitch)
        if w <= 0 or h <= 0 or pitch <= 0:
            return
        frame_bytes = bytes(payload)
        try:
            img = QtGui.QImage(frame_bytes, w, h, pitch, QtGui.QImage.Format_ARGB32)
            self._latest_video = img.copy()
            self._last_video_frame_id = frame_id
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return

    def _render_hsv_flow(self, payload: memoryview, width: int, height: int, pitch: int) -> QtGui.QImage | None:
        row_bytes = width * 4
        if row_bytes <= 0 or pitch < row_bytes:
            return None
        flow_rows = []
        for y in range(height):
            off = y * pitch
            flow_rows.append(bytes(payload[off : off + row_bytes]))
        flow_bytes = b"".join(flow_rows)
        uv = np.frombuffer(flow_bytes, dtype="<f2").reshape(height, width, 2).astype(np.float32)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
        mag = np.sqrt(u * u + v * v)
        angle = np.arctan2(v, u)
        hue = ((angle + np.pi) / (2.0 * np.pi)).astype(np.float32)
        sat = np.ones_like(hue, dtype=np.float32)
        val = np.clip(mag / float(self._flow_mag_scale), 0.0, 1.0).astype(np.float32)
        rgb = _hsv_to_rgb_u8(hue, sat, val)
        try:
            img = QtGui.QImage(rgb.data, width, height, width * 3, QtGui.QImage.Format_RGB888)
            return img.copy()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def _render_arrows_flow(self, payload: memoryview, width: int, height: int, pitch: int) -> QtGui.QImage | None:
        if self._latest_video is not None and self._latest_video.width() == width and self._latest_video.height() == height:
            canvas = self._latest_video.copy()
        else:
            canvas = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
            canvas.fill(QtGui.QColor(16, 16, 16))

        row_bytes = width * 4
        if row_bytes <= 0 or pitch < row_bytes:
            return canvas
        flow_rows = []
        for y in range(height):
            off = y * pitch
            flow_rows.append(bytes(payload[off : off + row_bytes]))
        flow_bytes = b"".join(flow_rows)
        uv = np.frombuffer(flow_bytes, dtype="<f2").reshape(height, width, 2).astype(np.float32)

        p = QtGui.QPainter(canvas)
        try:
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            stride = max(2, int(self._flow_stride))
            scale = 20.0 / float(self._flow_mag_scale)
            for y in range(stride // 2, height, stride):
                for x in range(stride // 2, width, stride):
                    dx = float(uv[y, x, 0])
                    dy = float(uv[y, x, 1])
                    mag = float(np.hypot(dx, dy))
                    if mag <= 0.01:
                        continue
                    v01 = max(0.0, min(1.0, mag / float(self._flow_mag_scale)))
                    rr = int(40 + 215 * v01)
                    gg = int(220 - 120 * v01)
                    bb = int(220 - 200 * v01)
                    pen = QtGui.QPen(QtGui.QColor(rr, gg, bb, 190))
                    pen.setWidthF(1.2)
                    p.setPen(pen)
                    x2 = float(x) + dx * scale
                    y2 = float(y) + dy * scale
                    p.drawLine(QtCore.QPointF(float(x), float(y)), QtCore.QPointF(x2, y2))
        finally:
            p.end()
        return canvas

    def _present(self, image: QtGui.QImage) -> None:
        pix = QtGui.QPixmap.fromImage(image)
        if self._scale_mode == "fit":
            target = self._image.size()
            if self._last_pixmap_size != target:
                self._last_pixmap_size = target
            try:
                pix = pix.scaled(target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            except (AttributeError, RuntimeError, TypeError):
                pass
        self._image.setPixmap(pix)

    def _tick(self) -> None:
        if not self.update_enabled():
            return
        self._update_video_cache()

        if self._flow_display_mode == "off":
            if self._latest_video is not None:
                self._present(self._latest_video)
            return

        if not self._ensure_flow_reader() or self._flow_reader is None:
            return
        try:
            header, payload = self._flow_reader.read_latest_frame()
        except Exception:
            return
        if header is None or payload is None:
            return
        if int(header.fmt) != VIDEO_FORMAT_FLOW2_F16:
            return
        frame_id = int(header.frame_id)
        if frame_id == self._last_flow_frame_id:
            return
        self._last_flow_frame_id = frame_id

        w = int(header.width)
        h = int(header.height)
        pitch = int(header.pitch)
        if w <= 0 or h <= 0 or pitch <= 0:
            return

        image: QtGui.QImage | None
        if self._flow_display_mode == "hsv":
            image = self._render_hsv_flow(payload, w, h, pitch)
        else:
            image = self._render_arrows_flow(payload, w, h, pitch)
        if image is None:
            return
        self._present(image)


class _VideoShmWidget(NodeBaseWidget):
    def __init__(
        self,
        parent=None,
        name: str = _WIDGET_NAME,
        label: str = "",
        *,
        on_update_toggled: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _VideoShmPane()
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

    def set_config(
        self,
        *,
        shm_name: str,
        throttle_ms: int,
        flow_shm_name: str,
        flow_display_mode: str,
        flow_mag_scale: float,
        flow_stride: int,
        scale_mode: str,
    ) -> None:
        self._pane.set_config(
            shm_name=shm_name,
            throttle_ms=throttle_ms,
            flow_shm_name=flow_shm_name,
            flow_display_mode=flow_display_mode,
            flow_mag_scale=flow_mag_scale,
            flow_stride=flow_stride,
            scale_mode=scale_mode,
        )

    def detach(self) -> None:
        self._pane.detach()


class VizVideoRenderNode(F8StudioOperatorBaseNode):
    def __init__(self):
        super().__init__(qgraphics_item=F8StudioVizOperatorNodeItem)
        self.add_ephemeral_widget(
            _VideoShmWidget(
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
            widget_type=_VideoShmWidget,
            apply_value=_VideoShmWidget.set_update_enabled,
        )

    def _widget(self) -> _VideoShmWidget | None:
        return self.widget_by_name(_WIDGET_NAME, _VideoShmWidget)

    def apply_ui_command(self, cmd: UiCommand) -> None:
        c = str(cmd.command or "")
        if c == "viz.video.detach":
            widget = self._widget()
            if widget is not None:
                widget.detach()
            return

        if c != "viz.video.set":
            return
        try:
            payload = dict(cmd.payload or {})
            shm_name = str(payload.get("shmName") or "").strip()
            throttle_ms = int(payload.get("throttleMs") or 33)
            flow_shm_name = str(payload.get("flowShmName") or "").strip()
            flow_display_mode = str(payload.get("flowDisplayMode") or "off").strip().lower()
            flow_mag_scale = float(payload.get("flowMagScale") or 20.0)
            flow_stride = int(payload.get("flowStride") or 12)
            scale_mode = str(payload.get("scaleMode") or "native").strip().lower()
        except (AttributeError, TypeError, ValueError):
            return
        widget = self._widget()
        if widget is None:
            return
        widget.set_config(
            shm_name=shm_name,
            throttle_ms=throttle_ms,
            flow_shm_name=flow_shm_name,
            flow_display_mode=flow_display_mode,
            flow_mag_scale=flow_mag_scale,
            flow_stride=flow_stride,
            scale_mode=scale_mode,
        )
