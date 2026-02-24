from __future__ import annotations

import time
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets
from NodeGraphQt.nodes.base_node import NodeBaseWidget

from f8pysdk.shm import VideoShmReader

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from ..nodegraph.viz_operator_nodeitem import F8StudioVizOperatorNodeItem
from ..ui_bus import UiCommand

_STATE_UI_UPDATE = "uiUpdate"
_WIDGET_NAME = "__videoshm"


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
        # Avoid forcing a wide minimum: keep width shrinkable inside narrow nodes.
        self._image.setMinimumSize(0, 120)
        self._image.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._image.setStyleSheet("background: rgba(0,0,0,60); border: 1px solid rgba(255,255,255,25);")

        layout.addLayout(top)
        layout.addWidget(self._image, 1)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)  # type: ignore[attr-defined]
        self._timer.setInterval(33)

        self._reader: VideoShmReader | None = None
        self._shm_name = ""
        self._last_frame_id = 0
        self._last_frame_bytes: bytes | None = None
        self._last_pixmap_size: QtCore.QSize | None = None

    def update_checkbox(self) -> QtWidgets.QCheckBox:
        return self._update

    def update_enabled(self) -> bool:
        return bool(self._update.isChecked())

    def set_update_enabled(self, enabled: bool) -> None:
        self._update.setChecked(bool(enabled))
        self._sync_timer_with_update_state()

    def set_config(self, *, shm_name: str, throttle_ms: int) -> None:
        shm_name = str(shm_name or "").strip()
        throttle_ms = max(0, int(throttle_ms))
        self._timer.setInterval(max(1, throttle_ms) if throttle_ms > 0 else 1)
        if shm_name != self._shm_name:
            self._shm_name = shm_name
            self._reset_reader()
        self._sync_timer_with_update_state()

    def detach(self) -> None:
        try:
            self._timer.stop()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._reset_reader()

    def _reset_reader(self) -> None:
        try:
            if self._reader is not None:
                self._reader.close()
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._reader = None
        self._last_frame_id = 0
        self._last_frame_bytes = None
        self._image.clear()
        self._status.setText("")

    def _sync_timer_with_update_state(self) -> None:
        if self.update_enabled() and self._shm_name:
            if not self._timer.isActive():
                self._timer.start()
            return
        if self._timer.isActive():
            self._timer.stop()

    def _ensure_reader(self) -> bool:
        if self._reader is not None:
            return True
        if not self._shm_name:
            self._status.setText("missing shmName")
            return False
        try:
            r = VideoShmReader(self._shm_name)
            r.open(use_event=False)
            self._reader = r
            self._status.setText(f"{self._shm_name}")
            return True
        except Exception as exc:
            self._status.setText(f"open failed: {exc}")
            self._reader = None
            return False

    def _tick(self) -> None:
        if not self.update_enabled():
            return
        if not self._ensure_reader():
            return
        assert self._reader is not None
        try:
            header, payload = self._reader.read_latest_bgra()
        except Exception as exc:
            self._status.setText(f"read failed: {exc}")
            return
        if header is None or payload is None:
            return
        if int(header.frame_id) == int(self._last_frame_id):
            return
        self._last_frame_id = int(header.frame_id)

        frame_bytes = bytes(payload)
        self._last_frame_bytes = frame_bytes
        w = int(header.width)
        h = int(header.height)
        pitch = int(header.pitch)
        if w <= 0 or h <= 0 or pitch <= 0:
            return

        try:
            img = QtGui.QImage(frame_bytes, w, h, pitch, QtGui.QImage.Format_ARGB32)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return
        pix = QtGui.QPixmap.fromImage(img)

        target = self._image.size()
        if self._last_pixmap_size != target:
            self._last_pixmap_size = target
        try:
            pix2 = pix.scaled(target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        except (AttributeError, RuntimeError, TypeError):
            pix2 = pix
        self._image.setPixmap(pix2)
        # self._title.setText(f"VideoSHM  {w}x{h}  frameId={self._last_frame_id}")
        # self._status.setText(f"{self._shm_name}  ts={int(header.ts_ms)}")


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

    def set_config(self, *, shm_name: str, throttle_ms: int) -> None:
        self._pane.set_config(shm_name=shm_name, throttle_ms=throttle_ms)

    def detach(self) -> None:
        self._pane.detach()


class VizVideoRenderNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.viz.video`.
    """

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
        except (AttributeError, TypeError, ValueError):
            return
        widget = self._widget()
        if widget is None:
            return
        widget.set_config(shm_name=shm_name, throttle_ms=throttle_ms)
