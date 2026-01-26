from __future__ import annotations

import time
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets
from NodeGraphQt.nodes.base_node import NodeBaseWidget

from f8pysdk.shm import VideoShmReader

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode


class _VideoShmPane(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._title = QtWidgets.QLabel("VideoSHM")
        self._title.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self._title.setStyleSheet("color: rgb(225, 225, 225);")

        self._image = QtWidgets.QLabel()
        self._image.setAlignment(QtCore.Qt.AlignCenter)
        self._image.setMinimumSize(240, 160)
        self._image.setStyleSheet("background: rgba(0,0,0,60); border: 1px solid rgba(255,255,255,25);")

        self._status = QtWidgets.QLabel("")
        self._status.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self._status.setStyleSheet("color: rgb(160, 160, 160);")

        layout.addWidget(self._title)
        layout.addWidget(self._image, 1)
        layout.addWidget(self._status)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)  # type: ignore[attr-defined]
        self._timer.setInterval(33)

        self._reader: VideoShmReader | None = None
        self._shm_name = ""
        self._last_frame_id = 0
        self._last_frame_bytes: bytes | None = None
        self._last_pixmap_size: QtCore.QSize | None = None

    def set_config(self, *, shm_name: str, throttle_ms: int) -> None:
        shm_name = str(shm_name or "").strip()
        throttle_ms = max(0, int(throttle_ms))
        self._timer.setInterval(max(1, throttle_ms) if throttle_ms > 0 else 1)
        if shm_name != self._shm_name:
            self._shm_name = shm_name
            self._reset_reader()
        if not self._timer.isActive():
            self._timer.start()

    def detach(self) -> None:
        try:
            self._timer.stop()
        except Exception:
            pass
        self._reset_reader()

    def _reset_reader(self) -> None:
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:
            pass
        self._reader = None
        self._last_frame_id = 0
        self._last_frame_bytes = None
        self._image.clear()
        self._status.setText("")

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
        except Exception:
            return
        pix = QtGui.QPixmap.fromImage(img)

        target = self._image.size()
        if self._last_pixmap_size != target:
            self._last_pixmap_size = target
        try:
            pix2 = pix.scaled(target, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        except Exception:
            pix2 = pix
        self._image.setPixmap(pix2)
        self._title.setText(f"VideoSHM  {w}x{h}  frameId={self._last_frame_id}")
        self._status.setText(f"{self._shm_name}  ts={int(header.ts_ms)}")


class _VideoShmWidget(NodeBaseWidget):
    def __init__(self, parent=None, name: str = "__videoshm", label: str = "") -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _VideoShmPane()
        self.set_custom_widget(self._pane)

    def get_value(self) -> object:
        return {}

    def set_value(self, value: object) -> None:
        return

    def set_config(self, *, shm_name: str, throttle_ms: int) -> None:
        self._pane.set_config(shm_name=shm_name, throttle_ms=throttle_ms)

    def detach(self) -> None:
        self._pane.detach()


class PyStudioVideoShmNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.video_shm_view`.
    """

    def __init__(self):
        super().__init__()
        try:
            self.add_custom_widget(_VideoShmWidget(self.view, name="__videoshm", label=""))
        except Exception:
            pass

    def apply_ui_command(self, cmd: Any) -> None:
        try:
            c = str(getattr(cmd, "command", "") or "")
        except Exception:
            return
        if c == "videoshm.detach":
            try:
                w = self.get_widget("__videoshm")
                if w and hasattr(w, "detach"):
                    w.detach()
            except Exception:
                pass
            return

        if c != "videoshm.set":
            return
        try:
            payload = getattr(cmd, "payload", {}) or {}
            shm_name = str(payload.get("shmName") or "").strip()
            throttle_ms = int(payload.get("throttleMs") or 33)
        except Exception:
            return
        try:
            w = self.get_widget("__videoshm")
            if w and hasattr(w, "set_config"):
                w.set_config(shm_name=shm_name, throttle_ms=throttle_ms)
        except Exception:
            return

