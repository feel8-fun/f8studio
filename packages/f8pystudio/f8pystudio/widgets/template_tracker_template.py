from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=False)


@dataclass(frozen=True)
class CaptureFrame:
    frame_id: int
    ts_ms: int
    image_bytes: bytes
    image_format: str
    width: int
    height: int


class RoiSelectLabel(QtWidgets.QLabel):
    """
    Simple ROI selection widget for a still image.

    Coordinates are returned in original image pixel space.
    """

    roi_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setCursor(QtCore.Qt.CrossCursor)

        self._orig_pix: QtGui.QPixmap | None = None
        self._scaled_pix: QtGui.QPixmap | None = None
        self._scaled_off = QtCore.QPoint(0, 0)
        self._scale = 1.0

        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._drag_start: QtCore.QPoint | None = None
        self._roi_img: QtCore.QRect | None = None

    def set_image_bytes(self, data: bytes) -> None:
        pix = QtGui.QPixmap()
        if not pix.loadFromData(data):
            self._orig_pix = None
            self._scaled_pix = None
            self._roi_img = None
            self.setText("Failed to decode image")
            self.roi_changed.emit()
            return
        self._orig_pix = pix
        self._roi_img = None
        self._update_scaled_pix()
        self.roi_changed.emit()

    def roi_image_rect(self) -> QtCore.QRect | None:
        return QtCore.QRect(self._roi_img) if self._roi_img is not None else None

    def clear_roi(self) -> None:
        self._roi_img = None
        self._rubber.hide()
        self.roi_changed.emit()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_scaled_pix()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.LeftButton or self._orig_pix is None or self._scaled_pix is None:
            return super().mousePressEvent(event)
        self._drag_start = event.position().toPoint()
        self._rubber.setGeometry(QtCore.QRect(self._drag_start, QtCore.QSize()))
        self._rubber.show()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_start is None or self._orig_pix is None or self._scaled_pix is None:
            return super().mouseMoveEvent(event)
        p = event.position().toPoint()
        self._rubber.setGeometry(QtCore.QRect(self._drag_start, p).normalized())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.LeftButton or self._drag_start is None or self._orig_pix is None or self._scaled_pix is None:
            return super().mouseReleaseEvent(event)
        end = event.position().toPoint()
        rect = QtCore.QRect(self._drag_start, end).normalized()
        self._drag_start = None

        img_rect = self._widget_rect_to_image_rect(rect)
        if img_rect is None or img_rect.width() < 2 or img_rect.height() < 2:
            self.clear_roi()
            return
        self._roi_img = img_rect
        self.roi_changed.emit()

    def _update_scaled_pix(self) -> None:
        if self._orig_pix is None:
            self._scaled_pix = None
            self._scale = 1.0
            self._scaled_off = QtCore.QPoint(0, 0)
            self.setText("No image")
            return

        w = max(1, int(self.width()))
        h = max(1, int(self.height()))
        scaled = self._orig_pix.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._scaled_pix = scaled
        self.setPixmap(scaled)

        sw = int(scaled.width())
        sh = int(scaled.height())
        self._scaled_off = QtCore.QPoint(max(0, (w - sw) // 2), max(0, (h - sh) // 2))
        ow = int(self._orig_pix.width())
        self._scale = float(sw) / float(max(1, ow))

    def _widget_rect_to_image_rect(self, r: QtCore.QRect) -> QtCore.QRect | None:
        if self._orig_pix is None or self._scaled_pix is None:
            return None
        off = self._scaled_off
        sw = int(self._scaled_pix.width())
        sh = int(self._scaled_pix.height())
        src_w = int(self._orig_pix.width())
        src_h = int(self._orig_pix.height())

        # clamp to scaled image region
        x1 = max(off.x(), min(off.x() + sw, r.left()))
        y1 = max(off.y(), min(off.y() + sh, r.top()))
        x2 = max(off.x(), min(off.x() + sw, r.right()))
        y2 = max(off.y(), min(off.y() + sh, r.bottom()))
        if x2 <= x1 or y2 <= y1:
            return None

        rx1 = int(round((x1 - off.x()) / max(1e-6, self._scale)))
        ry1 = int(round((y1 - off.y()) / max(1e-6, self._scale)))
        rx2 = int(round((x2 - off.x()) / max(1e-6, self._scale)))
        ry2 = int(round((y2 - off.y()) / max(1e-6, self._scale)))

        rx1 = max(0, min(src_w - 1, rx1))
        ry1 = max(0, min(src_h - 1, ry1))
        rx2 = max(0, min(src_w, rx2))
        ry2 = max(0, min(src_h, ry2))
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        return QtCore.QRect(rx1, ry1, rx2 - rx1, ry2 - ry1)


def _encode_png_b64(img: QtGui.QImage, *, max_b64_bytes: int = 900_000) -> tuple[str, dict[str, Any]]:
    """
    Encode a QImage as PNG base64, resizing down if needed for NATS payload size.
    """
    if img.isNull():
        raise ValueError("empty image")

    cur = img
    for _ in range(12):
        buf = QtCore.QBuffer()
        buf.open(QtCore.QIODevice.WriteOnly)
        ok = cur.save(buf, "PNG")
        if not ok:
            raise RuntimeError("failed to encode PNG")
        raw = bytes(buf.data())
        b64 = _b64encode(raw)
        if len(b64) <= int(max_b64_bytes):
            return b64, {"format": "png", "width": int(cur.width()), "height": int(cur.height()), "bytes": int(len(raw)), "b64Bytes": int(len(b64))}

        w = int(cur.width())
        h = int(cur.height())
        if w <= 32 or h <= 32:
            break
        cur = cur.scaled(max(32, int(w * 0.85)), max(32, int(h * 0.85)), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    raise ValueError("encoded template too large for max_b64_bytes")


class TemplateCaptureDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget | None,
        bridge: Any,
        service_id: str,
        request_capture: Callable[[Callable[[CaptureFrame | None, str | None], None]], None],
        set_template_b64: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Template Tracker - Set Template")
        self.setModal(True)
        self.resize(980, 640)

        self._bridge = bridge
        self._service_id = str(service_id)
        self._request_capture = request_capture
        self._set_template_b64 = set_template_b64

        self._capture: CaptureFrame | None = None

        self._img = RoiSelectLabel()
        self._img.setText("Click Capture to fetch a frame")
        self._img.roi_changed.connect(self._update_buttons)  # type: ignore[attr-defined]

        self._btn_capture = QtWidgets.QPushButton("Capture")
        self._btn_capture.clicked.connect(self._do_capture)  # type: ignore[attr-defined]

        self._btn_clear = QtWidgets.QPushButton("Clear ROI")
        self._btn_clear.clicked.connect(self._img.clear_roi)  # type: ignore[attr-defined]

        self._btn_apply = QtWidgets.QPushButton("Apply Template")
        self._btn_apply.clicked.connect(self._apply_template)  # type: ignore[attr-defined]

        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: rgb(210,210,210);")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self._btn_capture)
        top.addWidget(self._btn_clear)
        top.addStretch(1)
        top.addWidget(self._btn_apply)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._img, 1)
        layout.addWidget(self._status)

        self._update_buttons()
        # Auto-capture once on open to reduce friction.
        try:
            QtCore.QTimer.singleShot(0, self._do_capture)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _set_status(self, msg: str) -> None:
        self._status.setText(str(msg or ""))

    def _update_buttons(self) -> None:
        has_img = self._capture is not None
        roi = self._img.roi_image_rect()
        has_roi = roi is not None and roi.width() >= 2 and roi.height() >= 2
        self._btn_clear.setEnabled(bool(has_roi))
        self._btn_apply.setEnabled(bool(has_img and has_roi))

    def _do_capture(self) -> None:
        self._set_status("Capturing...")
        self._btn_capture.setEnabled(False)
        self._btn_apply.setEnabled(False)

        def _done(cap: CaptureFrame | None, err: str | None) -> None:
            self._btn_capture.setEnabled(True)
            if err:
                self._set_status(f"Capture failed: {err}")
                self._capture = None
                self._img.setText("Capture failed")
                self._img.clear_roi()
                self._update_buttons()
                return
            if cap is None:
                self._set_status("Capture failed: empty response")
                self._capture = None
                self._update_buttons()
                return
            self._capture = cap
            self._img.set_image_bytes(cap.image_bytes)
            self._set_status(f"Captured frameId={cap.frame_id} ({cap.width}x{cap.height})")
            self._update_buttons()

        try:
            self._request_capture(_done)
        except Exception as exc:
            _done(None, str(exc))

    def _apply_template(self) -> None:
        if self._capture is None:
            return
        roi = self._img.roi_image_rect()
        if roi is None:
            return

        img = QtGui.QImage.fromData(self._capture.image_bytes)
        if img.isNull():
            self._set_status("Failed to decode captured image")
            return
        crop = img.copy(roi)
        try:
            b64, meta = _encode_png_b64(crop, max_b64_bytes=900_000)
        except Exception as exc:
            self._set_status(f"Template encode failed: {exc}")
            return

        try:
            self._set_template_b64(b64)
        except Exception as exc:
            self._set_status(f"set_state failed: {exc}")
            return

        self._set_status(f"Template applied ({meta.get('width')}x{meta.get('height')}, b64Bytes={meta.get('b64Bytes')})")
        self.accept()
