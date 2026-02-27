from __future__ import annotations

from enum import Enum
from typing import Callable

from qtpy import QtCore, QtGui, QtWidgets


class StudioIcon(Enum):
    FOLDER_OPEN = "fa5s.folder-open"
    SAVE = "fa5s.save"
    SEND = "mdi6.send"
    STOP_ALL = "mdi.stop"
    STOP = "fa5s.stop"
    PLAY = "fa5s.play"
    PAUSE = "fa5s.pause"
    REFRESH = "fa5s.exchange-alt"
    REDO = "fa5s.redo"
    TOGGLE_ON = "fa5s.toggle-on"
    TOGGLE_OFF = "fa5s.toggle-off"
    CODE = "fa5s.code"
    EYE = "fa5s.eye"
    EYE_SLASH = "fa5s.eye-slash"


_DrawFn = Callable[[QtGui.QPainter, QtCore.QSize, QtGui.QColor], None]


def icon_for(widget: QtWidgets.QWidget, token: StudioIcon) -> QtGui.QIcon:
    drawer = _ICON_DRAWERS[token]
    icon = _render_icon(widget, token, drawer)
    if icon.isNull():
        raise RuntimeError(f"icon renderer returned null icon token={token.value}")
    return icon


def _render_icon(widget: QtWidgets.QWidget, token: StudioIcon, drawer: _DrawFn) -> QtGui.QIcon:
    size = _icon_size(widget)
    color = _icon_color(widget, token)
    pixmap = QtGui.QPixmap(size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    drawer(painter, size, color)
    painter.end()
    return QtGui.QIcon(pixmap)


def _icon_size(widget: QtWidgets.QWidget) -> QtCore.QSize:
    if isinstance(widget, QtWidgets.QAbstractButton):
        size = widget.iconSize()
        if size.width() > 0 and size.height() > 0:
            return size
    return QtCore.QSize(16, 16)


def _icon_color(widget: QtWidgets.QWidget, token: StudioIcon) -> QtGui.QColor:
    fixed = _ICON_COLORS.get(token)
    if fixed is not None:
        return QtGui.QColor(fixed)
    color = widget.palette().color(QtGui.QPalette.ColorRole.ButtonText)
    if not color.isValid():
        return QtGui.QColor(235, 235, 235)
    return color


def _stroke(painter: QtGui.QPainter, color: QtGui.QColor, width: float = 1.8) -> None:
    pen = QtGui.QPen(color, width, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)


def _fill(painter: QtGui.QPainter, color: QtGui.QColor) -> None:
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)


def _draw_folder_open(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    _stroke(painter, color, 1.6)
    top = QtCore.QRectF(2.0, h * 0.22, w * 0.38, h * 0.2)
    body = QtCore.QRectF(2.0, h * 0.35, w - 4.0, h * 0.45)
    painter.drawRoundedRect(top, 1.5, 1.5)
    painter.drawRoundedRect(body, 2.0, 2.0)
    _fill(painter, color)
    arrow = QtGui.QPolygonF(
        [
            QtCore.QPointF(w * 0.56, h * 0.53),
            QtCore.QPointF(w * 0.78, h * 0.53),
            QtCore.QPointF(w * 0.78, h * 0.46),
            QtCore.QPointF(w * 0.92, h * 0.60),
            QtCore.QPointF(w * 0.78, h * 0.74),
            QtCore.QPointF(w * 0.78, h * 0.67),
            QtCore.QPointF(w * 0.56, h * 0.67),
        ]
    )
    painter.drawPolygon(arrow)


def _draw_save(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    _stroke(painter, color, 1.6)
    body = QtCore.QRectF(2.0, 2.0, w - 4.0, h - 4.0)
    painter.drawRoundedRect(body, 2.0, 2.0)
    painter.drawLine(QtCore.QPointF(w * 0.25, h * 0.45), QtCore.QPointF(w * 0.75, h * 0.45))
    painter.drawLine(QtCore.QPointF(w * 0.32, h * 0.67), QtCore.QPointF(w * 0.68, h * 0.67))
    _fill(painter, color)
    notch = QtCore.QRectF(w * 0.62, h * 0.18, w * 0.16, h * 0.2)
    painter.drawRect(notch)


def _draw_send(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    _fill(painter, color)
    tri = QtGui.QPolygonF(
        [
            QtCore.QPointF(w * 0.16, h * 0.2),
            QtCore.QPointF(w * 0.88, h * 0.5),
            QtCore.QPointF(w * 0.16, h * 0.8),
            QtCore.QPointF(w * 0.32, h * 0.54),
            QtCore.QPointF(w * 0.58, h * 0.5),
            QtCore.QPointF(w * 0.32, h * 0.46),
        ]
    )
    painter.drawPolygon(tri)


def _draw_stop(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    _fill(painter, color)
    rect = QtCore.QRectF(w * 0.25, h * 0.25, w * 0.5, h * 0.5)
    painter.drawRoundedRect(rect, 1.8, 1.8)


def _draw_play(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    _fill(painter, color)
    tri = QtGui.QPolygonF(
        [
            QtCore.QPointF(w * 0.32, h * 0.22),
            QtCore.QPointF(w * 0.78, h * 0.5),
            QtCore.QPointF(w * 0.32, h * 0.78),
        ]
    )
    painter.drawPolygon(tri)


def _draw_pause(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    _fill(painter, color)
    left = QtCore.QRectF(w * 0.28, h * 0.22, w * 0.16, h * 0.56)
    right = QtCore.QRectF(w * 0.56, h * 0.22, w * 0.16, h * 0.56)
    painter.drawRoundedRect(left, 1.2, 1.2)
    painter.drawRoundedRect(right, 1.2, 1.2)


def _draw_upload_refresh(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    # Used for deploy/sync: tray + upward arrow (distinct from restart).
    w = float(size.width())
    h = float(size.height())
    _stroke(painter, color, 1.8)
    painter.drawLine(QtCore.QPointF(w * 0.2, h * 0.74), QtCore.QPointF(w * 0.8, h * 0.74))
    painter.drawLine(QtCore.QPointF(w * 0.2, h * 0.74), QtCore.QPointF(w * 0.2, h * 0.62))
    painter.drawLine(QtCore.QPointF(w * 0.8, h * 0.74), QtCore.QPointF(w * 0.8, h * 0.62))
    painter.drawLine(QtCore.QPointF(w * 0.5, h * 0.72), QtCore.QPointF(w * 0.5, h * 0.28))
    _fill(painter, color)
    head = QtGui.QPolygonF(
        [
            QtCore.QPointF(w * 0.5, h * 0.14),
            QtCore.QPointF(w * 0.34, h * 0.36),
            QtCore.QPointF(w * 0.66, h * 0.36),
        ]
    )
    painter.drawPolygon(head)


def _draw_redo(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    _stroke(painter, color, 1.8)
    arc = QtCore.QRectF(w * 0.18, h * 0.18, w * 0.64, h * 0.64)
    painter.drawArc(arc, 40 * 16, 260 * 16)
    _fill(painter, color)
    head = QtGui.QPolygonF(
        [
            QtCore.QPointF(w * 0.84, h * 0.28),
            QtCore.QPointF(w * 0.63, h * 0.26),
            QtCore.QPointF(w * 0.74, h * 0.45),
        ]
    )
    painter.drawPolygon(head)


def _draw_toggle_on(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    track = QtCore.QRectF(w * 0.1, h * 0.3, w * 0.8, h * 0.4)
    _stroke(painter, color, 1.5)
    painter.drawRoundedRect(track, h * 0.2, h * 0.2)
    _fill(painter, color)
    knob = QtCore.QRectF(w * 0.56, h * 0.24, h * 0.52, h * 0.52)
    painter.drawEllipse(knob)


def _draw_toggle_off(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    track = QtCore.QRectF(w * 0.1, h * 0.3, w * 0.8, h * 0.4)
    _stroke(painter, color, 1.5)
    painter.drawRoundedRect(track, h * 0.2, h * 0.2)
    _fill(painter, color)
    knob = QtCore.QRectF(w * 0.24, h * 0.24, h * 0.52, h * 0.52)
    painter.drawEllipse(knob)


def _draw_code(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    _stroke(painter, color, 1.9)
    painter.drawLine(QtCore.QPointF(w * 0.38, h * 0.24), QtCore.QPointF(w * 0.52, h * 0.76))
    painter.drawLine(QtCore.QPointF(w * 0.28, h * 0.50), QtCore.QPointF(w * 0.14, h * 0.38))
    painter.drawLine(QtCore.QPointF(w * 0.14, h * 0.38), QtCore.QPointF(w * 0.28, h * 0.26))
    painter.drawLine(QtCore.QPointF(w * 0.62, h * 0.24), QtCore.QPointF(w * 0.86, h * 0.50))
    painter.drawLine(QtCore.QPointF(w * 0.86, h * 0.50), QtCore.QPointF(w * 0.62, h * 0.76))


def _draw_eye(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    w = float(size.width())
    h = float(size.height())
    _stroke(painter, color, 1.6)
    eye = QtCore.QRectF(w * 0.12, h * 0.28, w * 0.76, h * 0.44)
    painter.drawEllipse(eye)
    _fill(painter, color)
    pupil = QtCore.QRectF(w * 0.42, h * 0.4, w * 0.16, h * 0.16)
    painter.drawEllipse(pupil)


def _draw_eye_slash(painter: QtGui.QPainter, size: QtCore.QSize, color: QtGui.QColor) -> None:
    _draw_eye(painter, size, color)
    w = float(size.width())
    h = float(size.height())
    _stroke(painter, color, 1.9)
    painter.drawLine(QtCore.QPointF(w * 0.18, h * 0.8), QtCore.QPointF(w * 0.82, h * 0.2))


_ICON_DRAWERS: dict[StudioIcon, _DrawFn] = {
    StudioIcon.FOLDER_OPEN: _draw_folder_open,
    StudioIcon.SAVE: _draw_save,
    StudioIcon.SEND: _draw_send,
    StudioIcon.STOP_ALL: _draw_stop,
    StudioIcon.STOP: _draw_stop,
    StudioIcon.PLAY: _draw_play,
    StudioIcon.PAUSE: _draw_pause,
    StudioIcon.REFRESH: _draw_upload_refresh,
    StudioIcon.REDO: _draw_redo,
    StudioIcon.TOGGLE_ON: _draw_toggle_on,
    StudioIcon.TOGGLE_OFF: _draw_toggle_off,
    StudioIcon.CODE: _draw_code,
    StudioIcon.EYE: _draw_eye,
    StudioIcon.EYE_SLASH: _draw_eye_slash,
}


_ICON_COLORS: dict[StudioIcon, QtGui.QColor] = {
    StudioIcon.PLAY: QtGui.QColor("#4CAF50"),
    StudioIcon.PAUSE: QtGui.QColor("#E6C229"),
    StudioIcon.STOP: QtGui.QColor("#E05252"),
    StudioIcon.STOP_ALL: QtGui.QColor("#E05252"),
    StudioIcon.SEND: QtGui.QColor("#7EDB8A"),
}
