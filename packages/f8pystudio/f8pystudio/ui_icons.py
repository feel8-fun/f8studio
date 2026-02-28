from __future__ import annotations

from enum import Enum
from pathlib import Path

from qtpy import QtCore, QtGui, QtSvg, QtWidgets


class StudioIcon(Enum):
    FOLDER_OPEN = "folder-open.svg"
    FOLDER_PLUS = "folder-plus.svg"
    SAVE = "save.svg"
    SEND = "send.svg"
    STOP_ALL = "stop-all.svg"
    STOP = "stop.svg"
    PLAY = "play.svg"
    PAUSE = "pause.svg"
    REFRESH = "refresh.svg"
    REDO = "redo.svg"
    TOGGLE_ON = "toggle-on.svg"
    TOGGLE_OFF = "toggle-off.svg"
    CODE = "code.svg"
    EYE = "eye.svg"
    EYE_SLASH = "eye-slash.svg"


def icon_for(widget: QtWidgets.QWidget, token: StudioIcon) -> QtGui.QIcon:
    icon = _render_svg_icon(widget, token, token.value)
    if icon.isNull():
        raise RuntimeError(f"icon renderer returned null icon token={token.name}")
    return icon


def _icons_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "icons"


def _render_svg_icon(widget: QtWidgets.QWidget, token: StudioIcon, icon_name: str) -> QtGui.QIcon:
    size = _icon_size(widget)
    color = _icon_color(widget, token)
    icon_path = _icons_dir() / icon_name
    try:
        svg_source = icon_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"failed to read icon svg file={icon_path}") from exc
    svg_colored = svg_source.replace("currentColor", color.name())
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg_colored.encode("utf-8")))
    if not renderer.isValid():
        raise RuntimeError(f"invalid icon svg file={icon_path}")
    pixmap = QtGui.QPixmap(size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
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


_ICON_COLORS: dict[StudioIcon, QtGui.QColor] = {
    StudioIcon.PLAY: QtGui.QColor("#4CAF50"),
    StudioIcon.PAUSE: QtGui.QColor("#E6C229"),
    StudioIcon.STOP: QtGui.QColor("#E05252"),
    StudioIcon.STOP_ALL: QtGui.QColor("#E05252"),
    StudioIcon.SEND: QtGui.QColor("#7EDB8A"),
}
