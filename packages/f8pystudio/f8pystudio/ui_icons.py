from __future__ import annotations

from enum import Enum
from pathlib import Path

from qtpy import QtCore, QtGui, QtSvg, QtWidgets


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


def icon_for(widget: QtWidgets.QWidget, token: StudioIcon) -> QtGui.QIcon:
    icon_name = _ICON_MAP[token]
    icon = _render_svg_icon(widget, token, icon_name)
    if icon.isNull():
        raise RuntimeError(f"icon renderer returned null icon token={token.value}")
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


_ICON_MAP: dict[StudioIcon, str] = {
    StudioIcon.FOLDER_OPEN: "folder-open.svg",
    StudioIcon.SAVE: "save.svg",
    StudioIcon.SEND: "send.svg",
    StudioIcon.STOP_ALL: "stop-all.svg",
    StudioIcon.STOP: "stop.svg",
    StudioIcon.PLAY: "play.svg",
    StudioIcon.PAUSE: "pause.svg",
    StudioIcon.REFRESH: "refresh.svg",
    StudioIcon.REDO: "redo.svg",
    StudioIcon.TOGGLE_ON: "toggle-on.svg",
    StudioIcon.TOGGLE_OFF: "toggle-off.svg",
    StudioIcon.CODE: "code.svg",
    StudioIcon.EYE: "eye.svg",
    StudioIcon.EYE_SLASH: "eye-slash.svg",
}
