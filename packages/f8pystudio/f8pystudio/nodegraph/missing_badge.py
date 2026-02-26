from __future__ import annotations

import logging

from qtpy import QtCore, QtGui, QtWidgets


logger = logging.getLogger(__name__)


class MissingBadgeMixin:
    _missing_badge_item: QtWidgets.QGraphicsPixmapItem
    _missing_badge_icon_error_logged: bool

    def _init_missing_badge(self) -> None:
        self._missing_badge_item = QtWidgets.QGraphicsPixmapItem(self)
        self._missing_badge_item.setVisible(False)
        self._missing_badge_item.setZValue(20_000)
        self._missing_badge_icon_error_logged = False

    @staticmethod
    def _missing_badge_pixmap() -> QtGui.QPixmap:
        size = 14
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pm)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(232, 64, 64))
            painter.drawEllipse(0, 0, size - 1, size - 1)
            painter.setBrush(QtGui.QColor(255, 255, 255))
            painter.drawRoundedRect(6, 3, 2, 6, 1, 1)
            painter.drawEllipse(6, 10, 2, 2)
        finally:
            painter.end()
        return pm

    def _missing_badge_info(self) -> tuple[bool, str]:
        try:
            node = self._backend_node()
        except Exception:
            return False, ""
        if node is None:
            return False, ""
        try:
            missing_locked = bool(node.is_missing_locked())
        except Exception:
            return False, ""
        if not missing_locked:
            return False, ""
        try:
            missing_type = str(node.missing_type() or "").strip()
        except Exception:
            missing_type = ""
        if missing_type:
            return True, f"missing {missing_type}"
        return True, "missing dependency"

    def _refresh_missing_badge(self) -> None:
        show_badge, tooltip = self._missing_badge_info()
        if not show_badge:
            self._missing_badge_item.setVisible(False)
            self._missing_badge_item.setToolTip("")
            return

        if self._missing_badge_item.pixmap().isNull():
            try:
                self._missing_badge_item.setPixmap(self._missing_badge_pixmap())
                self._missing_badge_icon_error_logged = False
            except Exception:
                if not self._missing_badge_icon_error_logged:
                    logger.exception("Failed to build missing badge icon.")
                    self._missing_badge_icon_error_logged = True
                self._missing_badge_item.setVisible(False)
                return

        self._missing_badge_item.setToolTip(tooltip)
        self._missing_badge_item.setVisible(True)
        rect = self.boundingRect()
        self._missing_badge_item.setPos(rect.left() + 4.0, rect.top() + 4.0)
