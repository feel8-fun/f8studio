from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets

from ...ui.support.studio_theme import qss_rgba, studio_dark_theme
from ...ui.support.ui_control import parse_ui_control
from ...ui.support.ui_notifications import show_warning
from ...ui.support.qt_lifecycle import qt_object_is_valid as _qt_object_is_valid

_COMBO_REOPEN_GUARD_S = 0.05
_COMBO_POPUP_FLAGS = QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint
logger = logging.getLogger(__name__)


def _strip_data_url_prefix(b64: str) -> tuple[str, str | None]:
    """
    Accepts either raw base64 or a minimal data URL like:
      data:image/png;base64,....
    Returns (base64_payload, mime or None).
    """
    s = str(b64 or "").strip()
    if not s.startswith("data:"):
        return s, None
    m = re.match(r"^data:([^;]+);base64,(.*)$", s, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return s, None
    return m.group(2).strip(), m.group(1).strip()


def _b64decode_to_bytes(b64: str) -> bytes:
    payload, _mime = _strip_data_url_prefix(b64)
    if not payload:
        return b""
    return base64.b64decode(payload.encode("ascii"), validate=False)


def _b64encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _graphics_proxy_for_widget(widget: QtWidgets.QWidget) -> QtWidgets.QGraphicsProxyWidget | None:
    current: QtWidgets.QWidget | None = widget
    while current is not None:
        try:
            proxy = current.graphicsProxyWidget()
        except (RuntimeError, TypeError):
            proxy = None
        if proxy is not None:
            return proxy
        try:
            current = current.parentWidget()
        except (RuntimeError, TypeError):
            return None
    return None


def _window_for_graphics_proxy(proxy: QtWidgets.QGraphicsProxyWidget) -> QtWidgets.QWidget | None:
    try:
        scene = proxy.scene()
    except (RuntimeError, TypeError):
        return None
    if scene is None:
        return None
    try:
        views = list(scene.views() or [])
    except (RuntimeError, TypeError):
        return None
    if not views:
        return None
    visible_views: list[Any] = []
    for view in views:
        try:
            if bool(view.isVisible()):
                visible_views.append(view)
        except (RuntimeError, TypeError):
            continue
    view = visible_views[0] if visible_views else views[0]
    try:
        window = view.window()
    except (RuntimeError, TypeError):
        return None
    return window


def _resolve_embedded_dialog_parent(widget: QtWidgets.QWidget) -> QtWidgets.QWidget | None:
    proxy = _graphics_proxy_for_widget(widget)
    if proxy is not None:
        proxy_window = _window_for_graphics_proxy(proxy)
        if proxy_window is not None:
            return proxy_window
    try:
        widget_window = widget.window()
    except (RuntimeError, TypeError):
        widget_window = None
    if widget_window is not None:
        return widget_window
    try:
        return QtWidgets.QApplication.activeWindow()
    except (RuntimeError, TypeError):
        return None


def parse_select_pool(ui_control: str) -> str | None:
    """
    Parse uiControl patterns for option pools:
      "select[poolStateField]"
    """
    return parse_ui_control(ui_control).select_pool_field


def parse_multiselect_pool(ui_control: str) -> str | None:
    """
    Parse uiControl patterns for multi-select pools:
      "multiselect[poolStateField]"
    """
    return parse_ui_control(ui_control).multiselect_pool_field


def _popup_above_y(anchor_y: int, popup_h: int) -> int:
    return int(anchor_y) - max(0, int(popup_h))


def _combo_popup_debug_enabled() -> bool:
    raw = str(os.getenv("F8_DEBUG_COMBO_POPUP", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _choose_best_view_for_scene_point(views: list[Any], scene_pos: QtCore.QPointF) -> Any | None:
    visible_views: list[Any] = []
    for view in list(views or []):
        try:
            if bool(view.isVisible()):
                visible_views.append(view)
        except (AttributeError, RuntimeError, TypeError):
            continue
    if not visible_views:
        return None

    containing_views: list[Any] = []
    for view in visible_views:
        try:
            view_pos = view.mapFromScene(scene_pos)
            viewport = view.viewport()
            if viewport is not None and bool(viewport.rect().contains(view_pos)):
                containing_views.append(view)
        except (AttributeError, RuntimeError, TypeError):
            continue

    candidates = containing_views if containing_views else visible_views
    for view in candidates:
        try:
            if bool(view.hasFocus()):
                return view
        except (AttributeError, RuntimeError, TypeError):
            continue
    for view in candidates:
        try:
            if bool(view.isActiveWindow()):
                return view
        except (AttributeError, RuntimeError, TypeError):
            continue
    return candidates[0] if candidates else None


class _F8ComboPopup(QtWidgets.QFrame):
    valueSelected = QtCore.Signal(int)

    def __init__(self, parent_combo: "F8OptionCombo") -> None:
        super().__init__(None, _COMBO_POPUP_FLAGS)
        self._combo = parent_combo
        self._last_show_monotonic_s: float = 0.0
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        theme_palette = studio_dark_theme().palette
        self._bg_color = QtGui.QColor(theme_palette.panel_bg)
        self._border_color = QtGui.QColor(theme_palette.border)
        self._radius = 6.0
        self.setStyleSheet(
            f"""
            QListView {{
                background: transparent;
                color: {theme_palette.text_primary};
                selection-background-color: {theme_palette.selection_bg};
                outline: 0;
                border: 0px;
            }}
            """
        )

        self._view = QtWidgets.QListView(self)
        self._view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._view.setUniformItemSizes(True)
        self._view.clicked.connect(self._on_clicked)  # type: ignore[attr-defined]
        self._view.activated.connect(self._on_clicked)  # type: ignore[attr-defined]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._view)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        del event
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QtGui.QPen(self._border_color, 1.0))
        p.setBrush(self._bg_color)
        p.drawRoundedRect(rect, self._radius, self._radius)

    def set_model(self, model: QtCore.QAbstractItemModel) -> None:
        self._view.setModel(model)

    def set_current_index(self, index: int) -> None:
        model = self._view.model()
        if model is None or index < 0:
            return
        try:
            idx = model.index(index, 0)
        except (AttributeError, RuntimeError, TypeError):
            return
        self._view.setCurrentIndex(idx)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # type: ignore[override]
        super().hideEvent(event)
        if _combo_popup_debug_enabled():
            dt_ms = (time.monotonic() - self._last_show_monotonic_s) * 1000.0
            logger.warning(
                "combo_popup hide name=%s dtMs=%.1f visible=%s",
                self._combo.objectName() or "<unnamed>",
                dt_ms,
                bool(self.isVisible()),
            )
        self._combo._block_popup_for(_COMBO_REOPEN_GUARD_S)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in (QtCore.Qt.Key_Escape,):
            self.hide()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_clicked(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return
        self.valueSelected.emit(index.row())
        self.hide()


class F8OptionCombo(QtWidgets.QComboBox):
    """
    Combo box with value helpers and a top-level popup (avoids NodeGraphQt Z issues).
    """

    valueChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._values: list[Any] = []
        self._context_tooltip = ""
        self._read_only = False
        self._popup_block_until_s: float = 0.0
        self._pending_popup_show = False
        self.setEditable(False)
        self.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumHeight(22)
        self.setMaxVisibleItems(16)
        view = QtWidgets.QListView()
        view.setUniformItemSizes(True)
        self.setView(view)
        self.currentIndexChanged.connect(self._emit)  # type: ignore[attr-defined]
        self._popup: _F8ComboPopup | None = None

    def _on_popup_destroyed(self, _obj: Any) -> None:
        self._popup = None

    def _delete_popup_later_on_destroyed(self, _obj: Any = None) -> None:
        popup = self._popup
        self._popup = None
        if not _qt_object_is_valid(popup):
            return
        popup.deleteLater()

    def _ensure_popup(self) -> _F8ComboPopup:
        popup = self._popup
        if popup is not None:
            return popup
        popup = _F8ComboPopup(self)
        popup.valueSelected.connect(self._on_popup_selected)  # type: ignore[attr-defined]
        popup.destroyed.connect(self._on_popup_destroyed)  # type: ignore[attr-defined]
        self.destroyed.connect(self._delete_popup_later_on_destroyed)  # type: ignore[attr-defined]
        self._popup = popup
        return popup

    def _block_popup_for(self, seconds: float) -> None:
        until = time.monotonic() + max(0.0, float(seconds))
        self._popup_block_until_s = max(self._popup_block_until_s, until)

    def set_read_only(self, read_only: bool) -> None:
        """
        Read-only mode that keeps text selectable/copyable.

        Unlike disabling the widget, this allows users to select/copy the
        displayed value while preventing changes.
        """
        ro = bool(read_only)
        self._read_only = ro
        line_edit = self.lineEdit()
        if ro:
            self.setEditable(True)
            line_edit = self.lineEdit()
            if line_edit is not None:
                line_edit.setReadOnly(True)
        else:
            if line_edit is not None:
                line_edit.setReadOnly(False)
            self.setEditable(False)

    def set_context_tooltip(self, tooltip: str) -> None:
        self._context_tooltip = str(tooltip or "").strip()
        for i in range(self.count()):
            self.setItemData(i, self._item_tooltip(i), QtCore.Qt.ToolTipRole)

    def set_options(
        self,
        values: list[Any],
        *,
        labels: list[str] | None = None,
        tooltips: list[str] | None = None,
    ) -> None:
        cur = self.value()
        with QtCore.QSignalBlocker(self):
            self.clear()
            self._values = list(values)
            if labels is None:
                labels = [str(v) for v in values]
            labels = list(labels)
            if tooltips is None:
                tooltips = ["" for _ in values]
            tooltips = list(tooltips)
            for i, v in enumerate(values):
                label = labels[i] if i < len(labels) else str(v)
                self.addItem(str(label), v)
                tip = tooltips[i] if i < len(tooltips) else ""
                if self._context_tooltip or tip:
                    self.setItemData(i, self._item_tooltip(i, tip), QtCore.Qt.ToolTipRole)
        self.set_value(cur)

    def set_value(self, value: Any) -> None:
        if value is None:
            with QtCore.QSignalBlocker(self):
                self.setCurrentIndex(-1)
            return
        target = str(value)
        for i, v in enumerate(self._values):
            if str(v) == target:
                with QtCore.QSignalBlocker(self):
                    self.setCurrentIndex(i)
                return

    def value(self) -> Any:
        idx = self.currentIndex()
        if idx < 0:
            return None
        data = self.itemData(idx, QtCore.Qt.UserRole)
        return data if data is not None else self.currentText()

    def showPopup(self) -> None:  # type: ignore[override]
        if self._read_only:
            return
        if time.monotonic() < self._popup_block_until_s:
            return
        popup = self._ensure_popup()
        if popup.isVisible():
            # Toggle behavior: clicking the combobox again collapses the popup.
            self.hidePopup()
            return
        if not self.isEnabled():
            return
        model = self.model()
        if model is None or model.rowCount() == 0:
            return
        popup.set_model(model)
        popup.set_current_index(self.currentIndex())
        popup.resize(self._popup_size())
        anchor = self._anchor_global()
        pos = self._popup_pos(anchor, popup.height())
        if _combo_popup_debug_enabled():
            logger.warning(
                "combo_popup show name=%s anchor=%s target=%s popupSize=%sx%s parent=%s",
                self.objectName() or "<unnamed>",
                anchor,
                pos,
                popup.width(),
                popup.height(),
                type(popup.parentWidget()).__name__ if popup.parentWidget() is not None else "<none>",
            )
        self._pending_popup_show = True
        QtCore.QTimer.singleShot(0, lambda: self._show_popup_deferred(pos))

    def hidePopup(self) -> None:  # type: ignore[override]
        self._pending_popup_show = False
        popup = self._popup
        if popup is None:
            return
        try:
            visible = popup.isVisible()
        except RuntimeError:
            # Qt shutdown can delete popup before QComboBox hidePopup callback runs.
            self._popup = None
            return
        if visible:
            self._block_popup_for(_COMBO_REOPEN_GUARD_S)
            popup.hide()

    def _show_popup_deferred(self, pos: QtCore.QPoint) -> None:
        if not self._pending_popup_show:
            return
        self._pending_popup_show = False
        popup = self._popup
        if popup is None or not self.isVisible() or not self.isEnabled():
            return
        popup._last_show_monotonic_s = time.monotonic()
        popup.move(pos)
        popup.show()
        popup.raise_()
        popup.activateWindow()
        popup.setFocus(QtCore.Qt.PopupFocusReason)
        if _combo_popup_debug_enabled():
            screen = QtGui.QGuiApplication.screenAt(pos)
            logger.warning(
                "combo_popup shown name=%s visible=%s geo=%s screen=%s",
                self.objectName() or "<unnamed>",
                bool(popup.isVisible()),
                popup.geometry(),
                type(screen).__name__ if screen is not None else "<none>",
            )

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        if self._read_only:
            event.ignore()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if self._read_only:
            try:
                if event.key() in (
                    QtCore.Qt.Key.Key_Up,
                    QtCore.Qt.Key.Key_Down,
                    QtCore.Qt.Key.Key_PageUp,
                    QtCore.Qt.Key.Key_PageDown,
                    QtCore.Qt.Key.Key_Home,
                    QtCore.Qt.Key.Key_End,
                ):
                    event.ignore()
                    return
            except (AttributeError, RuntimeError, TypeError):
                pass
        super().keyPressEvent(event)

    def _popup_size(self) -> QtCore.QSize:
        model = self.model()
        rows = model.rowCount() if model is not None else 0
        visible = min(rows, max(1, self.maxVisibleItems()))
        row_h = self.view().sizeHintForRow(0)
        if row_h <= 0:
            row_h = self.fontMetrics().height() + 8
        height = visible * row_h + 12
        width = max(self.width(), self.view().sizeHintForColumn(0) + 20)
        return QtCore.QSize(width, height)

    def _popup_pos(self, anchor: QtCore.QPoint, popup_h: int) -> QtCore.QPoint:
        below = anchor
        above = QtCore.QPoint(anchor.x(), _popup_above_y(anchor.y(), popup_h))
        screen = QtGui.QGuiApplication.screenAt(below)
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            if _combo_popup_debug_enabled():
                logger.warning("combo_popup no_screen anchor=%s", anchor)
            return below
        geo = screen.availableGeometry()
        if below.y() + popup_h <= geo.bottom():
            return below
        if above.y() >= geo.top():
            return above
        return below

    def _anchor_global(self) -> QtCore.QPoint:
        try:
            proxy = None
            w: QtWidgets.QWidget | None = self
            while w is not None and proxy is None:
                proxy = w.graphicsProxyWidget()
                w = w.parentWidget()
            if proxy is not None:
                scene = proxy.scene()
                if scene is not None:
                    views = scene.views()
                    if views:
                        root = proxy.widget()
                        if root is not None:
                            local_pt = self.mapTo(root, self.rect().bottomLeft())
                            scene_pos = proxy.mapToScene(QtCore.QPointF(local_pt))
                        else:
                            scene_pos = proxy.mapToScene(QtCore.QPointF(self.rect().bottomLeft()))
                        view = _choose_best_view_for_scene_point(list(views), scene_pos)
                        if view is None:
                            return self.mapToGlobal(QtCore.QPoint(0, self.height()))
                        view_pt = view.mapFromScene(scene_pos)
                        return view.viewport().mapToGlobal(view_pt)
        except (AttributeError, RuntimeError, TypeError):
            pass
        return self.mapToGlobal(QtCore.QPoint(0, self.height()))

    def _on_popup_selected(self, row: int) -> None:
        if row < 0:
            return
        self.setCurrentIndex(row)
        self.hidePopup()

    def _item_tooltip(self, index: int, tooltip: str = "") -> str:
        label = self.itemText(index)
        base = str(tooltip or "").strip()
        if not base:
            base = label
        extra = []
        if self._context_tooltip:
            extra.append(self._context_tooltip)
        v = None
        if 0 <= index < len(self._values):
            v = self._values[index]
        if v is not None and str(v) != label:
            extra.append(f"Value: {v}")
        return "\n".join([base] + extra) if extra else base

    def _emit(self, _index: int) -> None:
        self.valueChanged.emit(self.value())


class F8Switch(QtWidgets.QAbstractButton):
    """
    Switch-style boolean toggle.
    """

    valueChanged = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._label_on = "True"
        self._label_off = "False"
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumHeight(22)
        self.toggled.connect(self._emit)  # type: ignore[attr-defined]

    def set_labels(self, on_label: str, off_label: str) -> None:
        self._label_on = str(on_label or "")
        self._label_off = str(off_label or "")
        self.update()

    def set_value(self, value: Any) -> None:
        with QtCore.QSignalBlocker(self):
            self.setChecked(bool(value))
        self.update()

    def value(self) -> bool:
        return bool(self.isChecked())

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return QtCore.QSize(72, 22)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        del event
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        track_h = min(18.0, rect.height())
        track_rect = QtCore.QRectF(
            rect.left(),
            rect.center().y() - track_h / 2.0,
            rect.width(),
            track_h,
        )
        radius = track_rect.height() / 2.0
        knob_d = max(10.0, track_rect.height() - 4.0)
        knob_y = track_rect.center().y() - knob_d / 2.0
        if self.isChecked():
            knob_x = track_rect.right() - knob_d - 2.0
        else:
            knob_x = track_rect.left() + 2.0

        enabled = self.isEnabled()
        border = QtGui.QColor(255, 255, 255, 70 if enabled else 35)
        bg = QtGui.QColor(0, 0, 0, 45 if enabled else 25)
        fill = QtGui.QColor(120, 200, 255, 80 if enabled else 35)
        knob = QtGui.QColor(235, 235, 235, 235 if enabled else 120)
        text = QtGui.QColor(235, 235, 235, 210 if enabled else 110)

        p.setPen(QtGui.QPen(border, 1.0))
        p.setBrush(fill if self.isChecked() else bg)
        p.drawRoundedRect(track_rect, radius, radius)

        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(knob)
        p.drawEllipse(QtCore.QRectF(knob_x, knob_y, knob_d, knob_d))

        p.setPen(text)
        label = self._label_on if self.isChecked() else self._label_off
        p.drawText(track_rect, QtCore.Qt.AlignCenter, label)

    def _emit(self, v: bool) -> None:
        self.valueChanged.emit(bool(v))


class F8ValueBar(QtWidgets.QWidget):
    """
    Click/drag to set a numeric value. Renders as a filled bar with centered text.
    """

    valueChanging = QtCore.Signal(object)  # float|int
    valueCommitted = QtCore.Signal(object)  # float|int

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: float | int | None = None,
        integer: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setMinimumHeight(22)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._min = float(minimum)
        self._max = float(maximum)
        if self._max < self._min:
            self._min, self._max = self._max, self._min
        self._integer = bool(integer)
        self._value: float | int = float(value) if value is not None else float(self._min)
        self._dragging = False
        self._read_only = False
        self._text_prefix = ""

    def set_text_prefix(self, prefix: str) -> None:
        self._text_prefix = str(prefix or "").strip()
        self.update()

    def set_read_only(self, read_only: bool) -> None:
        self._read_only = bool(read_only)
        self.setCursor(
            QtCore.Qt.ArrowCursor if self._read_only or not self.isEnabled() else QtCore.Qt.PointingHandCursor
        )
        self.update()

    def is_read_only(self) -> bool:
        return bool(self._read_only)

    def setEnabled(self, enabled: bool) -> None:  # type: ignore[override]
        super().setEnabled(bool(enabled))
        self.setCursor(
            QtCore.Qt.ArrowCursor if self._read_only or not self.isEnabled() else QtCore.Qt.PointingHandCursor
        )

    def set_range(self, minimum: float | int | None, maximum: float | int | None) -> None:
        lo = float(0.0 if minimum is None else minimum)
        hi = float(1.0 if maximum is None else maximum)
        if hi < lo:
            lo, hi = hi, lo
        self._min, self._max = lo, hi
        self.set_value(self._value)

    def set_value(self, value: Any) -> None:
        v = self._coerce(value)
        self._value = v
        self.update()

    def value(self) -> float | int:
        return self._value

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled() and not self._read_only:
            self._dragging = True
            self._set_from_pos(event.position().x(), commit=False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._dragging and self.isEnabled() and not self._read_only:
            self._set_from_pos(event.position().x(), commit=False)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._dragging and event.button() == QtCore.Qt.LeftButton and self.isEnabled() and not self._read_only:
            self._dragging = False
            self._set_from_pos(event.position().x(), commit=True)
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        del event
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = 4.0

        interactive = self.isEnabled() and not self._read_only
        border = QtGui.QColor(255, 255, 255, 55 if interactive else 25)
        bg = QtGui.QColor(0, 0, 0, 45)
        fill = QtGui.QColor(120, 200, 255, 70 if interactive else 30)

        p.setPen(QtGui.QPen(border, 1.0))
        p.setBrush(bg)
        p.drawRoundedRect(rect, radius, radius)

        frac = self._fraction()
        if frac > 0.0:
            fill_rect = QtCore.QRectF(rect)
            fill_rect.setWidth(rect.width() * frac)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(fill)
            p.drawRoundedRect(fill_rect, radius, radius)

        value_text = self._format_value(self._value)
        text = f"{self._text_prefix} {value_text}" if self._text_prefix else value_text
        p.setPen(QtGui.QColor(235, 235, 235, 255 if interactive else 120))
        p.drawText(rect, QtCore.Qt.AlignCenter, text)

    def _fraction(self) -> float:
        if self._max <= self._min:
            return 0.0
        v = float(self._value)
        return max(0.0, min(1.0, (v - self._min) / (self._max - self._min)))

    def _coerce(self, value: Any) -> float | int:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float(self._min)
        v = max(self._min, min(self._max, v))
        if self._integer:
            return int(round(v))
        return v

    def _format_value(self, v: float | int) -> str:
        if self._integer:
            return str(int(v))
        # keep it readable
        return ("{:.6f}".format(float(v))).rstrip("0").rstrip(".")

    def _set_from_pos(self, x: float, *, commit: bool) -> None:
        w = max(1.0, float(self.width()))
        frac = max(0.0, min(1.0, float(x) / w))
        v = self._min + frac * (self._max - self._min)
        v2 = self._coerce(v)
        if v2 == self._value and not commit:
            return
        self._value = v2
        self.update()
        if commit:
            self.valueCommitted.emit(v2)
        else:
            self.valueChanging.emit(v2)


class F8RangeBar(QtWidgets.QWidget):
    """Two compact value bars for editing a numeric ``[minimum, maximum]`` range."""

    valueChanging = QtCore.Signal(object)  # list[float|int]
    valueCommitted = QtCore.Signal(object)  # list[float|int]

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: Any = None,
        integer: bool = False,
    ) -> None:
        super().__init__(parent)
        self._integer = bool(integer)
        self._lower_bar = F8ValueBar(self, minimum=minimum, maximum=maximum, integer=integer)
        self._upper_bar = F8ValueBar(self, minimum=minimum, maximum=maximum, integer=integer)
        self._lower_bar.set_text_prefix("Min")
        self._upper_bar.set_text_prefix("Max")
        self._lower_bar.valueChanging.connect(lambda raw: self._on_bar_value(0, raw, commit=False))  # type: ignore[attr-defined]
        self._lower_bar.valueCommitted.connect(lambda raw: self._on_bar_value(0, raw, commit=True))  # type: ignore[attr-defined]
        self._upper_bar.valueChanging.connect(lambda raw: self._on_bar_value(1, raw, commit=False))  # type: ignore[attr-defined]
        self._upper_bar.valueCommitted.connect(lambda raw: self._on_bar_value(1, raw, commit=True))  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._lower_bar, 1)
        layout.addWidget(self._upper_bar, 1)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.set_value(value if value is not None else [minimum, maximum])

    def set_range(self, minimum: float | int | None, maximum: float | int | None) -> None:
        current = self.value()
        self._lower_bar.set_range(minimum, maximum)
        self._upper_bar.set_range(minimum, maximum)
        self.set_value(current)

    def set_value(self, value: Any) -> None:
        lower, upper = self._coerce_pair(value)
        with QtCore.QSignalBlocker(self._lower_bar):
            self._lower_bar.set_value(lower)
        with QtCore.QSignalBlocker(self._upper_bar):
            self._upper_bar.set_value(upper)

    def value(self) -> list[float | int]:
        return [self._lower_bar.value(), self._upper_bar.value()]

    def set_read_only(self, read_only: bool) -> None:
        self._lower_bar.set_read_only(bool(read_only))
        self._upper_bar.set_read_only(bool(read_only))

    def is_read_only(self) -> bool:
        return self._lower_bar.is_read_only() and self._upper_bar.is_read_only()

    def lower_bar(self) -> F8ValueBar:
        return self._lower_bar

    def upper_bar(self) -> F8ValueBar:
        return self._upper_bar

    def _coerce_pair(self, value: Any) -> tuple[float | int, float | int]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            lower_raw, upper_raw = value[0], value[1]
        else:
            lower_raw, upper_raw = self._lower_bar.value(), self._upper_bar.value()
        self._lower_bar.set_value(lower_raw)
        self._upper_bar.set_value(upper_raw)
        lower = self._lower_bar.value()
        upper = self._upper_bar.value()
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    def _on_bar_value(self, index: int, raw: Any, *, commit: bool) -> None:
        current = self.value()
        current[index] = raw
        lower, upper = current
        if lower > upper:
            if index == 0:
                lower = upper
            else:
                upper = lower
        self.set_value([lower, upper])
        value = self.value()
        if commit:
            self.valueCommitted.emit(value)
        else:
            self.valueChanging.emit(value)


class F8Dial(QtWidgets.QWidget):
    """
    Circular numeric dial with a full 360-degree range and a seam at 12 o'clock.
    """

    valueChanging = QtCore.Signal(object)  # float|int
    valueCommitted = QtCore.Signal(object)  # float|int

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: float | int | None = None,
        integer: bool = False,
        loop: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setMinimumSize(56, 56)
        self.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._min = float(minimum)
        self._max = float(maximum)
        if self._max < self._min:
            self._min, self._max = self._max, self._min
        self._integer = bool(integer)
        self._loop = bool(loop)
        self._value: float | int = self._coerce(value if value is not None else self._min)
        self._dragging = False
        self._drag_fraction = self._value_fraction()
        self._read_only = False
        self._invalid_reason = ""
        self._context_tooltip = ""

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return QtCore.QSize(76, 76)

    def set_range(self, minimum: float | int | None, maximum: float | int | None) -> None:
        lo = float(0.0 if minimum is None else minimum)
        hi = float(1.0 if maximum is None else maximum)
        if hi < lo:
            lo, hi = hi, lo
        self._min, self._max = lo, hi
        self.set_value(self._value)

    def set_value(self, value: Any) -> None:
        self._value = self._coerce(value)
        self._drag_fraction = self._value_fraction()
        self.update()

    def value(self) -> float | int:
        return self._value

    def set_loop(self, loop: bool) -> None:
        self._loop = bool(loop)

    def loop(self) -> bool:
        return self._loop

    def set_read_only(self, read_only: bool) -> None:
        self._read_only = bool(read_only)
        self._refresh_enabled()

    def set_invalid_reason(self, reason: str) -> None:
        self._invalid_reason = str(reason or "").strip()
        self._refresh_tooltip()
        self._refresh_enabled()

    def set_context_tooltip(self, tooltip: str) -> None:
        self._context_tooltip = str(tooltip or "").strip()
        self._refresh_tooltip()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self._dragging = True
            self._set_from_pos(event.position(), commit=False, absolute=True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._dragging and self.isEnabled():
            self._set_from_pos(event.position(), commit=False, absolute=False)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._dragging and event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self._dragging = False
            self._set_from_pos(event.position(), commit=True, absolute=False)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = QtCore.QRectF(self.rect()).adjusted(2.0, 2.0, -2.0, -2.0)
        size = min(rect.width(), rect.height())
        square = QtCore.QRectF(0.0, 0.0, size, size)
        square.moveCenter(rect.center())
        ring_width = max(6.0, size * 0.11)

        enabled = self.isEnabled()
        border = QtGui.QColor(255, 255, 255, 60 if enabled else 28)
        bg = QtGui.QColor(0, 0, 0, 45 if enabled else 25)
        fill = QtGui.QColor(120, 200, 255, 190 if enabled else 70)
        text = QtGui.QColor(235, 235, 235, 245 if enabled else 120)
        knob = QtGui.QColor(235, 235, 235, 245 if enabled else 120)

        painter.setPen(QtGui.QPen(border, 1.0))
        painter.setBrush(bg)
        painter.drawEllipse(square)

        ring_rect = square.adjusted(ring_width / 2.0, ring_width / 2.0, -ring_width / 2.0, -ring_width / 2.0)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 32 if enabled else 16), ring_width))
        painter.drawEllipse(ring_rect)

        fraction = self._value_fraction()
        if fraction > 0.0:
            fill_pen = QtGui.QPen(fill, ring_width)
            fill_pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(fill_pen)
            painter.drawArc(ring_rect, 90 * 16, -int(round(fraction * 360.0 * 16.0)))

        center = ring_rect.center()
        indicator_radius = ring_rect.width() / 2.0
        indicator_x = center.x() + math.cos((fraction * 2.0 * math.pi) - (math.pi / 2.0)) * indicator_radius
        indicator_y = center.y() + math.sin((fraction * 2.0 * math.pi) - (math.pi / 2.0)) * indicator_radius
        knob_radius = max(3.5, ring_width * 0.42)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(knob)
        painter.drawEllipse(QtCore.QPointF(indicator_x, indicator_y), knob_radius, knob_radius)

        painter.setPen(text)
        display_text = self._format_value(self._value)
        inner_rect = square.adjusted(ring_width * 1.9, ring_width * 1.9, -ring_width * 1.9, -ring_width * 1.9)
        font = QtGui.QFont(painter.font())
        pixel_size = max(8, int(inner_rect.height() * 0.32))
        font.setPixelSize(pixel_size)
        painter.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        while pixel_size > 8 and (
            metrics.horizontalAdvance(display_text) > inner_rect.width() * 0.92
            or metrics.height() > inner_rect.height() * 0.8
        ):
            pixel_size -= 1
            font.setPixelSize(pixel_size)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
        painter.drawText(inner_rect, QtCore.Qt.AlignCenter, display_text)

    def _refresh_enabled(self) -> None:
        self.setEnabled((not self._read_only) and (not self._invalid_reason))

    def _refresh_tooltip(self) -> None:
        tip_parts: list[str] = []
        if self._context_tooltip:
            tip_parts.append(self._context_tooltip)
        if self._invalid_reason:
            tip_parts.append(self._invalid_reason)
        self.setToolTip("\n".join(tip_parts))

    def _coerce(self, value: Any) -> float | int:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float(self._min)
        numeric = max(self._min, min(self._max, numeric))
        if self._integer:
            return int(round(numeric))
        return numeric

    def _format_value(self, value: float | int) -> str:
        if self._integer:
            return str(int(value))
        return ("{:.6f}".format(float(value))).rstrip("0").rstrip(".")

    def _span(self) -> float:
        return max(0.0, self._max - self._min)

    def _value_fraction(self) -> float:
        span = self._span()
        if span <= 0.0:
            return 0.0
        fraction = (float(self._value) - self._min) / span
        return max(0.0, min(1.0, fraction))

    def _fraction_to_value(self, fraction: float) -> float | int:
        span = self._span()
        if span <= 0.0:
            return int(round(self._min)) if self._integer else float(self._min)
        numeric = self._min + max(0.0, min(1.0, fraction)) * span
        if self._integer:
            return int(max(self._min, min(self._max, round(numeric))))
        return float(max(self._min, min(self._max, numeric)))

    def _point_to_fraction(self, pos: QtCore.QPointF) -> float:
        rect = QtCore.QRectF(self.rect()).adjusted(2.0, 2.0, -2.0, -2.0)
        center = rect.center()
        dx = float(pos.x() - center.x())
        dy = float(pos.y() - center.y())
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return self._value_fraction()
        angle = math.atan2(dy, dx)
        clockwise_from_top = (angle + (math.pi / 2.0)) % (2.0 * math.pi)
        return clockwise_from_top / (2.0 * math.pi)

    @staticmethod
    def _wrap_delta(next_fraction: float, current_fraction: float) -> float:
        delta = float(next_fraction) - float(current_fraction)
        if delta > 0.5:
            return delta - 1.0
        if delta < -0.5:
            return delta + 1.0
        return delta

    def _set_from_pos(self, pos: QtCore.QPointF, *, commit: bool, absolute: bool) -> None:
        pointer_fraction = self._point_to_fraction(pos)
        if absolute:
            value_fraction = pointer_fraction
        else:
            delta = self._wrap_delta(pointer_fraction, self._drag_fraction)
            value_fraction = self._value_fraction() + delta
            if self._loop:
                value_fraction = value_fraction % 1.0
            else:
                value_fraction = max(0.0, min(1.0, value_fraction))
        self._drag_fraction = pointer_fraction
        next_value = self._fraction_to_value(value_fraction)
        if next_value == self._value and not commit:
            return
        self._value = next_value
        self.update()
        if commit:
            self.valueCommitted.emit(next_value)
        else:
            self.valueChanging.emit(next_value)


@dataclass(frozen=True)
class _ImageB64Result:
    b64: str
    changed: bool


class _F8ImageB64Dialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, *, b64: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image")
        self.setModal(True)
        self.resize(980, 640)

        self._b64 = str(b64 or "")
        self._changed = False

        self._label = QtWidgets.QLabel()
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._label.setMinimumSize(480, 270)
        self._label.setStyleSheet(
            f"border: 1px solid {qss_rgba(studio_dark_theme().palette.text_primary, 45)}; border-radius: 4px;"
        )

        self._btn_load = QtWidgets.QPushButton("Load File")
        self._btn_load.clicked.connect(self._load_file)  # type: ignore[attr-defined]

        self._btn_clear = QtWidgets.QPushButton("Clear")
        self._btn_clear.clicked.connect(self._clear)  # type: ignore[attr-defined]

        self._btn_close = QtWidgets.QPushButton("Close")
        self._btn_close.clicked.connect(self.accept)  # type: ignore[attr-defined]

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self._btn_load)
        top.addWidget(self._btn_clear)
        top.addStretch(1)
        top.addWidget(self._btn_close)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._label, 1)

        self._refresh()

    def result_value(self) -> _ImageB64Result:
        return _ImageB64Result(b64=self._b64, changed=bool(self._changed))

    def _refresh(self) -> None:
        data = b""
        try:
            data = _b64decode_to_bytes(self._b64)
        except (binascii.Error, UnicodeEncodeError, ValueError):
            data = b""
        if not data:
            self._label.setText("No image")
            self._label.setPixmap(QtGui.QPixmap())
            return
        pix = QtGui.QPixmap()
        if not pix.loadFromData(data):
            self._label.setText("Invalid image data")
            self._label.setPixmap(QtGui.QPixmap())
            return
        self._label.setText("")
        self._label.setPixmap(pix.scaled(self._label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh()

    def _load_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as exc:
            show_warning(self, "Load failed", str(exc))
            return
        self._b64 = _b64encode_bytes(data)
        self._changed = True
        self._refresh()

    def _clear(self) -> None:
        self._b64 = ""
        self._changed = True
        self._refresh()


class F8ImageB64Editor(QtWidgets.QWidget):
    """
    Compact editor for a base64-encoded image.
    """

    valueChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._b64 = ""

        self._btn = QtWidgets.QPushButton("View Image")
        self._btn.setMinimumHeight(22)
        self._btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._btn.clicked.connect(self._open)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._btn, 1)

    def set_value(self, b64: str) -> None:
        self._b64 = str(b64 or "")
        self._btn.setText("View/Replace Image" if self._b64 else "Select Image")

    def value(self) -> str:
        return self._b64

    def set_disabled(self, disabled: bool) -> None:
        self._btn.setDisabled(bool(disabled))

    def _resolve_dialog_parent(self) -> QtWidgets.QWidget | None:
        return _resolve_embedded_dialog_parent(self)

    def _open(self) -> None:
        parent = self._resolve_dialog_parent()
        dlg = _F8ImageB64Dialog(parent, b64=self._b64)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        res = dlg.result_value()
        if res.changed:
            self.set_value(res.b64)
            self.valueChanged.emit(self._b64)


class F8MultiSelect(QtWidgets.QWidget):
    """
    Compact multi-select editor.

    Uses a dialog-based checklist instead of QMenu popups because this widget
    can be embedded inside QGraphicsProxyWidget (NodeGraph), where popup menus
    are not always reliable.
    """

    valueChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._values: list[str] = []
        self._labels: dict[str, str] = {}
        self._tooltips: dict[str, str] = {}
        self._selected: list[str] = []
        self._context_tooltip = ""
        self._read_only = False

        self._button = QtWidgets.QToolButton()
        self._button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._button.setMinimumHeight(22)
        self._button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._button.setText("None")
        self._button.clicked.connect(self._open_dialog)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._button, 1)

    def set_read_only(self, read_only: bool) -> None:
        self._read_only = bool(read_only)
        self._button.setEnabled(not self._read_only)

    def set_context_tooltip(self, tooltip: str) -> None:
        self._context_tooltip = str(tooltip or "").strip()
        self._refresh_caption()

    def set_options(
        self,
        values: list[Any],
        *,
        labels: list[str] | None = None,
        tooltips: list[str] | None = None,
    ) -> None:
        self._values = [str(v) for v in list(values)]
        self._labels = {}
        self._tooltips = {}
        labels_list = list(labels) if labels is not None else []
        tips_list = list(tooltips) if tooltips is not None else []
        for i, value in enumerate(self._values):
            if i < len(labels_list):
                self._labels[value] = str(labels_list[i])
            if i < len(tips_list):
                self._tooltips[value] = str(tips_list[i])
        valid_values = set(self._values)
        self._selected = [v for v in self._selected if v in valid_values]
        self._refresh_caption()

    def set_value(self, value: Any) -> None:
        values = self._normalize_values(value)
        selected_set = set(values)
        self._selected = [v for v in self._values if v in selected_set]
        self._refresh_caption()

    def value(self) -> list[str]:
        return list(self._selected)

    def _normalize_values(self, value: Any) -> list[str]:
        raw_values: list[str] = []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raw_values = []
            else:
                parsed: Any = None
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, (list, tuple)):
                    raw_values = [str(v) for v in parsed]
                else:
                    raw_values = [v.strip() for v in text.split(",")]
        elif isinstance(value, (list, tuple, set)):
            raw_values = [str(v) for v in value]
        else:
            raw_values = []
        out: list[str] = []
        seen: set[str] = set()
        for v in raw_values:
            name = str(v).strip()
            if not name or name in seen:
                continue
            out.append(name)
            seen.add(name)
        return out

    def _resolve_dialog_parent(self) -> QtWidgets.QWidget | None:
        return _resolve_embedded_dialog_parent(self)

    @staticmethod
    def _set_list_checked(list_widget: QtWidgets.QListWidget, checked: bool) -> None:
        state = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
        for idx in range(list_widget.count()):
            item = list_widget.item(idx)
            if item is None:
                continue
            item.setCheckState(state)

    def _open_dialog(self) -> None:
        if self._read_only:
            return
        parent = self._resolve_dialog_parent()
        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle("Select Classes")
        dlg.setModal(True)
        dlg.resize(420, 520)

        list_widget = QtWidgets.QListWidget(dlg)
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        selected_set = set(self._selected)

        user_role = QtCore.Qt.ItemDataRole.UserRole
        checked_state = QtCore.Qt.CheckState.Checked
        unchecked_state = QtCore.Qt.CheckState.Unchecked
        user_checkable_flag = QtCore.Qt.ItemFlag.ItemIsUserCheckable

        for value in self._values:
            label = self._labels.get(value, value)
            item = QtWidgets.QListWidgetItem(label, list_widget)
            item.setData(user_role, value)
            item.setFlags(item.flags() | user_checkable_flag)
            item.setCheckState(checked_state if value in selected_set else unchecked_state)
            tip = str(self._tooltips.get(value, "")).strip()
            if tip:
                item.setToolTip(tip)

        btn_all = QtWidgets.QPushButton("Select All", dlg)
        btn_all.clicked.connect(lambda: self._set_list_checked(list_widget, True))  # type: ignore[attr-defined]
        btn_clear = QtWidgets.QPushButton("Clear", dlg)
        btn_clear.clicked.connect(lambda: self._set_list_checked(list_widget, False))  # type: ignore[attr-defined]

        row = QtWidgets.QHBoxLayout()
        row.addWidget(btn_all)
        row.addWidget(btn_clear)
        row.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg
        )
        buttons.accepted.connect(dlg.accept)  # type: ignore[attr-defined]
        buttons.rejected.connect(dlg.reject)  # type: ignore[attr-defined]

        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addLayout(row)
        layout.addWidget(list_widget, 1)
        layout.addWidget(buttons)

        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return

        out: list[str] = []
        for idx in range(list_widget.count()):
            item = list_widget.item(idx)
            if item is None:
                continue
            if item.checkState() != checked_state:
                continue
            value = str(item.data(user_role) or "").strip()
            if value:
                out.append(value)
        self._selected = out
        self._refresh_caption()
        self.valueChanged.emit(self.value())

    def _refresh_caption(self) -> None:
        count = len(self._selected)
        total = len(self._values)
        if count <= 0:
            text = "None"
        elif count == total and total > 0:
            text = f"All ({total})"
        elif count <= 3:
            labels = [self._labels.get(v, v) for v in self._selected]
            text = ", ".join(labels)
        else:
            text = f"{count} selected"
        self._button.setText(text)

        selected_labels = [self._labels.get(v, v) for v in self._selected]
        selected_text = ", ".join(selected_labels) if selected_labels else "None"
        tip_parts: list[str] = []
        if self._context_tooltip:
            tip_parts.append(self._context_tooltip)
        tip_parts.append(f"Selected: {selected_text}")
        self._button.setToolTip("\n".join(tip_parts))
