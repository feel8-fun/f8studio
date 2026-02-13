from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets

_COMBO_REOPEN_GUARD_S = 0.05


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


def parse_select_pool(ui_control: str) -> str | None:
    """
    Parse uiControl patterns for option pools:
      "select:[poolStateField]"
      "options:[poolStateField]"
    """
    ui = str(ui_control or "").strip()
    m = re.match(r"^(select|options)\s*:\s*\[([A-Za-z_][A-Za-z0-9_]*)\]\s*$", ui, flags=re.IGNORECASE)
    if not m:
        return None
    return str(m.group(2))


def parse_multiselect_pool(ui_control: str) -> str | None:
    """
    Parse uiControl patterns for multi-select pools:
      "multiselect:[poolStateField]"
    """
    ui = str(ui_control or "").strip()
    m = re.match(r"^multiselect\s*:\s*\[([A-Za-z_][A-Za-z0-9_]*)\]\s*$", ui, flags=re.IGNORECASE)
    if not m:
        return None
    return str(m.group(1))

class _F8ComboPopup(QtWidgets.QFrame):
    valueSelected = QtCore.Signal(int)

    def __init__(self, parent_combo: "F8OptionCombo") -> None:
        super().__init__(
            None,
            QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint,
        )
        self._combo = parent_combo
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._bg_color = QtGui.QColor(35, 35, 35)
        self._border_color = QtGui.QColor(255, 255, 255, 55)
        self._radius = 6.0
        self.setStyleSheet(
            """
            QListView {
                background: transparent;
                color: rgb(235, 235, 235);
                selection-background-color: rgb(80, 130, 180);
                outline: 0;
                border: 0px;
            }
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
        except Exception:
            return
        self._view.setCurrentIndex(idx)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:  # type: ignore[override]
        super().focusOutEvent(event)
        self.hide()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # type: ignore[override]
        super().hideEvent(event)
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
        self.setEditable(False)
        self.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumHeight(22)
        self.setMaxVisibleItems(16)
        view = QtWidgets.QListView()
        view.setUniformItemSizes(True)
        self.setView(view)
        self.currentIndexChanged.connect(self._emit)  # type: ignore[attr-defined]
        self._popup = _F8ComboPopup(self)
        self._popup.valueSelected.connect(self._on_popup_selected)  # type: ignore[attr-defined]

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
        if ro:
            self.setEditable(True)
            le = self.lineEdit()
            if le is not None:
                le.setReadOnly(True)
                le.setTextInteractionFlags(
                    QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                    | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
                )
        else:
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
        if self._popup.isVisible():
            # Toggle behavior: clicking the combobox again collapses the popup.
            self.hidePopup()
            return
        if not self.isEnabled():
            return
        model = self.model()
        if model is None or model.rowCount() == 0:
            return
        self._popup.set_model(model)
        self._popup.set_current_index(self.currentIndex())
        self._popup.resize(self._popup_size())
        pos = self._popup_pos(self._popup.height())
        self._popup.move(pos)
        self._popup.raise_()
        self._popup.show()
        self._popup.activateWindow()

    def hidePopup(self) -> None:  # type: ignore[override]
        if self._popup.isVisible():
            self._block_popup_for(_COMBO_REOPEN_GUARD_S)
            self._popup.hide()

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
            except Exception:
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

    def _popup_pos(self, popup_h: int) -> QtCore.QPoint:
        anchor = self._anchor_global()
        top_left = QtCore.QPoint(anchor.x(), anchor.y() - self.height())
        below = anchor
        above = QtCore.QPoint(top_left.x(), top_left.y() - popup_h)
        screen = QtGui.QGuiApplication.screenAt(below)
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
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
                        view = next((v for v in views if v.isVisible()), views[0])
                        root = proxy.widget()
                        if root is not None:
                            local_pt = self.mapTo(root, self.rect().bottomLeft())
                            scene_pos = proxy.mapToScene(QtCore.QPointF(local_pt))
                        else:
                            scene_pos = proxy.mapToScene(QtCore.QPointF(self.rect().bottomLeft()))
                        view_pt = view.mapFromScene(scene_pos)
                        return view.viewport().mapToGlobal(view_pt)
        except Exception:
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
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self._dragging = True
            self._set_from_pos(event.position().x(), commit=False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._dragging and self.isEnabled():
            self._set_from_pos(event.position().x(), commit=False)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._dragging and event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self._dragging = False
            self._set_from_pos(event.position().x(), commit=True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        del event
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = 4.0

        border = QtGui.QColor(255, 255, 255, 55 if self.isEnabled() else 25)
        bg = QtGui.QColor(0, 0, 0, 45)
        fill = QtGui.QColor(120, 200, 255, 70 if self.isEnabled() else 30)

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

        text = self._format_value(self._value)
        p.setPen(QtGui.QColor(235, 235, 235, 255 if self.isEnabled() else 120))
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
        self._label.setStyleSheet("border: 1px solid rgba(255,255,255,35); border-radius: 4px;")

        self._btn_load = QtWidgets.QPushButton("Load File…")
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
        except Exception:
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
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
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

        self._btn = QtWidgets.QPushButton("View Image…")
        self._btn.setMinimumHeight(22)
        self._btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._btn.clicked.connect(self._open)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._btn, 1)

    def set_value(self, b64: str) -> None:
        self._b64 = str(b64 or "")
        self._btn.setText("View/Replace Image…" if self._b64 else "Select Image…")

    def value(self) -> str:
        return self._b64

    def set_disabled(self, disabled: bool) -> None:
        self._btn.setDisabled(bool(disabled))

    def _resolve_dialog_parent(self) -> QtWidgets.QWidget | None:
        # When embedded in a QGraphicsProxyWidget, self.window() can be the proxy,
        # which makes dialogs appear inside the scene (scaled/transparent). Find
        # the real window from the scene view instead.
        proxy = None
        try:
            w: QtWidgets.QWidget | None = self
            while w is not None and proxy is None:
                try:
                    proxy = w.graphicsProxyWidget()
                except Exception:
                    proxy = None
                try:
                    w = w.parentWidget()
                except Exception:
                    w = None
        except Exception:
            proxy = None
        if proxy is not None:
            try:
                scene = proxy.scene()
            except Exception:
                scene = None
            if scene is not None:
                try:
                    views = scene.views()
                except Exception:
                    views = []
                if views:
                    view = next((v for v in views if v.isVisible()), views[0])
                    try:
                        w = view.window()
                        if w is not None:
                            return w
                    except Exception:
                        pass
        try:
            w = self.window()
            if w is not None:
                return w
        except Exception:
            pass
        try:
            return QtWidgets.QApplication.activeWindow()
        except Exception:
            return None

    def _open(self) -> None:
        parent = self._resolve_dialog_parent()
        dlg = _F8ImageB64Dialog(parent, b64=self._b64)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        res = dlg.result_value()
        if res.changed:
            self.set_value(res.b64)
            self.valueChanged.emit(self._b64)


class F8PropOptionCombo(QtWidgets.QWidget):
    """
    PropertiesBin-compatible option editor (combo box).
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = ""
        self._combo = F8OptionCombo()
        self._combo.valueChanged.connect(self._emit)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo, 1)

        self._pool_field: str | None = None
        self._pool_resolver: Callable[[str], list[str]] | None = None

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_items(self, items: list[str]) -> None:
        self._combo.set_options(list(items), labels=list(items))

    def set_pool(self, pool_field: str, resolver: Callable[[str], list[str]]) -> None:
        self._pool_field = str(pool_field or "")
        self._pool_resolver = resolver
        self.refresh_options()

    def refresh_options(self) -> None:
        if not self._pool_field or self._pool_resolver is None:
            return
        items = self._pool_resolver(self._pool_field)
        self.set_items(items)

    def set_value(self, value: Any) -> None:
        self._combo.set_value("" if value is None else str(value))

    def get_value(self) -> Any:
        v = self._combo.value()
        if v is None:
            return None
        return str(v)

    def set_context_tooltip(self, tooltip: str) -> None:
        self._combo.set_context_tooltip(tooltip)

    def set_read_only(self, read_only: bool) -> None:
        self._combo.set_read_only(bool(read_only))

    def _emit(self, v: Any) -> None:
        self.value_changed.emit(self.get_name(), None if v is None else str(v))


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
                except Exception:
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
        # Same pattern as F8ImageB64Editor: when embedded in QGraphicsProxyWidget,
        # resolve to the real window to avoid scene-embedded dialogs.
        proxy = None
        try:
            w: QtWidgets.QWidget | None = self
            while w is not None and proxy is None:
                try:
                    proxy = w.graphicsProxyWidget()
                except Exception:
                    proxy = None
                try:
                    w = w.parentWidget()
                except Exception:
                    w = None
        except Exception:
            proxy = None
        if proxy is not None:
            try:
                scene = proxy.scene()
            except Exception:
                scene = None
            if scene is not None:
                try:
                    views = scene.views()
                except Exception:
                    views = []
                if views:
                    view = next((v for v in views if v.isVisible()), views[0])
                    try:
                        top = view.window()
                        if top is not None:
                            return top
                    except Exception:
                        pass
        try:
            top = self.window()
            if top is not None:
                return top
        except Exception:
            pass
        try:
            return QtWidgets.QApplication.activeWindow()
        except Exception:
            return None

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

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg)
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


class F8PropMultiSelect(QtWidgets.QWidget):
    """
    PropertiesBin-compatible multi-select editor.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = ""
        self._multi = F8MultiSelect()
        self._multi.valueChanged.connect(self._emit)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._multi, 1)

        self._pool_field: str | None = None
        self._pool_resolver: Callable[[str], list[str]] | None = None

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_items(self, items: list[str]) -> None:
        self._multi.set_options(list(items), labels=list(items))

    def set_pool(self, pool_field: str, resolver: Callable[[str], list[str]]) -> None:
        self._pool_field = str(pool_field or "")
        self._pool_resolver = resolver
        self.refresh_options()

    def refresh_options(self) -> None:
        if not self._pool_field or self._pool_resolver is None:
            return
        items = self._pool_resolver(self._pool_field)
        self.set_items(items)

    def set_value(self, value: Any) -> None:
        self._multi.set_value(value)

    def get_value(self) -> Any:
        return self._multi.value()

    def set_context_tooltip(self, tooltip: str) -> None:
        self._multi.set_context_tooltip(tooltip)

    def set_read_only(self, read_only: bool) -> None:
        self._multi.set_read_only(bool(read_only))

    def _emit(self, v: Any) -> None:
        out = [str(x) for x in list(v or [])]
        self.value_changed.emit(self.get_name(), out)


class F8PropBoolSwitch(QtWidgets.QWidget):
    """
    PropertiesBin-compatible boolean editor (switch).
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = ""
        self._switch = F8Switch()
        self._switch.set_labels("True", "False")
        self._switch.valueChanged.connect(self._emit)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._switch, 1)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_value(self, value: Any) -> None:
        self._switch.set_value(bool(value) if value is not None else False)

    def get_value(self) -> Any:
        return bool(self._switch.value())

    def set_context_tooltip(self, tooltip: str) -> None:
        self._switch.setToolTip(str(tooltip or ""))

    def set_read_only(self, read_only: bool) -> None:
        self._switch.setEnabled(not bool(read_only))

    def _emit(self, v: Any) -> None:
        self.value_changed.emit(self.get_name(), bool(v))


class F8PropValueBar(QtWidgets.QWidget):
    """
    PropertiesBin-compatible value bar (emits on release).
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, data_type: type[int] | type[float]):
        super().__init__(parent)
        self._name = ""
        self._data_type = data_type
        self._min: float | int | None = None
        self._max: float | int | None = None
        self._bar = F8ValueBar(integer=(data_type is int), minimum=0.0, maximum=1.0)
        self._bar.valueCommitted.connect(self._emit)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._bar, 1)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_min(self, v) -> None:
        self._min = v
        self._bar.set_range(self._min, self._max)

    def set_max(self, v) -> None:
        self._max = v
        self._bar.set_range(self._min, self._max)

    def set_value(self, value: Any) -> None:
        self._bar.set_value(value)

    def get_value(self) -> Any:
        v = self._bar.value()
        return int(v) if self._data_type is int else float(v)

    def _emit(self, v: Any) -> None:
        out = int(v) if self._data_type is int else float(v)
        self.value_changed.emit(self.get_name(), out)


class F8PropImageB64(QtWidgets.QWidget):
    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = ""
        self._w = F8ImageB64Editor()
        self._w.valueChanged.connect(self._emit)  # type: ignore[attr-defined]
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._w, 1)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_value(self, value: Any) -> None:
        self._w.set_value("" if value is None else str(value))

    def get_value(self) -> Any:
        return str(self._w.value() or "")

    def _emit(self, v: str) -> None:
        self.value_changed.emit(self.get_name(), str(v or ""))
