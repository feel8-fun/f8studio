from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets


class _F8FlowLayout(QtWidgets.QLayout):
    """
    Simple flow layout (wraps widgets horizontally).
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, margin: int = 0, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # type: ignore[override]
        self._items.append(item)

    def count(self) -> int:  # type: ignore[override]
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # type: ignore[override]
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # type: ignore[override]
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> QtCore.Qt.Orientations:  # type: ignore[override]
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return True

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # type: ignore[override]
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:  # type: ignore[override]
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        l, t, r, b = self.getContentsMargins()
        size += QtCore.QSize(l + r, t + b)
        return size

    def _do_layout(self, rect: QtCore.QRect, *, test_only: bool) -> int:
        l, t, r, b = self.getContentsMargins()
        effective = rect.adjusted(l, t, -r, -b)
        x = effective.x()
        y = effective.y()
        line_height = 0
        space_x = self.spacing()
        space_y = self.spacing()

        for item in self._items:
            w = item.widget()
            if w is not None and not w.isVisible():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + space_x
            if next_x - space_x > effective.right() and line_height > 0:
                x = effective.x()
                y += line_height + space_y
                next_x = x + hint.width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return (y + line_height + b) - rect.y()


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


class F8ToggleButton(QtWidgets.QToolButton):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumHeight(22)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setStyleSheet(
            """
            QToolButton {
                color: rgb(235, 235, 235);
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 4px;
                padding: 2px 8px;
            }
            QToolButton:hover {
                border-color: rgba(255, 255, 255, 70);
            }
            QToolButton:checked {
                background: rgba(120, 200, 255, 60);
                border-color: rgba(120, 200, 255, 140);
            }
            QToolButton:disabled {
                color: rgba(235, 235, 235, 90);
                border-color: rgba(255, 255, 255, 18);
                background: rgba(0, 0, 0, 18);
            }
            QToolButton:checked:disabled {
                color: rgba(235, 235, 235, 120);
                background: rgba(120, 200, 255, 28);
                border-color: rgba(120, 200, 255, 70);
            }
            """
        )


class F8ExclusiveToggleRow(QtWidgets.QWidget):
    """
    Exclusive toggle buttons (radio-like) built from QToolButtons.
    """

    valueChanged = QtCore.Signal(object)  # object value (typically str/bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, spacing: int = 4) -> None:
        super().__init__(parent)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[F8ToggleButton] = []
        self._values: list[Any] = []
        self._context_tooltip = ""

        layout = _F8FlowLayout(self, margin=0, spacing=spacing)
        self.setLayout(layout)
        self._layout = layout

        self._group.buttonToggled.connect(self._on_toggled)  # type: ignore[attr-defined]

    def set_context_tooltip(self, tooltip: str) -> None:
        self._context_tooltip = str(tooltip or "").strip()
        for b in self._buttons:
            b.setToolTip(self._button_tooltip(b))

    def set_options(
        self,
        values: list[Any],
        *,
        labels: list[str] | None = None,
        tooltips: list[str] | None = None,
    ) -> None:
        cur = self.value()

        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()

        self._buttons.clear()
        self._values = list(values)
        if labels is None:
            labels = [str(v) for v in values]
        labels = list(labels)
        if tooltips is None:
            tooltips = ["" for _ in values]
        tooltips = list(tooltips)

        for i, v in enumerate(values):
            b = F8ToggleButton(self)
            b.setText(str(labels[i]) if i < len(labels) else str(v))
            b.setProperty("_f8_toggle_value", v)
            b.setAutoExclusive(True)
            b.setToolTip(self._button_tooltip(b, tooltip=(tooltips[i] if i < len(tooltips) else "")))
            self._group.addButton(b, i)
            self._layout.addWidget(b)
            self._buttons.append(b)

        # restore selection best-effort.
        self.set_value(cur)

    def set_value(self, value: Any) -> None:
        idx = -1
        for i, v in enumerate(self._values):
            if v == value:
                idx = i
                break
        with QtCore.QSignalBlocker(self._group):
            for i, b in enumerate(self._buttons):
                b.setChecked(i == idx)

    def value(self) -> Any:
        b = self._group.checkedButton()
        if b is None:
            return None
        return b.property("_f8_toggle_value")

    def set_disabled(self, disabled: bool) -> None:
        for b in self._buttons:
            b.setDisabled(bool(disabled))

    def _button_tooltip(self, b: QtWidgets.QAbstractButton, tooltip: str = "") -> str:
        label = str(getattr(b, "text", lambda: "")() or "")
        v = b.property("_f8_toggle_value")
        base = str(tooltip or "").strip()
        if not base:
            base = label
        extra = []
        if self._context_tooltip:
            extra.append(self._context_tooltip)
        if v is not None and str(v) != label:
            extra.append(f"Value: {v}")
        return "\n".join([base] + extra) if extra else base

    def _on_toggled(self, button: QtWidgets.QAbstractButton, checked: bool) -> None:  # noqa: ARG002
        if not checked:
            return
        self.valueChanged.emit(self.value())


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

    def _open(self) -> None:
        parent = self.window()
        dlg = _F8ImageB64Dialog(parent, b64=self._b64)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        res = dlg.result_value()
        if res.changed:
            self.set_value(res.b64)
            self.valueChanged.emit(self._b64)


class F8PropOptionToggle(QtWidgets.QWidget):
    """
    PropertiesBin-compatible option editor (exclusive toggles).
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = ""
        self._row = F8ExclusiveToggleRow()
        self._row.valueChanged.connect(self._emit)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._row, 1)

        self._pool_field: str | None = None
        self._pool_resolver: Callable[[str], list[str]] | None = None

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_items(self, items: list[str]) -> None:
        self._row.set_options(list(items), labels=list(items))

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
        self._row.set_value("" if value is None else str(value))

    def get_value(self) -> Any:
        v = self._row.value()
        if v is None:
            return None
        return str(v)

    def _emit(self, v: Any) -> None:
        self.value_changed.emit(self.get_name(), None if v is None else str(v))


class F8PropBoolToggle(QtWidgets.QWidget):
    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = ""
        self._row = F8ExclusiveToggleRow()
        self._row.set_options([True, False], labels=["True", "False"], tooltips=["Set True", "Set False"])
        self._row.valueChanged.connect(self._emit)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._row, 1)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_value(self, value: Any) -> None:
        self._row.set_value(bool(value) if value is not None else None)

    def get_value(self) -> Any:
        v = self._row.value()
        if v is None:
            return None
        return bool(v)

    def _emit(self, v: Any) -> None:
        self.value_changed.emit(self.get_name(), None if v is None else bool(v))


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
