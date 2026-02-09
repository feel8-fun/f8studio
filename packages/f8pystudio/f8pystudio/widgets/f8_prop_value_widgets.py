from __future__ import annotations

import json
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets
import qtawesome as qta


class F8CodeEditorDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, title: str, code: str):
        super().__init__(parent)
        self.setWindowTitle(title)

        self._edit = QtWidgets.QPlainTextEdit()
        self._edit.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        try:
            font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            self._edit.setFont(font)
        except Exception:
            pass
        self._edit.setPlainText(str(code or ""))

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._edit, 1)
        layout.addWidget(buttons)

        self.resize(900, 650)

    def code(self) -> str:
        return str(self._edit.toPlainText() or "")


class F8CodePropWidget(QtWidgets.QWidget):
    """
    Read-only preview with an "Edit…" button that opens a code editor dialog.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, title: str = "Edit Code"):
        super().__init__(parent)
        self._name = ""
        self._value = ""
        self._title = str(title or "Edit Code")

        self._preview = QtWidgets.QLineEdit()
        self._preview.setReadOnly(True)
        self._preview.setClearButtonEnabled(False)

        self._btn = QtWidgets.QPushButton("Edit…")
        self._btn.clicked.connect(self._on_edit_clicked)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._preview, 1)
        layout.addWidget(self._btn, 0)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def get_value(self) -> str:
        return str(self._value or "")

    def set_value(self, value: Any) -> None:
        self._value = str(value or "")
        lines = self._value.splitlines()
        n = len(lines)
        preview = f"{n} line" if n == 1 else f"{n} lines"
        if lines:
            head = lines[0].strip()
            if head:
                preview = f"{preview} — {head[:80]}"
        self._preview.setText(preview)

    def _on_edit_clicked(self) -> None:
        dlg = F8CodeEditorDialog(self, title=self._title, code=self.get_value())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        code = dlg.code()
        self.set_value(code)
        self.value_changed.emit(self.get_name(), code)


class F8CodeButtonPropWidget(QtWidgets.QWidget):
    """
    A single "Edit…" button that opens a code editor dialog.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, title: str = "Edit Code"):
        super().__init__(parent)
        self._name = ""
        self._value = ""
        self._title = str(title or "Edit Code")

        self._btn = QtWidgets.QPushButton("Edit…")
        try:
            self._btn.setIcon(qta.icon("fa5s.code", color="white"))
        except Exception:
            pass
        self._btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self._btn.clicked.connect(self._on_edit_clicked)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._btn, 1)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def get_value(self) -> str:
        return str(self._value or "")

    def set_value(self, value: Any) -> None:
        self._value = str(value or "")

    def _on_edit_clicked(self) -> None:
        dlg = F8CodeEditorDialog(self, title=self._title, code=self.get_value())
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        code = dlg.code()
        self.set_value(code)
        self.value_changed.emit(self.get_name(), code)


class F8JsonPropTextEdit(QtWidgets.QTextEdit):
    """
    QTextEdit property widget that round-trips JSON values as python objects.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name: str | None = None
        self._prev_text = ""
        self._prev_value: Any = None

    def get_name(self) -> str:
        return self._name or ""

    def set_name(self, name: str) -> None:
        self._name = name

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._prev_text = self.toPlainText()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if self._prev_text == self.toPlainText():
            return
        text = self.toPlainText().strip()
        if not text:
            self._prev_value = None
            self.value_changed.emit(self.get_name(), None)
            self._prev_text = ""
            return
        try:
            obj = json.loads(text)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid JSON", str(e))
            self.setPlainText(self._prev_text)
            return
        self._prev_value = obj
        self.value_changed.emit(self.get_name(), obj)
        self._prev_text = text

    def get_value(self):
        return self._prev_value

    def set_value(self, value: Any) -> None:
        self._prev_value = value
        with QtCore.QSignalBlocker(self):
            if value is None:
                self.setPlainText("")
            else:
                self.setPlainText(json.dumps(value, ensure_ascii=False, indent=2))


class F8NumberPropLineEdit(QtWidgets.QLineEdit):
    """
    LineEdit that validates and emits int/float values.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, data_type: type = float):
        super().__init__(parent)
        self._name = ""
        self._data_type = data_type
        self._min: float | None = None
        self._max: float | None = None
        self.setMinimumWidth(120)
        self.editingFinished.connect(self._emit_value)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_min(self, v) -> None:
        try:
            self._min = float(v)
        except Exception:
            self._min = None

    def set_max(self, v) -> None:
        try:
            self._max = float(v)
        except Exception:
            self._max = None

    def get_value(self):
        t = str(self.text() or "").strip()
        if t == "":
            return None
        try:
            v = float(t)
            if self._min is not None:
                v = max(v, self._min)
            if self._max is not None:
                v = min(v, self._max)
            if self._data_type is int:
                return int(round(v))
            return float(v)
        except Exception:
            return None

    def set_value(self, value) -> None:
        if value is None:
            return
        with QtCore.QSignalBlocker(self):
            self.setText(str(value))

    def _emit_value(self) -> None:
        v = self.get_value()
        if v is None and str(self.text() or "").strip() != "":
            # invalid -> keep focus and don't emit.
            return
        self.value_changed.emit(self.get_name(), v)


class F8IntSpinBoxPropWidget(QtWidgets.QSpinBox):
    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = ""
        self.setMinimumWidth(120)
        self.editingFinished.connect(self._emit_value)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_min(self, v) -> None:
        try:
            self.setMinimum(int(v))
        except Exception:
            pass

    def set_max(self, v) -> None:
        try:
            self.setMaximum(int(v))
        except Exception:
            pass

    def get_value(self):
        try:
            return int(self.value())
        except Exception:
            return None

    def set_value(self, value) -> None:
        if value is None:
            return
        with QtCore.QSignalBlocker(self):
            self.setValue(int(value))

    def _emit_value(self) -> None:
        self.value_changed.emit(self.get_name(), self.get_value())


class F8DoubleSpinBoxPropWidget(QtWidgets.QDoubleSpinBox):
    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = ""
        self.setMinimumWidth(120)
        try:
            self.setDecimals(6)
        except Exception:
            pass
        self.editingFinished.connect(self._emit_value)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_min(self, v) -> None:
        try:
            self.setMinimum(float(v))
        except Exception:
            pass

    def set_max(self, v) -> None:
        try:
            self.setMaximum(float(v))
        except Exception:
            pass

    def get_value(self):
        try:
            return float(self.value())
        except Exception:
            return None

    def set_value(self, value) -> None:
        if value is None:
            return
        with QtCore.QSignalBlocker(self):
            self.setValue(float(value))

    def _emit_value(self) -> None:
        self.value_changed.emit(self.get_name(), self.get_value())

