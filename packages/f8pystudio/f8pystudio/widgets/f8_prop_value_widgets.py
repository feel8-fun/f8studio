from __future__ import annotations

import json
import os
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


class F8MonacoEditorDialog(QtWidgets.QDialog):
    """
    Monaco-based editor dialog (syntax highlighting, modern keybindings).

    Monaco assets can be loaded from:
    - `F8_MONACO_BASE_URL` (recommended for packaged/offline builds)
    - CDN fallback (default) for dev
    """

    def __init__(self, parent=None, *, title: str, code: str, language: str = "python"):
        super().__init__(parent)
        self.setWindowTitle(str(title or "Edit Code"))
        self._code: str = str(code or "")
        self._language: str = str(language or "plaintext").strip() or "plaintext"

        from PySide6 import QtWebEngineWidgets  # type: ignore[import-not-found]

        self._view = QtWebEngineWidgets.QWebEngineView(self)
        self._view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_save_clicked)  # type: ignore[attr-defined]
        buttons.rejected.connect(self.reject)  # type: ignore[attr-defined]

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._view, 1)
        layout.addWidget(buttons)

        self.resize(1020, 720)
        self._load_page()

    def code(self) -> str:
        return str(self._code or "")

    def _monaco_base_url(self) -> str:
        v = str(os.environ.get("F8_MONACO_BASE_URL") or "").strip().rstrip("/")
        if v:
            return v
        return "https://cdn.jsdelivr.net/npm/monaco-editor@0.50.0/min"

    def _load_page(self) -> None:
        base = self._monaco_base_url()
        initial = {"code": self._code, "language": self._language, "theme": "vs-dark"}
        initial_json = json.dumps(initial, ensure_ascii=False)

        html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      html, body, #container {{
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
        background: #1e1e1e;
      }}
    </style>
    <script>
      window.__F8_INITIAL__ = {initial_json};
    </script>
    <script src="{base}/vs/loader.js"></script>
    <script>
      window._f8_editor = null;
      window._f8_getValue = function() {{
        try {{
          if (!window._f8_editor) return "";
          return window._f8_editor.getValue();
        }} catch (e) {{
          return "";
        }}
      }};

      require.config({{ paths: {{ 'vs': '{base}/vs' }} }});
      require(['vs/editor/editor.main'], function() {{
        const init = window.__F8_INITIAL__ || {{ code: '', language: 'plaintext', theme: 'vs-dark' }};
        window._f8_editor = monaco.editor.create(document.getElementById('container'), {{
          value: String(init.code || ''),
          language: String(init.language || 'plaintext'),
          theme: String(init.theme || 'vs-dark'),
          automaticLayout: true,
          minimap: {{ enabled: false }},
          fontLigatures: true,
          fontSize: 13,
          tabSize: 4,
          insertSpaces: true,
          scrollBeyondLastLine: false,
          wordWrap: 'off',
        }});
      }});
    </script>
  </head>
  <body>
    <div id="container"></div>
  </body>
</html>
"""
        self._view.setHtml(html)

    def _on_save_clicked(self) -> None:
        try:
            page = self._view.page()
        except Exception:
            page = None
        if page is None:
            self.accept()
            return

        def _on_value(value: Any) -> None:
            try:
                self._code = "" if value is None else str(value)
            except Exception:
                self._code = ""
            self.accept()

        try:
            page.runJavaScript("window._f8_getValue && window._f8_getValue();", _on_value)  # type: ignore[call-arg]
        except Exception:
            self.accept()


def open_code_editor_dialog(
    parent: QtWidgets.QWidget | None,
    *,
    title: str,
    code: str,
    language: str,
) -> str | None:
    """
    Open the best available code editor dialog and return updated code, or None if cancelled.
    """
    try:
        dlg = F8MonacoEditorDialog(parent, title=title, code=code, language=language)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        return dlg.code()
    except Exception:
        dlg2 = F8CodeEditorDialog(parent, title=title, code=code)
        if dlg2.exec() != QtWidgets.QDialog.Accepted:
            return None
        return dlg2.code()


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
        updated = open_code_editor_dialog(self, title=self._title, code=self.get_value(), language="python")
        if updated is None:
            return
        self.set_value(updated)
        self.value_changed.emit(self.get_name(), updated)


class F8CodeButtonPropWidget(QtWidgets.QWidget):
    """
    A single "Edit…" button that opens a code editor dialog.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, title: str = "Edit Code", language: str = "python"):
        super().__init__(parent)
        self._name = ""
        self._value = ""
        self._title = str(title or "Edit Code")
        self._language = str(language or "plaintext").strip() or "plaintext"

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
        updated = open_code_editor_dialog(self, title=self._title, code=self.get_value(), language=self._language)
        if updated is None:
            return
        self.set_value(updated)
        self.value_changed.emit(self.get_name(), updated)


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

