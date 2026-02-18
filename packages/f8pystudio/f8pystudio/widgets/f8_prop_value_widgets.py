from __future__ import annotations

import json
import math
import os
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets
import qtawesome as qta


def _ask_save_before_close(parent: QtWidgets.QWidget) -> QtWidgets.QMessageBox.StandardButton:
    return QtWidgets.QMessageBox.question(
        parent,
        "Unsaved Changes",
        "You have unsaved changes. Save before closing?",
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No
        | QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Yes,
    )


class F8CodeEditorDialog(QtWidgets.QDialog):
    code_saved = QtCore.Signal(str)

    def __init__(self, parent=None, *, title: str, code: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._close_on_save = True

        self._edit = QtWidgets.QPlainTextEdit()
        self._edit.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        try:
            font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            self._edit.setFont(font)
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._edit.setPlainText(str(code or ""))

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_save_clicked)
        buttons.rejected.connect(self.reject)
        self._save_button = buttons.button(QtWidgets.QDialogButtonBox.Save)
        self._save_button.setEnabled(False)

        self._last_saved_code = str(code or "")
        self._edit.textChanged.connect(self._on_text_changed)

        self._save_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self._save_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._save_shortcut.activated.connect(self._on_save_clicked)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._edit, 1)
        layout.addWidget(buttons)

        self.resize(900, 650)

    def code(self) -> str:
        return str(self._edit.toPlainText() or "")

    def _on_save_clicked(self) -> None:
        if not self._save_button.isEnabled():
            return
        text = self.code()
        self.code_saved.emit(text)
        self._last_saved_code = text
        self._save_button.setEnabled(False)
        if self._close_on_save:
            self.accept()

    def _on_text_changed(self) -> None:
        self._save_button.setEnabled(self.code() != self._last_saved_code)

    def set_close_on_save(self, close_on_save: bool) -> None:
        self._close_on_save = bool(close_on_save)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if not self._save_button.isEnabled():
            event.accept()
            return
        answer = _ask_save_before_close(self)
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            text = self.code()
            self.code_saved.emit(text)
            self._last_saved_code = text
            self._save_button.setEnabled(False)
            event.accept()
            return
        if answer == QtWidgets.QMessageBox.StandardButton.No:
            event.accept()
            return
        event.ignore()


class _EditorUiBridge(QtCore.QObject):
    dirty_changed = QtCore.Signal(bool)
    save_requested = QtCore.Signal()
    close_requested = QtCore.Signal()

    @QtCore.Slot(bool)
    def notify_dirty(self, dirty: bool) -> None:
        self.dirty_changed.emit(bool(dirty))

    @QtCore.Slot()
    def request_save(self) -> None:
        self.save_requested.emit()

    @QtCore.Slot()
    def request_close(self) -> None:
        self.close_requested.emit()


class _PythonEditorAssistBridge(QtCore.QObject):
    completion_ready = QtCore.Signal(str, object)
    hover_ready = QtCore.Signal(str, object)

    @QtCore.Slot(str, str, int, int)
    def request_completions(self, request_id: str, code: str, line: int, column: int) -> None:
        result: list[dict[str, str]] = []
        try:
            import jedi  # type: ignore[import-not-found]

            script = jedi.Script(code=str(code or ""))
            completions = script.complete(line=max(1, int(line)), column=max(0, int(column)))
            for item in completions[:200]:
                try:
                    detail = str(item.description or "")
                except Exception:
                    detail = ""
                try:
                    item_type = str(item.type or "")
                except Exception:
                    item_type = ""
                result.append(
                    {
                        "label": str(item.name or ""),
                        "insertText": str(item.name or ""),
                        "detail": detail,
                        "kind": item_type,
                    }
                )
        except Exception:
            result = []
        self.completion_ready.emit(str(request_id), result)

    @QtCore.Slot(str, str, int, int)
    def request_hover(self, request_id: str, code: str, line: int, column: int) -> None:
        result: dict[str, str] = {}
        try:
            import jedi  # type: ignore[import-not-found]

            script = jedi.Script(code=str(code or ""))
            names = script.help(line=max(1, int(line)), column=max(0, int(column)))
            if names:
                desc = ""
                try:
                    desc = str(names[0].description or "")
                except Exception:
                    desc = ""
                doc = ""
                try:
                    doc = str(names[0].docstring(raw=True) or "")
                except Exception:
                    doc = ""
                result = {"description": desc, "docstring": doc}
        except Exception:
            result = {}
        self.hover_ready.emit(str(request_id), result)


class F8MonacoEditorDialog(QtWidgets.QDialog):
    """
    Monaco-based editor dialog (syntax highlighting, modern keybindings).

    Monaco assets can be loaded from:
    - `F8_MONACO_BASE_URL` (recommended for packaged/offline builds)
    - CDN fallback (default) for dev
    """

    code_saved = QtCore.Signal(str)

    def __init__(self, parent=None, *, title: str, code: str, language: str = "python"):
        super().__init__(parent)
        self.setWindowTitle(str(title or "Edit Code"))
        self._code: str = str(code or "")
        self._dirty: bool = False
        self._close_on_save: bool = True
        self._language: str = str(language or "plaintext").strip() or "plaintext"
        self._python_assist_enabled: bool = self._is_python_assist_enabled()

        from PySide6 import QtWebChannel, QtWebEngineWidgets  # type: ignore[import-not-found]

        self._view = QtWebEngineWidgets.QWebEngineView(self)
        self._view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        self._ui_bridge = _EditorUiBridge(self)
        self._assist_bridge: _PythonEditorAssistBridge | None = None
        self._web_channel: Any = QtWebChannel.QWebChannel(self._view.page())
        self._web_channel.registerObject("f8EditorUi", self._ui_bridge)
        if self._python_assist_enabled and self._language.lower() == "python":
            self._assist_bridge = _PythonEditorAssistBridge(self)
            self._web_channel.registerObject("pyAssist", self._assist_bridge)
        self._view.page().setWebChannel(self._web_channel)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_save_clicked)  # type: ignore[attr-defined]
        buttons.rejected.connect(self.reject)  # type: ignore[attr-defined]
        self._save_button = buttons.button(QtWidgets.QDialogButtonBox.Save)
        self._save_button.setEnabled(False)

        self._ui_bridge.dirty_changed.connect(self._on_dirty_changed)  # type: ignore[attr-defined]
        self._ui_bridge.save_requested.connect(self._on_save_clicked)  # type: ignore[attr-defined]
        self._ui_bridge.close_requested.connect(self.close)  # type: ignore[attr-defined]

        self._save_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self._save_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._save_shortcut.activated.connect(self._on_save_clicked)
        self._close_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Esc"), self)
        self._close_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._close_shortcut.activated.connect(self.close)

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
        initial = {
            "code": self._code,
            "language": self._language,
            "theme": "vs-dark",
            "pythonAssistEnabled": bool(self._python_assist_enabled),
        }
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
    <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
    <script src="{base}/vs/loader.js"></script>
    <script>
      window._f8_editor = null;
      window._f8_editorUi = null;
      window._f8_pyAssist = null;
      window._f8_pendingCompletions = Object.create(null);
      window._f8_pendingHovers = Object.create(null);
      window._f8_lastDirty = false;
      window._f8_savedValue = '';
      window._f8_getValue = function() {{
        try {{
          if (!window._f8_editor) return "";
          return window._f8_editor.getValue();
        }} catch (e) {{
          return "";
        }}
      }};
      window._f8_isDirty = function() {{
        try {{
          if (!window._f8_editor) return false;
          return window._f8_editor.getValue() !== String(window._f8_savedValue || '');
        }} catch (e) {{
          return false;
        }}
      }};
      window._f8_notifyDirty = function() {{
        try {{
          const dirty = Boolean(window._f8_isDirty());
          if (dirty === window._f8_lastDirty) return;
          window._f8_lastDirty = dirty;
          if (window._f8_editorUi && window._f8_editorUi.notify_dirty) {{
            window._f8_editorUi.notify_dirty(dirty);
          }}
        }} catch (e) {{
        }}
      }};
      window._f8_markSaved = function() {{
        try {{
          window._f8_savedValue = window._f8_getValue();
          window._f8_lastDirty = false;
          if (window._f8_editorUi && window._f8_editorUi.notify_dirty) {{
            window._f8_editorUi.notify_dirty(false);
          }}
        }} catch (e) {{
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
        window._f8_savedValue = String(init.code || '');
        window._f8_lastDirty = false;
        window._f8_editor.onDidChangeModelContent(function() {{
          window._f8_notifyDirty();
        }});
        window._f8_editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, function() {{
          if (window._f8_editorUi && window._f8_editorUi.request_save) {{
            window._f8_editorUi.request_save();
          }}
        }});
        window._f8_editor.addCommand(monaco.KeyCode.Escape, function() {{
          if (window._f8_editorUi && window._f8_editorUi.request_close) {{
            window._f8_editorUi.request_close();
          }}
        }});

        function _monacoKind(kind) {{
          const k = String(kind || '').toLowerCase();
          if (k === 'function' || k === 'method') return monaco.languages.CompletionItemKind.Function;
          if (k === 'class') return monaco.languages.CompletionItemKind.Class;
          if (k === 'module') return monaco.languages.CompletionItemKind.Module;
          if (k === 'property') return monaco.languages.CompletionItemKind.Property;
          if (k === 'param') return monaco.languages.CompletionItemKind.Variable;
          if (k === 'keyword') return monaco.languages.CompletionItemKind.Keyword;
          return monaco.languages.CompletionItemKind.Text;
        }}

        function _setupPythonAssist(channel) {{
          if (String(init.language || '').toLowerCase() !== 'python') return;
          if (!Boolean(init.pythonAssistEnabled)) return;
          const assist = channel && channel.objects ? channel.objects.pyAssist : null;
          window._f8_pyAssist = assist || null;
          if (!assist) return;

          assist.completion_ready.connect(function(requestId, items) {{
            const id = String(requestId || '');
            const resolver = window._f8_pendingCompletions[id];
            if (!resolver) return;
            delete window._f8_pendingCompletions[id];
            const src = Array.isArray(items) ? items : [];
            const out = [];
            for (const item of src) {{
              const label = String((item && item.label) || '');
              if (!label) continue;
              const insertText = String((item && item.insertText) || label);
              const detail = String((item && item.detail) || '');
              const kind = _monacoKind((item && item.kind) || '');
              out.push({{ label, insertText, detail, kind }});
            }}
            resolver({{ suggestions: out }});
          }});

          assist.hover_ready.connect(function(requestId, payload) {{
            const id = String(requestId || '');
            const resolver = window._f8_pendingHovers[id];
            if (!resolver) return;
            delete window._f8_pendingHovers[id];
            const obj = payload || {{}};
            const description = String(obj.description || '');
            const doc = String(obj.docstring || '');
            const blocks = [];
            if (description) blocks.push({{ value: '```python\\n' + description + '\\n```' }});
            if (doc) blocks.push({{ value: doc }});
            resolver(blocks.length ? {{ contents: blocks }} : null);
          }});

          monaco.languages.registerCompletionItemProvider('python', {{
            triggerCharacters: ['.', '_'],
            provideCompletionItems: function(model, position) {{
              return new Promise(function(resolve) {{
                try {{
                  const id = String(crypto.randomUUID ? crypto.randomUUID() : Math.random());
                  window._f8_pendingCompletions[id] = resolve;
                  const code = model.getValue();
                  assist.request_completions(id, code, Number(position.lineNumber), Number(position.column - 1));
                  setTimeout(function() {{
                    if (window._f8_pendingCompletions[id]) {{
                      delete window._f8_pendingCompletions[id];
                      resolve({{ suggestions: [] }});
                    }}
                  }}, 350);
                }} catch (e) {{
                  resolve({{ suggestions: [] }});
                }}
              }});
            }}
          }});

          monaco.languages.registerHoverProvider('python', {{
            provideHover: function(model, position) {{
              return new Promise(function(resolve) {{
                try {{
                  const id = String(crypto.randomUUID ? crypto.randomUUID() : Math.random());
                  window._f8_pendingHovers[id] = resolve;
                  const code = model.getValue();
                  assist.request_hover(id, code, Number(position.lineNumber), Number(position.column - 1));
                  setTimeout(function() {{
                    if (window._f8_pendingHovers[id]) {{
                      delete window._f8_pendingHovers[id];
                      resolve(null);
                    }}
                  }}, 350);
                }} catch (e) {{
                  resolve(null);
                }}
              }});
            }}
          }});
        }}

        if (typeof QWebChannel !== 'undefined' && window.qt && qt.webChannelTransport) {{
          new QWebChannel(qt.webChannelTransport, function(channel) {{
            window._f8_editorUi = channel.objects.f8EditorUi || null;
            window._f8_notifyDirty();
            _setupPythonAssist(channel);
          }});
        }}
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
        if not self._dirty:
            return
        self._save_current(close_after=self._close_on_save)

    def _save_current(self, *, close_after: bool) -> None:
        try:
            page = self._view.page()
        except Exception:
            page = None
        if page is None:
            if close_after:
                self.accept()
            return

        def _on_value(value: Any) -> None:
            try:
                self._code = "" if value is None else str(value)
            except Exception:
                self._code = ""
            self._set_dirty(False)
            self.code_saved.emit(self._code)
            if close_after:
                self.accept()

        try:
            page.runJavaScript("window._f8_getValue && window._f8_getValue();", _on_value)  # type: ignore[call-arg]
            page.runJavaScript("window._f8_markSaved && window._f8_markSaved();")
        except (AttributeError, RuntimeError, TypeError):
            pass

    @staticmethod
    def _is_python_assist_enabled() -> bool:
        raw = str(os.environ.get("F8_MONACO_PY_ASSIST") or "").strip().lower()
        return raw in {"1", "true", "yes", "on", "enable", "enabled"}

    @QtCore.Slot(bool)
    def _on_dirty_changed(self, dirty: bool) -> None:
        self._set_dirty(bool(dirty))

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = bool(dirty)
        self._save_button.setEnabled(self._dirty)

    def set_close_on_save(self, close_on_save: bool) -> None:
        self._close_on_save = bool(close_on_save)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        if not self._dirty:
            event.accept()
            return
        answer = _ask_save_before_close(self)
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._save_current(close_after=True)
            event.ignore()
            return
        if answer == QtWidgets.QMessageBox.StandardButton.No:
            event.accept()
            return
        event.ignore()


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


def open_code_editor_window(
    parent: QtWidgets.QWidget | None,
    *,
    title: str,
    code: str,
    language: str,
    on_saved: Callable[[str], None],
) -> QtWidgets.QDialog:
    dlg: QtWidgets.QDialog
    try:
        # Always create as a top-level window (no Qt parent) so it behaves as an
        # independent editor window in the OS window manager/task switcher.
        dlg = F8MonacoEditorDialog(None, title=title, code=code, language=language)
    except Exception:
        dlg = F8CodeEditorDialog(None, title=title, code=code)

    dlg.setModal(False)
    dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
    dlg.setWindowFlag(QtCore.Qt.WindowType.Window, True)
    dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    if isinstance(dlg, F8MonacoEditorDialog):
        dlg.set_close_on_save(False)
        dlg.code_saved.connect(on_saved)  # type: ignore[arg-type]
    elif isinstance(dlg, F8CodeEditorDialog):
        dlg.set_close_on_save(False)
        dlg.code_saved.connect(on_saved)  # type: ignore[arg-type]

    # Best-effort initial placement near caller without making it a child window.
    if parent is not None:
        try:
            anchor = parent.window() if parent.window() is not None else parent
            center = anchor.frameGeometry().center()
            frame = dlg.frameGeometry()
            frame.moveCenter(center)
            dlg.move(frame.topLeft())
        except (AttributeError, RuntimeError, TypeError):
            pass

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


class F8CodePropWidget(QtWidgets.QWidget):
    """
    Read-only preview with an "Edit..." button that opens a code editor dialog.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, title: str = "Edit Code"):
        super().__init__(parent)
        self._name = ""
        self._value = ""
        self._title = str(title or "Edit Code")
        self._editor_window: QtWidgets.QDialog | None = None

        self._preview = QtWidgets.QLineEdit()
        self._preview.setReadOnly(True)
        self._preview.setClearButtonEnabled(False)

        self._btn = QtWidgets.QPushButton("Edit...")
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
                preview = f"{preview} - {head[:80]}"
        self._preview.setText(preview)

    def _on_edit_clicked(self) -> None:
        if self._editor_window is not None:
            try:
                self._editor_window.raise_()
                self._editor_window.activateWindow()
                return
            except Exception:
                self._editor_window = None

        def _on_saved(updated: str) -> None:
            self.set_value(updated)
            self.value_changed.emit(self.get_name(), updated)

        try:
            dlg = open_code_editor_window(self, title=self._title, code=self.get_value(), language="python", on_saved=_on_saved)
            self._editor_window = dlg
            dlg.destroyed.connect(self._on_editor_destroyed)  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError):
            return

    @QtCore.Slot()
    def _on_editor_destroyed(self) -> None:
        self._editor_window = None


class F8CodeButtonPropWidget(QtWidgets.QWidget):
    """
    A single "Edit..." button that opens a code editor dialog.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, title: str = "Edit Code", language: str = "python"):
        super().__init__(parent)
        self._name = ""
        self._value = ""
        self._title = str(title or "Edit Code")
        self._language = str(language or "plaintext").strip() or "plaintext"
        self._editor_window: QtWidgets.QDialog | None = None

        self._btn = QtWidgets.QPushButton("Edit...")
        try:
            self._btn.setIcon(qta.icon("fa5s.code", color="white"))
        except (AttributeError, RuntimeError, TypeError):
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

    def set_read_only(self, read_only: bool) -> None:
        self._btn.setEnabled(not bool(read_only))

    def _on_edit_clicked(self) -> None:
        if self._editor_window is not None:
            try:
                self._editor_window.raise_()
                self._editor_window.activateWindow()
                return
            except Exception:
                self._editor_window = None

        def _on_saved(updated: str) -> None:
            self.set_value(updated)
            self.value_changed.emit(self.get_name(), updated)

        try:
            dlg = open_code_editor_window(
                self,
                title=self._title,
                code=self.get_value(),
                language=self._language,
                on_saved=_on_saved,
            )
            self._editor_window = dlg
            dlg.destroyed.connect(self._on_editor_destroyed)  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError):
            return

    @QtCore.Slot()
    def _on_editor_destroyed(self) -> None:
        self._editor_window = None


class F8InlineCodePropWidget(QtWidgets.QPlainTextEdit):
    """
    Inline multiline editor used for lightweight expressions (`uiControl=code_inline`).

    Emits `value_changed` on focus-out and on Ctrl+Enter.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, language: str = "plaintext"):
        super().__init__(parent)
        self._name: str = ""
        self._prev_text: str = ""
        self._language = str(language or "plaintext").strip().lower() or "plaintext"

        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        try:
            font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            self.setFont(font)
        except (AttributeError, RuntimeError, TypeError):
            pass
        self.setMinimumHeight(44)
        self.setMaximumHeight(96)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def focusInEvent(self, event):  # type: ignore[override]
        super().focusInEvent(event)
        self._prev_text = self.toPlainText()

    def focusOutEvent(self, event):  # type: ignore[override]
        super().focusOutEvent(event)
        self._emit_if_changed()

    def keyPressEvent(self, event):  # type: ignore[override]
        try:
            if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and bool(
                event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
            ):
                self._emit_if_changed(force=True)
                event.accept()
                return
        except (AttributeError, RuntimeError, TypeError):
            pass
        super().keyPressEvent(event)

    def _emit_if_changed(self, *, force: bool = False) -> None:
        text = str(self.toPlainText() or "")
        if not force and text == self._prev_text:
            return
        self._prev_text = text
        self.value_changed.emit(self.get_name(), text)

    def set_value(self, value: Any) -> None:
        with QtCore.QSignalBlocker(self):
            self.setPlainText("" if value is None else str(value))
        self._prev_text = self.toPlainText()


class F8WrapLinePropWidget(QtWidgets.QPlainTextEdit):
    """
    Single-line editor that wraps long text.

    Intended for short expressions that must not contain newlines, but can be
    visually wrapped to fit the node width.

    Emits `value_changed` on focus-out and on Enter/Ctrl+Enter.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, language: str = "plaintext"):
        super().__init__(parent)
        self._name: str = ""
        self._prev_text: str = ""
        self._language = str(language or "plaintext").strip().lower() or "plaintext"

        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        try:
            font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            self.setFont(font)
        except (AttributeError, RuntimeError, TypeError):
            pass
        self.document().setDocumentMargin(4.0)

        self.setMinimumHeight(38)
        self.setMaximumHeight(64)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    @staticmethod
    def _normalize(value: str) -> str:
        s = str(value or "")
        if "\n" not in s and "\r" not in s:
            return s
        parts = [p.strip() for p in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        return " ".join([p for p in parts if p]).strip()

    def focusInEvent(self, event):  # type: ignore[override]
        super().focusInEvent(event)
        self._prev_text = str(self.toPlainText() or "")

    def focusOutEvent(self, event):  # type: ignore[override]
        super().focusOutEvent(event)
        self._emit_if_changed()

    def keyPressEvent(self, event):  # type: ignore[override]
        try:
            is_enter = event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)
            if is_enter:
                # Never insert newlines. Treat Enter as commit.
                self._emit_if_changed(force=True)
                try:
                    self.clearFocus()
                except RuntimeError:
                    pass
                event.accept()
                return
        except (AttributeError, RuntimeError, TypeError):
            pass
        super().keyPressEvent(event)

    def insertFromMimeData(self, source: QtCore.QMimeData) -> None:  # type: ignore[override]
        try:
            txt = ""
            if source is not None and source.hasText():
                txt = self._normalize(str(source.text() or ""))
            if txt:
                self.textCursor().insertText(txt)
            return
        except Exception:
            return super().insertFromMimeData(source)

    def _emit_if_changed(self, *, force: bool = False) -> None:
        text = self._normalize(str(self.toPlainText() or ""))
        if text != str(self.toPlainText() or ""):
            with QtCore.QSignalBlocker(self):
                self.setPlainText(text)
        if not force and text == self._prev_text:
            return
        self._prev_text = text
        self.value_changed.emit(self.get_name(), text)

    def set_value(self, value: Any) -> None:
        text = self._normalize("" if value is None else str(value))
        with QtCore.QSignalBlocker(self):
            self.setPlainText(text)
        self._prev_text = text


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
    value_changing = QtCore.Signal(str, object)

    def __init__(self, parent=None, *, data_type: type = float):
        super().__init__(parent)
        self._name = ""
        self._data_type = data_type
        self._min: float | None = None
        self._max: float | None = None
        self._scrub_enabled = True
        self._scrub_base_step: float | None = None
        self._scrub_active = False
        self._scrub_start_global_x = 0.0
        self._scrub_start_value = 0.0
        self._scrub_start_text = ""
        self._base_tooltip = ""
        self.setMinimumWidth(120)
        self._update_validator()
        self._refresh_tooltip()
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
        self._update_validator()

    def set_max(self, v) -> None:
        try:
            self._max = float(v)
        except Exception:
            self._max = None
        self._update_validator()

    def _update_validator(self) -> None:
        if self._data_type is int:
            vmin = int(self._min) if self._min is not None else -(2**31)
            vmax = int(self._max) if self._max is not None else (2**31 - 1)
            self.setValidator(QtGui.QIntValidator(vmin, vmax, self))
            return
        vmin = float(self._min) if self._min is not None else -1.0e18
        vmax = float(self._max) if self._max is not None else 1.0e18
        dv = QtGui.QDoubleValidator(vmin, vmax, 6, self)
        try:
            dv.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        except (AttributeError, RuntimeError, TypeError):
            pass
        self.setValidator(dv)

    def set_scrub_enabled(self, enabled: bool) -> None:
        self._scrub_enabled = bool(enabled)
        self._refresh_tooltip()

    def set_scrub_base_step(self, step: float | None) -> None:
        if step is None:
            self._scrub_base_step = None
            return
        try:
            out = abs(float(step))
        except (TypeError, ValueError):
            self._scrub_base_step = None
            return
        if out <= 0.0:
            self._scrub_base_step = None
            return
        self._scrub_base_step = out

    def setToolTip(self, text: str) -> None:  # type: ignore[override]
        self._base_tooltip = str(text or "").strip()
        self._refresh_tooltip()

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
            with QtCore.QSignalBlocker(self):
                self.setText("")
            return
        with QtCore.QSignalBlocker(self):
            self.setText(str(value))

    def _emit_value(self) -> None:
        v = self.get_value()
        if v is None and str(self.text() or "").strip() != "":
            # invalid -> keep focus and don't emit.
            return
        self.value_changed.emit(self.get_name(), v)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        is_middle_drag = bool(event.button() == QtCore.Qt.MiddleButton)
        if is_middle_drag and self._scrub_enabled and self.isEnabled() and not self.isReadOnly():
            self._scrub_begin(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._scrub_active:
            self._scrub_update(event, commit=False)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._scrub_active and event.button() == QtCore.Qt.MiddleButton:
            self._scrub_update(event, commit=True)
            self._scrub_end()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if self._scrub_active and event.key() == QtCore.Qt.Key_Escape:
            with QtCore.QSignalBlocker(self):
                self.setText(self._scrub_start_text)
            self._scrub_end()
            event.accept()
            return
        super().keyPressEvent(event)

    def _scrub_begin(self, event: QtGui.QMouseEvent) -> None:
        self._scrub_active = True
        self._scrub_start_global_x = float(event.globalPosition().x())
        self._scrub_start_text = str(self.text() or "")
        current = self.get_value()
        self._scrub_start_value = 0.0 if current is None else float(current)
        self.setCursor(QtCore.Qt.SizeHorCursor)
        self.grabMouse()
        self.setFocus(QtCore.Qt.MouseFocusReason)

    def _scrub_end(self) -> None:
        self._scrub_active = False
        self.unsetCursor()
        self.releaseMouse()

    def _scrub_update(self, event: QtGui.QMouseEvent, *, commit: bool) -> None:
        dx = float(event.globalPosition().x()) - self._scrub_start_global_x
        step = self._resolve_scrub_step()
        mult = self._resolve_scrub_multiplier(event.modifiers())
        candidate = self._scrub_start_value + dx * step * mult
        out = self._coerce_value(candidate)
        with QtCore.QSignalBlocker(self):
            self.setText(self._format_value(out))
        if commit:
            self.value_changed.emit(self.get_name(), out)
        else:
            self.value_changing.emit(self.get_name(), out)

    def _resolve_scrub_step(self) -> float:
        if self._scrub_base_step is not None:
            step = max(1e-12, float(self._scrub_base_step))
            if self._data_type is int:
                return max(1.0, step)
            return step
        magnitude = max(abs(float(self._scrub_start_value)), 1.0)
        exponent = math.floor(math.log10(magnitude))
        step = math.pow(10.0, float(exponent)) * 0.01
        if self._data_type is int:
            return max(1.0, step)
        return max(1e-12, step)

    @staticmethod
    def _resolve_scrub_multiplier(modifiers: QtCore.Qt.KeyboardModifiers) -> float:
        has_shift = bool(modifiers & QtCore.Qt.ShiftModifier)
        has_ctrl = bool(modifiers & QtCore.Qt.ControlModifier)
        if has_shift and has_ctrl:
            return 1.0
        if has_shift:
            return 0.1
        if has_ctrl:
            return 10.0
        return 1.0

    def _coerce_value(self, v: float) -> float | int:
        out = float(v)
        if self._min is not None and out < self._min:
            out = float(self._min)
        if self._max is not None and out > self._max:
            out = float(self._max)
        if self._data_type is int:
            return int(round(out))
        return float(out)

    def _format_value(self, v: float | int) -> str:
        if self._data_type is int:
            return str(int(v))
        return ("{:.6f}".format(float(v))).rstrip("0").rstrip(".")

    def _refresh_tooltip(self) -> None:
        hint = "Middle-Drag to scrub" if self._scrub_enabled else ""
        if self._base_tooltip and hint:
            text = f"{self._base_tooltip}\n{hint}"
        elif self._base_tooltip:
            text = self._base_tooltip
        else:
            text = hint
        super().setToolTip(text)
