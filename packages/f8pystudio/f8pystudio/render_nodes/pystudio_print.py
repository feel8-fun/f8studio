from __future__ import annotations

import json
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets
from NodeGraphQt.nodes.base_node import NodeBaseWidget

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode


class _JsonHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, doc: QtGui.QTextDocument) -> None:
        super().__init__(doc)

        def _fmt(*, fg: QtGui.QColor | None = None, bold: bool = False) -> QtGui.QTextCharFormat:
            f = QtGui.QTextCharFormat()
            if fg is not None:
                f.setForeground(fg)
            if bold:
                f.setFontWeight(QtGui.QFont.Weight.Bold)
            return f

        self._rules: list[tuple[QtCore.QRegularExpression, QtGui.QTextCharFormat]] = [
            # String values.
            (QtCore.QRegularExpression(r"\"([^\"\\\\]|\\\\.)*\""), _fmt(fg=QtGui.QColor(140, 220, 160))),
            # Key strings: "key": (must come after string to override it).
            (QtCore.QRegularExpression(r"\"([^\"\\\\]|\\\\.)*\"(?=\\s*:)"), _fmt(fg=QtGui.QColor(120, 200, 255), bold=True)),
            # Numbers.
            (
                QtCore.QRegularExpression(r"\\b-?(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b"),
                _fmt(fg=QtGui.QColor(255, 190, 120)),
            ),
            # true/false/null
            (QtCore.QRegularExpression(r"\\b(true|false|null)\\b"), _fmt(fg=QtGui.QColor(200, 150, 255), bold=True)),
            # Punctuation.
            (QtCore.QRegularExpression(r"[\\{\\}\\[\\],:]"), _fmt(fg=QtGui.QColor(150, 150, 150))),
        ]

    def highlightBlock(self, text: str) -> None:
        for regex, fmt in self._rules:
            it = regex.globalMatch(text)
            while it.hasNext():
                m = it.next()
                start = m.capturedStart()
                length = m.capturedLength()
                if start >= 0 and length > 0:
                    self.setFormat(start, length, fmt)


class _PrintPreviewPane(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._copy = QtWidgets.QToolButton()
        self._copy.setText("Copy")
        self._copy.setToolTip("Copy preview to clipboard")
        self._copy.setAutoRaise(True)
        self._update = QtWidgets.QCheckBox("Update")
        self._update.setToolTip("When unchecked, pause live updates.")
        self._update.setChecked(True)
        self._wrap = QtWidgets.QCheckBox("Wrap")
        self._wrap.setChecked(True)

        self._text = QtWidgets.QTextEdit()
        self._text.setReadOnly(True)
        self._text.setAcceptRichText(False)
        self._text.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self._text.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._pending_text: str | None = None

        try:
            f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            f.setPointSize(max(8, int(f.pointSize())))
            self._text.setFont(f)
        except Exception:
            pass

        try:
            fm = QtGui.QFontMetricsF(self._text.font())
            self._text.setTabStopDistance(float(fm.horizontalAdvance(" ") * 4.0))
        except Exception:
            pass

        self._highlighter = _JsonHighlighter(self._text.document())

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._copy)
        top.addWidget(self._update)
        top.addStretch(1)
        top.addWidget(self._wrap)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self._text, 1)

        self.setMinimumWidth(260)
        self.setMinimumHeight(160)
        self._apply_wrap(True)
        self._apply_dark_style()
        self._copy.clicked.connect(self._copy_to_clipboard)
        self._update.toggled.connect(self._on_update_toggled)

    def _apply_dark_style(self) -> None:
        # Ensure readability on dark node themes.
        self.setStyleSheet(
            """
            QToolButton {
                color: rgb(225, 225, 225);
                background: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 45);
                padding: 2px 8px;
                border-radius: 3px;
            }
            QToolButton:hover { background: rgba(255, 255, 255, 35); }
            QToolButton:pressed { background: rgba(255, 255, 255, 25); }
            QCheckBox { color: rgb(225, 225, 225); }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border: 1px solid rgba(255, 255, 255, 90);
                background: rgba(0, 0, 0, 35);
                border-radius: 2px;
            }
            QCheckBox::indicator:checked { background: rgba(120, 200, 255, 90); }
            QTextEdit {
                color: rgb(225, 225, 225);
                background: rgba(0, 0, 0, 35);
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 4px;
            }
            """
        )

    def _apply_wrap(self, enabled: bool) -> None:
        try:
            self._text.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth if enabled else QtWidgets.QTextEdit.NoWrap)
        except Exception:
            pass

    def _on_update_toggled(self, enabled: bool) -> None:
        if not enabled:
            return
        pending = self._pending_text
        self._pending_text = None
        if pending is not None:
            try:
                self._text.setPlainText(pending)
            except Exception:
                pass

    def _copy_to_clipboard(self) -> None:
        txt = ""
        try:
            if not self._update.isChecked() and self._pending_text is not None:
                txt = self._pending_text
            else:
                txt = self._text.toPlainText()
        except Exception:
            txt = ""
        try:
            cb = QtWidgets.QApplication.clipboard()
            if cb is not None:
                cb.setText(str(txt or ""))
        except Exception:
            pass

    def set_wrap(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._wrap.setChecked(enabled)
        self._apply_wrap(enabled)

    def set_update_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._update.setChecked(enabled)

    def wrap(self) -> bool:
        return bool(self._wrap.isChecked())

    def update_enabled(self) -> bool:
        return bool(self._update.isChecked())

    def update_checkbox(self) -> QtWidgets.QCheckBox:
        return self._update

    def wrap_checkbox(self) -> QtWidgets.QCheckBox:
        return self._wrap

    def set_text(self, text: str) -> None:
        # Avoid pathological updates on huge payloads.
        try:
            s = str(text or "")
        except Exception:
            s = ""
        if len(s) > 50_000:
            s = s[:50_000] + "\n… (truncated)"
        if not self._update.isChecked():
            self._pending_text = s
            return
        self._pending_text = None
        vbar = None
        hbar = None
        try:
            vbar = self._text.verticalScrollBar()
            hbar = self._text.horizontalScrollBar()
        except Exception:
            vbar = None
            hbar = None
        v_val = vbar.value() if vbar is not None else None
        v_max = vbar.maximum() if vbar is not None else None
        h_val = hbar.value() if hbar is not None else None
        h_max = hbar.maximum() if hbar is not None else None
        follow_tail = False
        try:
            if v_val is not None and v_max is not None:
                follow_tail = (v_max - v_val) <= 2
        except Exception:
            follow_tail = False
        try:
            self._text.setPlainText(s)
        except Exception:
            pass
        # Restore scroll position: if the user was at the bottom, keep following tail;
        # otherwise keep the previous position (relative from bottom, best-effort).
        try:
            if vbar is not None:
                new_max = int(vbar.maximum())
                if follow_tail:
                    vbar.setValue(new_max)
                else:
                    if v_val is not None and v_max is not None:
                        delta_from_bottom = int(v_max) - int(v_val)
                        vbar.setValue(max(0, new_max - delta_from_bottom))
        except Exception:
            pass
        try:
            if hbar is not None and h_val is not None and h_max is not None:
                new_hmax = int(hbar.maximum())
                delta_from_right = int(h_max) - int(h_val)
                hbar.setValue(max(0, new_hmax - delta_from_right))
        except Exception:
            pass


class _PrintPreviewWidget(NodeBaseWidget):
    """
    Node embedded widget for the Print node.

    - Read-only text area for preview.
    - Wrap checkbox persisted in the session.
    """

    def __init__(self, parent=None, name: str = "__print_preview", label: str = "") -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _PrintPreviewPane()
        self.set_custom_widget(self._pane)

        self._block = False
        self._pane.wrap_checkbox().toggled.connect(self.on_value_changed)
        self._pane.update_checkbox().toggled.connect(self.on_value_changed)

    def get_value(self) -> object:
        return {"wrap": bool(self._pane.wrap()), "update": bool(self._pane.update_enabled())}

    def set_value(self, value: object) -> None:
        wrap = True
        update_enabled = True
        if isinstance(value, dict):
            wrap = bool(value.get("wrap", True))
            update_enabled = bool(value.get("update", True))
        elif isinstance(value, bool):
            wrap = bool(value)
        try:
            self._block = True
            self._pane.set_wrap(wrap)
            self._pane.set_update_enabled(update_enabled)
        finally:
            self._block = False

    def on_value_changed(self, *args, **kwargs):
        if getattr(self, "_block", False):
            return
        return super().on_value_changed(*args, **kwargs)

    def set_preview_text(self, text: str) -> None:
        self._pane.set_text(text)


class PyStudioPrintNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.print`.

    Adds a preview text area that can be updated by the editor refresh loop.
    """

    def __init__(self):
        super().__init__()
        try:
            self.add_custom_widget(_PrintPreviewWidget(self.view, name="__print_preview", label=""))
        except Exception:
            pass

    def set_preview(self, value: Any) -> None:
        try:
            txt = json.dumps(value, ensure_ascii=False, indent=1, default=str)
        except Exception:
            txt = str(value)
        try:
            w = self.get_widget("__print_preview")
            if w and hasattr(w, "set_preview_text"):
                w.set_preview_text(txt)
        except Exception:
            return

    def apply_ui_command(self, cmd: Any) -> None:
        try:
            if str(getattr(cmd, "command", "")) != "preview.update":
                return
            payload = getattr(cmd, "payload", {}) or {}
            value = payload.get("value")
        except Exception:
            return
        self.set_preview(value)
