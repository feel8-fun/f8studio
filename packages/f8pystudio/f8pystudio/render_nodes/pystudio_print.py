from __future__ import annotations

import json
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets
from NodeGraphQt.nodes.base_node import NodeBaseWidget

from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from ..nodegraph.viz_operator_nodeitem import F8StudioVizOperatorNodeItem
from ..ui_bus import UiCommand

_STATE_UI_UPDATE = "uiUpdate"
_STATE_UI_WRAP = "uiWrap"
_WIDGET_NAME = "__print_preview"


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
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

        try:
            fm = QtGui.QFontMetricsF(self._text.font())
            self._text.setTabStopDistance(float(fm.horizontalAdvance(" ") * 4.0))
        except (AttributeError, RuntimeError, TypeError, ValueError):
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

        self.setMinimumWidth(220)
        self.setMinimumHeight(120)
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
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _on_update_toggled(self, enabled: bool) -> None:
        if not enabled:
            return
        pending = self._pending_text
        self._pending_text = None
        if pending is not None:
            try:
                self._text.setPlainText(pending)
            except (AttributeError, RuntimeError, TypeError):
                pass

    def _copy_to_clipboard(self) -> None:
        txt = ""
        try:
            if not self._update.isChecked() and self._pending_text is not None:
                txt = self._pending_text
            else:
                txt = self._text.toPlainText()
        except (AttributeError, RuntimeError, TypeError):
            txt = ""
        try:
            cb = QtWidgets.QApplication.clipboard()
            if cb is not None:
                cb.setText(str(txt or ""))
        except (AttributeError, RuntimeError, TypeError):
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
        except (TypeError, ValueError):
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
        except (AttributeError, RuntimeError, TypeError):
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
        except (TypeError, ValueError):
            follow_tail = False
        try:
            self._text.setPlainText(s)
        except (AttributeError, RuntimeError, TypeError):
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
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            if hbar is not None and h_val is not None and h_max is not None:
                new_hmax = int(hbar.maximum())
                delta_from_right = int(h_max) - int(h_val)
                hbar.setValue(max(0, new_hmax - delta_from_right))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass


class _PrintPreviewWidget(NodeBaseWidget):
    """
    Node embedded widget for the Print node.

    - Read-only text area for preview.
    - Wrap checkbox persisted in the session.
    """

    def __init__(
        self,
        parent=None,
        name: str = _WIDGET_NAME,
        label: str = "",
        *,
        on_wrap_toggled: Callable[[bool], None] | None = None,
        on_update_toggled: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, label=label)
        self._pane = _PrintPreviewPane()
        self.set_custom_widget(self._pane)

        self._block = False
        self._on_wrap_toggled_cb = on_wrap_toggled
        self._on_update_toggled_cb = on_update_toggled
        self._pane.wrap_checkbox().toggled.connect(self.on_value_changed)
        self._pane.wrap_checkbox().toggled.connect(self._on_wrap_toggled)
        self._pane.update_checkbox().toggled.connect(self.on_value_changed)
        self._pane.update_checkbox().toggled.connect(self._on_update_toggled)

    def get_value(self) -> object:
        return {"wrap": bool(self._pane.wrap()), "update": bool(self._pane.update_enabled())}

    def set_value(self, value: object) -> None:
        _ = value

    def set_wrap_enabled(self, enabled: bool) -> None:
        try:
            self._block = True
            self._pane.set_wrap(enabled)
        finally:
            self._block = False

    def set_update_enabled(self, enabled: bool) -> None:
        try:
            self._block = True
            self._pane.set_update_enabled(enabled)
        finally:
            self._block = False

    def on_value_changed(self, *args, **kwargs):
        if self._block:
            return
        return super().on_value_changed(*args, **kwargs)

    def _on_wrap_toggled(self, enabled: bool) -> None:
        if self._block:
            return
        cb = self._on_wrap_toggled_cb
        if cb is None:
            return
        cb(bool(enabled))

    def _on_update_toggled(self, enabled: bool) -> None:
        if self._block:
            return
        cb = self._on_update_toggled_cb
        if cb is None:
            return
        cb(bool(enabled))

    def set_preview_text(self, text: str) -> None:
        self._pane.set_text(text)


class PyStudioPrintNode(F8StudioOperatorBaseNode):
    """
    Render node for `f8.print`.

    Adds a preview text area that can be updated by the editor refresh loop.
    """

    def __init__(self):
        super().__init__(qgraphics_item=F8StudioVizOperatorNodeItem)
        self.add_ephemeral_widget(
            _PrintPreviewWidget(
                self.view,
                name=_WIDGET_NAME,
                label="",
                on_wrap_toggled=self._on_wrap_toggled,
                on_update_toggled=self._on_update_toggled,
            )
        )
        self._sync_wrap_checkbox_from_state(default=True)
        self._sync_update_checkbox_from_state(default=True)

    def sync_from_spec(self) -> None:
        super().sync_from_spec()
        self._sync_wrap_checkbox_from_state(default=True)
        self._sync_update_checkbox_from_state(default=True)

    def set_property(self, name, value, push_undo=True):  # type: ignore[override]
        super().set_property(name, value, push_undo=push_undo)
        key = str(name or "").strip()
        if key == _STATE_UI_WRAP:
            self._sync_wrap_checkbox_from_state(default=bool(value))
        if key == _STATE_UI_UPDATE:
            self._sync_update_checkbox_from_state(default=bool(value))

    def _on_wrap_toggled(self, enabled: bool) -> None:
        self.set_state_bool(_STATE_UI_WRAP, bool(enabled))

    def _on_update_toggled(self, enabled: bool) -> None:
        self.set_state_bool(_STATE_UI_UPDATE, bool(enabled))

    def _sync_wrap_checkbox_from_state(self, *, default: bool) -> None:
        self.sync_bool_state_to_widget(
            state_name=_STATE_UI_WRAP,
            default=default,
            widget_name=_WIDGET_NAME,
            widget_type=_PrintPreviewWidget,
            apply_value=_PrintPreviewWidget.set_wrap_enabled,
        )

    def _sync_update_checkbox_from_state(self, *, default: bool) -> None:
        self.sync_bool_state_to_widget(
            state_name=_STATE_UI_UPDATE,
            default=default,
            widget_name=_WIDGET_NAME,
            widget_type=_PrintPreviewWidget,
            apply_value=_PrintPreviewWidget.set_update_enabled,
        )

    def _widget(self) -> _PrintPreviewWidget | None:
        return self.widget_by_name(_WIDGET_NAME, _PrintPreviewWidget)

    def set_preview(self, value: Any) -> None:
        try:
            txt = json.dumps(value, ensure_ascii=False, indent=1, default=str)
        except (TypeError, ValueError):
            txt = str(value)
        widget = self._widget()
        if widget is None:
            return
        widget.set_preview_text(txt)

    def apply_ui_command(self, cmd: UiCommand) -> None:
        if str(cmd.command) != "preview.update":
            return
        try:
            payload = dict(cmd.payload or {})
        except (AttributeError, TypeError, ValueError):
            return
        value = payload.get("value")
        self.set_preview(value)
