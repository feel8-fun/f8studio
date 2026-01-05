from __future__ import annotations

from typing import Any

from qtpy import QtCore, QtWidgets

from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from .generic import GenericNode


class _EditorLogWidget(NodeBaseWidget):
    def __init__(self, parent: Any, name: str, *, max_lines: int = 200) -> None:
        super().__init__(parent, name, "")

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QtWidgets.QLabel("Log")
        title.setStyleSheet("color: rgba(200,200,200,160); font-size: 8pt;")
        layout.addWidget(title, 0)

        edit = QtWidgets.QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setUndoRedoEnabled(False)
        edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        edit.setFocusPolicy(QtCore.Qt.NoFocus)
        edit.setMinimumWidth(260)
        edit.setFixedHeight(110)
        try:
            edit.document().setMaximumBlockCount(int(max_lines))
        except Exception:
            pass
        edit.setStyleSheet(
            "QPlainTextEdit {"
            "  background: rgba(20,20,20,120);"
            "  border: 1px solid rgba(255,255,255,18);"
            "  color: rgba(230,230,230,210);"
            "  font-family: Consolas, 'Courier New', monospace;"
            "  font-size: 8pt;"
            "}"
        )
        layout.addWidget(edit, 1)

        self._edit = edit
        self.set_custom_widget(container)

    def append_line(self, text: str) -> None:
        try:
            self._edit.appendPlainText(text)
        except Exception:
            try:
                self._edit.appendPlainText(text)
            except Exception:
                pass
        try:
            sb = self._edit.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception:
            pass

    def clear(self) -> None:
        try:
            self._edit.setPlainText("")
        except Exception:
            pass


class EditorLogNode(GenericNode):  # type: ignore[misc]
    """
    Editor-only visualization node.

    Displays pulled data (cross-edge inputs) in a small text widget embedded
    on the node. The executor/poller lives in the editor process.
    """

    __identifier__ = "fun.feel8.op.renderer.editor"

    def __init__(self) -> None:
        super().__init__()
        self._ensure_log_widget()

    def _ensure_log_widget(self) -> None:
        w = getattr(self, "_editor_log_widget", None)
        if w is not None:
            return
        widget_name = "__editor_log__"
        w = _EditorLogWidget(self.view, widget_name)
        try:
            self.view.add_widget(w)
            self.view.draw_node()
        except Exception:
            pass
        setattr(self, "_editor_log_widget", w)

    def append_log(self, text: str) -> None:
        self._ensure_log_widget()
        w = getattr(self, "_editor_log_widget", None)
        if w is None:
            return
        try:
            w.append_line(text)
        except Exception:
            pass
        try:
            self.view.draw_node()
        except Exception:
            pass
