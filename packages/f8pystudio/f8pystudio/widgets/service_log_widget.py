from __future__ import annotations

import re
from dataclasses import dataclass

from qtpy import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern[str]
    fmt: QtGui.QTextCharFormat


class _LogHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, doc: QtGui.QTextDocument) -> None:
        super().__init__(doc)
        self._rules: list[_Rule] = []

        def f(color: str, *, bold: bool = False) -> QtGui.QTextCharFormat:
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(QtGui.QColor(color))
            if bold:
                fmt.setFontWeight(QtGui.QFont.Weight.Bold)
            return fmt

        # Rough VSCode-like colors.
        self._rules.extend(
            [
                _Rule(re.compile(r"\b(ERROR|FATAL)\b"), f("#f44747", bold=True)),
                _Rule(re.compile(r"\b(WARN|WARNING)\b"), f("#cca700", bold=True)),
                _Rule(re.compile(r"\b(INFO)\b"), f("#4fc1ff")),
                _Rule(re.compile(r"\b(DEBUG|TRACE)\b"), f("#b5cea8")),
                _Rule(re.compile(r"\b(Traceback \(most recent call last\):)\b"), f("#dcdcaa", bold=True)),
                _Rule(re.compile(r"\b(Exception|Error|AssertionError|ValueError|TypeError|KeyError)\b"), f("#ff7b72")),
                _Rule(re.compile(r"File \"[^\"]+\", line \d+"), f("#9cdcfe")),
            ]
        )

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt naming)
        for rule in self._rules:
            for m in rule.pattern.finditer(text):
                start, end = m.span()
                self.setFormat(start, end - start, rule.fmt)


class ServiceLogView(QtWidgets.QPlainTextEdit):
    """
    Read-only log view with a dark theme and basic keyword highlighting.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        # Terminal-like: wrap at widget width (avoid horizontal scrollbar).
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        try:
            self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        except Exception:
            pass
        self.setMaximumBlockCount(5000)

        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setPointSize(10)
        self.setFont(font)

        # Dark terminal-ish palette.
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#1e1e1e"))
        pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#d4d4d4"))
        self.setPalette(pal)

        self._highlighter = _LogHighlighter(self.document())

    def append_line(self, line: str) -> None:
        line = str(line or "")
        # Avoid double newlines.
        if line.endswith("\n"):
            line = line[:-1]
        self.appendPlainText(line)


class ServiceLogDock(QtWidgets.QDockWidget):
    """
    Dock widget with per-service log tabs.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Service Logs", parent)
        self.setObjectName("ServiceLogsDock")

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(True)
        self._tabs.setTabsClosable(False)
        self.setWidget(self._tabs)

        self._views: dict[str, ServiceLogView] = {}

    def _ensure_tab(self, service_id: str) -> ServiceLogView:
        sid = str(service_id or "").strip() or "unknown"
        view = self._views.get(sid)
        if view is not None:
            return view
        view = ServiceLogView()
        self._views[sid] = view
        self._tabs.addTab(view, sid)
        return view

    @QtCore.Slot(str, str)
    def append(self, service_id: str, line: str) -> None:
        view = self._ensure_tab(service_id)
        view.append_line(line)
