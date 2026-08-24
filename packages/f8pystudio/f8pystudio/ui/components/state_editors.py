from __future__ import annotations

"""
Shared state-value controls used by multiple hosts.

This module holds reusable editors/widgets for state values that can be
embedded in the properties panel and, in some cases, reused by inline node UI.
"""

import json
import logging
import math
import weakref
from typing import Any, Callable

from qtpy import QtCore, QtGui, QtWidgets

from ...agents.graph_context import GraphContextSnapshot
from ...editor_assist.agent_context import EditorAgentContext
from ...editor_assist.agent_scope import EditorAgentScope
from ...editor_assist.session import EditorSessionKey
from ...editor_assist.workspace import EditorAssistContext
from ...ui.support.qt_lifecycle import qt_object_is_valid
from ...ui.support.ui_notifications import show_warning
from ...ui.support.ui_icons import StudioIcon, icon_for
from .controls import F8Dial, F8ImageB64Editor, F8MultiSelect, F8OptionCombo, F8RangeBar, F8Switch, F8ValueBar
from ..support.json_text_editor import attach_json_enhancements
from ..support.monaco_editor_host import open_code_editor_window

logger = logging.getLogger(__name__)
_QT_COMPAT_ERRORS = (AttributeError, RuntimeError, TypeError)
_QT_EVENT_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_NUMERIC_CONVERSION_ERRORS = (TypeError, ValueError)
_MIME_INSERT_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_CODE_EDITOR_PERSISTENCE_ERRORS = (Exception,)


class F8CodeButtonEditor(QtWidgets.QWidget):
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
        self._assist_context: EditorAssistContext | None = None
        self._assist_context_provider: Callable[[], EditorAssistContext | None] | None = None
        self._persisted_value_getter: Callable[[], str] | None = None
        self._persisted_value_setter: Callable[[str], bool | None] | None = None
        self._persisted_target_exists_provider: Callable[[], bool] | None = None
        self._editor_session_key: EditorSessionKey | None = None
        self._editor_agent_scope: EditorAgentScope | None = None
        self._agent_tools: tuple[object, ...] = ()
        self._agent_context_providers: tuple[object, ...] = ()
        self._graph_context_snapshot_provider: Callable[[], GraphContextSnapshot | None] | None = None
        self._retained_agent_dependencies: tuple[object, ...] = ()
        self._agent_sidebar_launcher: Callable[[EditorAgentContext], None] | None = None
        self._editor_window: QtWidgets.QDialog | None = None

        self._btn = QtWidgets.QPushButton("Edit...")
        self._btn.setIcon(icon_for(self._btn, StudioIcon.CODE))
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

    def set_title(self, title: str) -> None:
        self._title = str(title or "Edit Code")

    def set_persisted_value_getter(self, getter: Callable[[], str] | None) -> None:
        self._persisted_value_getter = getter

    def set_persisted_value_setter(self, setter: Callable[[str], bool | None] | None) -> None:
        self._persisted_value_setter = setter

    def set_persisted_target_exists_provider(self, provider: Callable[[], bool] | None) -> None:
        self._persisted_target_exists_provider = provider

    def set_editor_session_key(self, session_key: EditorSessionKey | None) -> None:
        self._editor_session_key = session_key

    def set_editor_agent_scope(self, scope: EditorAgentScope | None) -> None:
        self._editor_agent_scope = scope

    def set_agent_tools(self, tools: tuple[object, ...]) -> None:
        self._agent_tools = tuple(tools)

    def set_agent_context_providers(self, context_providers: tuple[object, ...]) -> None:
        self._agent_context_providers = tuple(context_providers)

    def set_graph_context_snapshot_provider(
        self,
        provider: Callable[[], GraphContextSnapshot | None] | None,
    ) -> None:
        self._graph_context_snapshot_provider = provider

    def set_retained_agent_dependencies(self, dependencies: tuple[object, ...]) -> None:
        self._retained_agent_dependencies = tuple(dependencies)

    def set_agent_sidebar_launcher(self, launcher: Callable[[EditorAgentContext], None] | None) -> None:
        self._agent_sidebar_launcher = launcher

    def set_editor_assist_context(self, context: EditorAssistContext | None) -> None:
        self._assist_context = context

    def set_editor_assist_context_provider(
        self,
        provider: Callable[[], EditorAssistContext | None] | None,
    ) -> None:
        self._assist_context_provider = provider

    def _on_edit_clicked(self) -> None:
        if self._editor_window is not None:
            try:
                self._editor_window.raise_()
                self._editor_window.activateWindow()
                return
            except _QT_COMPAT_ERRORS as exc:
                logger.debug("Discarding invalid code editor window for property '%s'", self.get_name(), exc_info=exc)
                self._editor_window = None

        initial_code = self.get_value()
        if self._persisted_value_getter is not None:
            try:
                initial_code = str(self._persisted_value_getter() or "")
            except _CODE_EDITOR_PERSISTENCE_ERRORS:
                logger.exception("Failed to load persisted code for property '%s'", self.get_name())
            else:
                self.set_value(initial_code)

        widget_ref = weakref.ref(self)
        prop_name = self.get_name()
        persisted_value_setter = self._persisted_value_setter
        persisted_target_exists_provider = self._persisted_target_exists_provider

        def _on_saved(updated: str) -> bool:
            updated_text = str(updated or "")
            if persisted_value_setter is not None:
                try:
                    saved = persisted_value_setter(updated_text)
                except _CODE_EDITOR_PERSISTENCE_ERRORS:
                    logger.exception("Failed to persist code for property '%s'", prop_name)
                    return False
                if saved is False:
                    return False

            widget = widget_ref()
            if widget is None or not qt_object_is_valid(widget):
                return True
            widget.set_value(updated_text)
            widget.value_changed.emit(widget.get_name(), updated_text)
            return True

        def _target_exists() -> bool:
            if persisted_target_exists_provider is None:
                return True
            try:
                return bool(persisted_target_exists_provider())
            except _CODE_EDITOR_PERSISTENCE_ERRORS:
                logger.exception("Failed to check persisted code target for property '%s'", prop_name)
                return False

        dlg = open_code_editor_window(
            self,
            title=self._title,
            code=initial_code,
            language=self._language,
            on_saved=_on_saved,
            target_exists_provider=_target_exists,
            assist_context=self._assist_context,
            assist_context_provider=self._assist_context_provider,
            session_key=self._editor_session_key,
            agent_scope=self._editor_agent_scope,
            agent_tools=self._agent_tools,
            agent_context_providers=self._agent_context_providers,
            graph_context_snapshot_provider=self._graph_context_snapshot_provider,
            retained_agent_dependencies=self._retained_agent_dependencies,
            agent_sidebar_launcher=self._agent_sidebar_launcher,
        )
        self._editor_window = dlg
        dlg.destroyed.connect(self._on_editor_destroyed)  # type: ignore[attr-defined]

    @QtCore.Slot()
    def _on_editor_destroyed(self) -> None:
        self._editor_window = None


class F8InlineCodeEditor(QtWidgets.QPlainTextEdit):
    """
    Inline editor used for lightweight expressions (`uiControl=wrapline[...]` is the public path).

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
        except _QT_COMPAT_ERRORS as exc:
            logger.debug("Failed to apply fixed-width font to inline code editor", exc_info=exc)
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
        except _QT_EVENT_ERRORS as exc:
            logger.debug("Inline code editor key handling failed; falling back to default handler", exc_info=exc)
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


class F8IncrementButtonEditor(QtWidgets.QPushButton):
    """
    Stateless trigger button backed by a numeric state value.

    Each click emits `current + 1`, which works well with state-dedupe semantics
    and downstream `state_trigger` nodes.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        title: str = "Trigger",
        data_type: type[int] | type[float] = int,
    ) -> None:
        super().__init__(str(title or "Trigger"), parent)
        self._name = ""
        self._value: object = 0 if data_type is int else 0.0
        self._read_only = False
        self._invalid_reason = ""
        self._context_tooltip = ""
        self._data_type: type[int] | type[float] = int if data_type is int else float
        self.setMinimumHeight(24)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.clicked.connect(self._on_clicked)  # type: ignore[attr-defined]

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_value(self, value: Any) -> None:
        coerced = self._coerce_value(value)
        if coerced is None:
            self._value = 0 if self._data_type is int else 0.0
            return
        self._value = coerced

    def get_value(self) -> object:
        return self._value

    def set_button_text(self, text: str) -> None:
        self.setText(str(text or "Trigger"))

    def set_read_only(self, read_only: bool) -> None:
        self._read_only = bool(read_only)
        self._refresh_enabled()

    def set_context_tooltip(self, tooltip: str) -> None:
        self._context_tooltip = str(tooltip or "").strip()
        self._refresh_tooltip()

    def set_invalid_reason(self, reason: str) -> None:
        self._invalid_reason = str(reason or "").strip()
        self._refresh_tooltip()
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        self.setEnabled((not self._read_only) and (not self._invalid_reason))

    def _refresh_tooltip(self) -> None:
        tip_parts: list[str] = []
        if self._context_tooltip:
            tip_parts.append(self._context_tooltip)
        if self._invalid_reason:
            tip_parts.append(self._invalid_reason)
        self.setToolTip("\n".join(tip_parts))

    def _on_clicked(self) -> None:
        if self._read_only or self._invalid_reason:
            return
        next_value = self._increment(self._value)
        self._value = next_value
        self.value_changed.emit(self.get_name(), next_value)

    def _increment(self, value: object) -> object:
        if self._data_type is float:
            current = self._coerce_value(value)
            if current is None:
                current = 0.0
            return float(current) + 1.0
        current_int = self._coerce_value(value)
        if current_int is None:
            current_int = 0
        return int(current_int) + 1

    def _coerce_value(self, value: Any) -> int | float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value) if self._data_type is int else float(value)
        try:
            if self._data_type is float:
                return float(value)
            return int(value)
        except (TypeError, ValueError):
            return None


class F8WrapLineEditor(QtWidgets.QPlainTextEdit):
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
        except _QT_COMPAT_ERRORS as exc:
            logger.debug("Failed to apply fixed-width font to wrapline editor", exc_info=exc)
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
                except RuntimeError as exc:
                    logger.debug("Wrapline editor failed to clear focus after commit", exc_info=exc)
                event.accept()
                return
        except _QT_EVENT_ERRORS as exc:
            logger.debug("Wrapline editor key handling failed; falling back to default handler", exc_info=exc)
        super().keyPressEvent(event)

    def insertFromMimeData(self, source: QtCore.QMimeData) -> None:  # type: ignore[override]
        try:
            txt = ""
            if source is not None and source.hasText():
                txt = self._normalize(str(source.text() or ""))
            if txt:
                self.textCursor().insertText(txt)
            return
        except _MIME_INSERT_ERRORS as exc:
            logger.debug("Wrapline editor MIME normalization failed; falling back to default insert", exc_info=exc)
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


class F8JsonValueEditor(QtWidgets.QTextEdit):
    """
    QTextEdit property widget that round-trips JSON values as python objects.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name: str | None = None
        self._prev_text = ""
        self._prev_value: Any = None
        self.setAcceptRichText(False)
        attach_json_enhancements(self, read_only=False)

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
        except json.JSONDecodeError as exc:
            show_warning(self, "Invalid JSON", str(exc))
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


class F8NumberLineEditor(QtWidgets.QLineEdit):
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
        except _NUMERIC_CONVERSION_ERRORS:
            self._min = None
        self._update_validator()

    def set_max(self, v) -> None:
        try:
            self._max = float(v)
        except _NUMERIC_CONVERSION_ERRORS:
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
        except _QT_COMPAT_ERRORS as exc:
            logger.debug("Failed to set double validator notation", exc_info=exc)
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
        except _NUMERIC_CONVERSION_ERRORS:
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
        except _NUMERIC_CONVERSION_ERRORS:
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


class F8OptionComboEditor(QtWidgets.QWidget):
    """
    Property-value editor wrapper around the reusable combo control.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
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
        value = self._combo.value()
        if value is None:
            return None
        return str(value)

    def set_context_tooltip(self, tooltip: str) -> None:
        self._combo.set_context_tooltip(tooltip)

    def set_read_only(self, read_only: bool) -> None:
        self._combo.set_read_only(bool(read_only))

    def _emit(self, value: Any) -> None:
        self.value_changed.emit(self.get_name(), None if value is None else str(value))


class F8MultiSelectEditor(QtWidgets.QWidget):
    """
    Property-value editor wrapper around the reusable multi-select control.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
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

    def _emit(self, value: Any) -> None:
        out = [str(item) for item in list(value or [])]
        self.value_changed.emit(self.get_name(), out)


class F8BoolSwitchEditor(QtWidgets.QWidget):
    """
    Property-value editor wrapper around the reusable boolean switch control.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
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

    def _emit(self, value: Any) -> None:
        self.value_changed.emit(self.get_name(), bool(value))


class F8ValueBarEditor(QtWidgets.QWidget):
    """
    Property-value editor wrapper around the reusable value-bar control.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, data_type: type[int] | type[float]) -> None:
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

    def set_min(self, value: Any) -> None:
        self._min = value
        self._bar.set_range(self._min, self._max)

    def set_max(self, value: Any) -> None:
        self._max = value
        self._bar.set_range(self._min, self._max)

    def set_value(self, value: Any) -> None:
        self._bar.set_value(value)

    def get_value(self) -> Any:
        value = self._bar.value()
        return int(value) if self._data_type is int else float(value)

    def set_read_only(self, read_only: bool) -> None:
        self._bar.set_read_only(bool(read_only))

    def _emit(self, value: Any) -> None:
        out = int(value) if self._data_type is int else float(value)
        self.value_changed.emit(self.get_name(), out)


class F8RangeBarEditor(QtWidgets.QWidget):
    """Property-value editor wrapper for a two-value range slider."""

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, data_type: type[int] | type[float]) -> None:
        super().__init__(parent)
        self._name = ""
        self._data_type = data_type
        self._min: float | int | None = None
        self._max: float | int | None = None
        self._bar = F8RangeBar(integer=(data_type is int), minimum=0.0, maximum=1.0)
        self._bar.valueCommitted.connect(self._emit)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._bar, 1)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_min(self, value: Any) -> None:
        self._min = value
        self._bar.set_range(self._min, self._max)

    def set_max(self, value: Any) -> None:
        self._max = value
        self._bar.set_range(self._min, self._max)

    def set_value(self, value: Any) -> None:
        self._bar.set_value(value)

    def get_value(self) -> Any:
        values = self._bar.value()
        if self._data_type is int:
            return [int(values[0]), int(values[1])]
        return [float(values[0]), float(values[1])]

    def set_read_only(self, read_only: bool) -> None:
        self._bar.set_read_only(bool(read_only))

    def _emit(self, value: Any) -> None:
        if self._data_type is int:
            out = [int(value[0]), int(value[1])]
        else:
            out = [float(value[0]), float(value[1])]
        self.value_changed.emit(self.get_name(), out)


class F8DialEditor(QtWidgets.QWidget):
    """
    Property-value editor wrapper around the reusable circular dial control.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, data_type: type[int] | type[float]) -> None:
        super().__init__(parent)
        self._name = ""
        self._data_type = data_type
        self._min: float | int | None = None
        self._max: float | int | None = None
        self._dial = F8Dial(integer=(data_type is int), minimum=0.0, maximum=1.0)
        self._dial.valueCommitted.connect(self._emit)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._dial, 1)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_min(self, value: Any) -> None:
        self._min = value
        self._dial.set_range(self._min, self._max)

    def set_max(self, value: Any) -> None:
        self._max = value
        self._dial.set_range(self._min, self._max)

    def set_value(self, value: Any) -> None:
        self._dial.set_value(value)

    def get_value(self) -> Any:
        value = self._dial.value()
        return int(value) if self._data_type is int else float(value)

    def set_read_only(self, read_only: bool) -> None:
        self._dial.set_read_only(bool(read_only))

    def set_loop(self, loop: bool) -> None:
        self._dial.set_loop(bool(loop))

    def set_context_tooltip(self, tooltip: str) -> None:
        self._dial.set_context_tooltip(tooltip)

    def set_invalid_reason(self, reason: str) -> None:
        self._dial.set_invalid_reason(reason)

    def _emit(self, value: Any) -> None:
        out = int(value) if self._data_type is int else float(value)
        self.value_changed.emit(self.get_name(), out)


class F8ImageValueEditor(QtWidgets.QWidget):
    """
    Property-value editor wrapper around the reusable base64 image control.
    """

    value_changed = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = ""
        self._widget = F8ImageB64Editor()
        self._widget.valueChanged.connect(self._emit)  # type: ignore[attr-defined]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widget, 1)

    def set_name(self, name: str) -> None:
        self._name = str(name or "")

    def get_name(self) -> str:
        return self._name

    def set_value(self, value: Any) -> None:
        self._widget.set_value("" if value is None else str(value))

    def get_value(self) -> Any:
        return str(self._widget.value() or "")

    def _emit(self, value: str) -> None:
        self.value_changed.emit(self.get_name(), str(value or ""))


__all__ = [
    "F8CodeButtonEditor",
    "F8InlineCodeEditor",
    "F8IncrementButtonEditor",
    "F8WrapLineEditor",
    "F8JsonValueEditor",
    "F8NumberLineEditor",
    "F8OptionComboEditor",
    "F8MultiSelectEditor",
    "F8BoolSwitchEditor",
    "F8DialEditor",
    "F8ValueBarEditor",
    "F8ImageValueEditor",
]
