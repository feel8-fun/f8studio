from __future__ import annotations

import logging
from typing import Any, Callable

from qtpy import QtGui, QtWidgets

logger = logging.getLogger(__name__)

try:
    from pyqttoast import Toast, ToastPreset  # type: ignore[import-not-found]
except ImportError:
    Toast = None  # type: ignore[assignment]
    ToastPreset = None  # type: ignore[assignment]


def _resolve_parent(parent: QtWidgets.QWidget | None) -> QtWidgets.QWidget | None:
    if parent is not None:
        return parent
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None
    focus_widget = app.focusWidget()
    if focus_widget is not None:
        return focus_widget.window()
    active_modal_widget = app.activeModalWidget()
    if active_modal_widget is not None:
        return active_modal_widget.window()
    active_window = app.activeWindow()
    if active_window is not None:
        return active_window
    cursor_screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
    visible_top_levels: list[QtWidgets.QWidget] = []
    for widget in app.topLevelWidgets():
        if not widget.isVisible():
            continue
        visible_top_levels.append(widget)
    if cursor_screen is not None:
        for widget in visible_top_levels:
            handle = widget.windowHandle()
            if handle is not None and handle.screen() == cursor_screen:
                return widget.window()
    if visible_top_levels:
        return visible_top_levels[0].window()
    top_level_widgets = app.topLevelWidgets()
    if top_level_widgets:
        return top_level_widgets[0].window()
    return None


def _show_toast(
    *,
    parent: QtWidgets.QWidget | None,
    title: str,
    message: str,
    preset: Any,
    fallback: Callable[[QtWidgets.QWidget | None, str, str], None],
) -> None:
    target_parent = _resolve_parent(parent)
    title_text = str(title or "").strip()
    message_text = str(message or "").strip()
    if Toast is None or ToastPreset is None:
        fallback(target_parent, title_text, message_text)
        return
    try:
        toast = Toast(target_parent)
        toast.setTitle(title_text)
        toast.setText(message_text)
        toast.applyPreset(preset)
        toast.show()
    except Exception:
        logger.exception("Failed to show toast notification")
        fallback(target_parent, title_text, message_text)


def show_info(parent: QtWidgets.QWidget | None, title: str, message: str) -> None:
    preset = ToastPreset.INFORMATION if ToastPreset is not None else None
    _show_toast(
        parent=parent,
        title=title,
        message=message,
        preset=preset,
        fallback=QtWidgets.QMessageBox.information,
    )


def show_warning(parent: QtWidgets.QWidget | None, title: str, message: str) -> None:
    preset = ToastPreset.WARNING if ToastPreset is not None else None
    _show_toast(
        parent=parent,
        title=title,
        message=message,
        preset=preset,
        fallback=QtWidgets.QMessageBox.warning,
    )


def show_error(parent: QtWidgets.QWidget | None, title: str, message: str) -> None:
    preset = ToastPreset.ERROR if ToastPreset is not None else None
    _show_toast(
        parent=parent,
        title=title,
        message=message,
        preset=preset,
        fallback=QtWidgets.QMessageBox.critical,
    )
