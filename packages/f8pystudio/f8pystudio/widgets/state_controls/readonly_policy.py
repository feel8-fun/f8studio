from __future__ import annotations

from qtpy import QtCore, QtWidgets

from ..f8_editor_widgets import F8PropBoolSwitch, F8PropMultiSelect, F8PropOptionCombo
from ..f8_prop_value_widgets import F8CodeButtonPropWidget


def set_widget_read_only(widget: QtWidgets.QWidget, *, read_only: bool) -> None:
    if isinstance(widget, F8PropOptionCombo):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8PropMultiSelect):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8PropBoolSwitch):
        widget.set_read_only(bool(read_only))
        return
    if isinstance(widget, F8CodeButtonPropWidget):
        widget.set_read_only(bool(read_only))
        return

    if isinstance(widget, QtWidgets.QLineEdit):
        widget.setEnabled(True)
        widget.setReadOnly(bool(read_only))
        return
    if isinstance(widget, QtWidgets.QPlainTextEdit):
        widget.setEnabled(True)
        widget.setReadOnly(bool(read_only))
        return
    if isinstance(widget, QtWidgets.QTextEdit):
        widget.setEnabled(True)
        widget.setReadOnly(bool(read_only))
        if read_only:
            widget.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
        else:
            widget.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
        return
    if isinstance(widget, QtWidgets.QAbstractSpinBox):
        widget.setEnabled(True)
        widget.setReadOnly(bool(read_only))
        if read_only:
            widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        else:
            widget.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        return

    widget.setEnabled(not bool(read_only))
