from __future__ import annotations

import pytest
from qtpy import QtWidgets, QtGui

from f8pystudio.ui_icons import StudioIcon, _ICON_MAP, icon_for


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is not None:
        return app
    return QtWidgets.QApplication([])


def test_icon_for_all_tokens_returns_non_null_icon() -> None:
    _ensure_app()
    widget = QtWidgets.QWidget()
    for token in StudioIcon:
        icon = icon_for(widget, token)
        assert isinstance(icon, QtGui.QIcon)
        assert not icon.isNull()


def test_icon_for_raises_key_error_when_token_not_mapped() -> None:
    _ensure_app()
    widget = QtWidgets.QWidget()
    token = StudioIcon.CODE
    saved = _ICON_MAP.pop(token)
    try:
        with pytest.raises(KeyError):
            icon_for(widget, token)
    finally:
        _ICON_MAP[token] = saved


class _NullIconStyle(QtWidgets.QProxyStyle):
    def standardIcon(self, standardIcon, option=None, widget=None):  # type: ignore[override]
        del standardIcon, option, widget
        return QtGui.QIcon()


def test_icon_for_raises_runtime_error_when_qstyle_returns_null_icon() -> None:
    _ensure_app()
    widget = QtWidgets.QWidget()
    style = _NullIconStyle()
    widget.setStyle(style)
    with pytest.raises(RuntimeError):
        icon_for(widget, StudioIcon.CODE)
