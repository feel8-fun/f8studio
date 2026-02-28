from __future__ import annotations

import pytest
from qtpy import QtWidgets, QtGui

from f8pystudio.ui_icons import StudioIcon, _ICON_MAP, _icons_dir, icon_for


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


def test_all_icon_assets_exist() -> None:
    base_dir = _icons_dir()
    for token in StudioIcon:
        icon_name = _ICON_MAP[token]
        assert (base_dir / icon_name).is_file()


def test_icon_for_raises_runtime_error_when_asset_file_missing() -> None:
    _ensure_app()
    widget = QtWidgets.QWidget()
    token = StudioIcon.CODE
    saved = _ICON_MAP[token]
    _ICON_MAP[token] = "__missing__.svg"
    try:
        with pytest.raises(RuntimeError):
            icon_for(widget, token)
    finally:
        _ICON_MAP[token] = saved
