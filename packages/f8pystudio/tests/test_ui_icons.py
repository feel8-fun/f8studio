from __future__ import annotations

import pytest
from qtpy import QtWidgets, QtGui

from f8pystudio.ui_icons import StudioIcon, _icons_dir, _render_svg_icon, icon_for


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


def test_icon_enum_values_use_svg_extension() -> None:
    for token in StudioIcon:
        assert token.value.endswith(".svg")


def test_all_icon_assets_exist() -> None:
    base_dir = _icons_dir()
    for token in StudioIcon:
        assert (base_dir / token.value).is_file()


def test_icon_for_raises_runtime_error_when_asset_file_missing() -> None:
    _ensure_app()
    widget = QtWidgets.QWidget()
    with pytest.raises(RuntimeError):
        _render_svg_icon(widget, StudioIcon.CODE, "__missing__.svg")
