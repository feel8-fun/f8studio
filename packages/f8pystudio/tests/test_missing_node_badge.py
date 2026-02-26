from __future__ import annotations

from qtpy import QtWidgets

from f8pystudio.nodegraph.missing_operator_basenode import F8StudioMissingOperatorNodeItem
from f8pystudio.nodegraph.missing_service_basenode import F8StudioMissingServiceNodeItem
from f8pystudio.nodegraph.service_basenode import F8StudioServiceNodeItem


class _FakeMissingNode:
    def __init__(self, *, locked: bool, missing_type: str) -> None:
        self._locked = bool(locked)
        self._missing_type = str(missing_type)

    def is_missing_locked(self) -> bool:
        return self._locked

    def missing_type(self) -> str:
        return self._missing_type


class _TestMissingServiceNodeItem(F8StudioMissingServiceNodeItem):
    def __init__(self, fake_node: _FakeMissingNode):
        super().__init__(name="n")
        self._fake_node = fake_node

    def _backend_node(self):  # type: ignore[override]
        return self._fake_node


class _TestMissingOperatorNodeItem(F8StudioMissingOperatorNodeItem):
    def __init__(self, fake_node: _FakeMissingNode):
        super().__init__(name="n")
        self._fake_node = fake_node

    def _backend_node(self):  # type: ignore[override]
        return self._fake_node


class _RetryMissingServiceNodeItem(_TestMissingServiceNodeItem):
    def __init__(self, fake_node: _FakeMissingNode):
        super().__init__(fake_node)
        self._badge_calls = 0

    def _missing_badge_pixmap(self):  # type: ignore[override]
        self._badge_calls += 1
        if self._badge_calls == 1:
            raise RuntimeError("first badge build failed")
        return super()._missing_badge_pixmap()


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is not None:
        return app
    return QtWidgets.QApplication([])


def test_missing_service_item_shows_badge_and_tooltip() -> None:
    _ensure_app()
    item = _TestMissingServiceNodeItem(_FakeMissingNode(locked=True, missing_type="svc.f8.cvkit.templatematch"))
    item._refresh_missing_badge()
    assert item._missing_badge_item.isVisible() is True
    assert "missing svc.f8.cvkit.templatematch" in str(item._missing_badge_item.toolTip())


def test_missing_operator_item_shows_badge_and_tooltip() -> None:
    _ensure_app()
    item = _TestMissingOperatorNodeItem(_FakeMissingNode(locked=True, missing_type="svc.f8.cvkit.denseoptflow"))
    item._refresh_missing_badge()
    assert item._missing_badge_item.isVisible() is True
    assert "missing svc.f8.cvkit.denseoptflow" in str(item._missing_badge_item.toolTip())


def test_base_service_item_does_not_have_missing_badge() -> None:
    _ensure_app()
    item = F8StudioServiceNodeItem(name="n")
    assert not hasattr(item, "_missing_badge_item")


def test_missing_badge_retries_after_initial_failure() -> None:
    _ensure_app()
    item = _RetryMissingServiceNodeItem(_FakeMissingNode(locked=True, missing_type="svc.f8.cvkit.templatematch"))
    item._refresh_missing_badge()
    assert item._missing_badge_item.isVisible() is False
    item._refresh_missing_badge()
    assert item._missing_badge_item.isVisible() is True
