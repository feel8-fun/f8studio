from __future__ import annotations

from typing import Any

from qtpy import QtWidgets

from f8pystudio.nodegraph.items.inline_command_panel import (
    _on_command_pressed,
    _restore_selected_node_ids,
    _snapshot_selected_node_ids,
    ensure_inline_command_widget,
    invoke_command,
)


class _FakeNode:
    def __init__(self, node_id: str, selected: bool = False) -> None:
        self.id = node_id
        self.selected = selected

    def set_property(self, name: str, value: Any, push_undo: bool = True) -> None:
        del push_undo
        if name == "selected":
            self.selected = bool(value)


class _FakeGraph:
    def __init__(self, nodes: list[_FakeNode]) -> None:
        self._nodes = list(nodes)

    def all_nodes(self) -> list[_FakeNode]:
        return list(self._nodes)

    def selected_nodes(self) -> list[_FakeNode]:
        return [node for node in self._nodes if node.selected]


class _FakeBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def invoke_remote_command(self, service_id: str, name: str, args: dict[str, Any] | None = None) -> None:
        self.calls.append((service_id, name, dict(args or {})))


class _FakeNodeItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, *, graph: _FakeGraph, service_running: bool = True) -> None:
        super().__init__(0.0, 0.0, 10.0, 10.0)
        self._fake_graph = graph
        self._service_running = service_running
        self._bridge_obj = _FakeBridge()
        self._invoke_count = 0
        self.id = "A"

        self._cmd_proxy = None
        self._cmd_widget = None
        self._cmd_buttons: list[QtWidgets.QPushButton] = []
        self._cmd_serial = ""
        self._tooltip_filters: list[Any] = []

        self._backend = _FakeBackendNode()

    def _graph(self) -> _FakeGraph:
        return self._fake_graph

    def _invoke_command(self, command: Any) -> None:
        del command
        self._invoke_count += 1
        # Simulate NodeGraph selection side effect: command press briefly selects node A.
        for node in self._fake_graph.all_nodes():
            node.selected = (node.id == "A")

    def _ensure_bridge_process_hook(self) -> None:
        return

    def _backend_node(self) -> Any:
        return self._backend

    def _is_service_running(self) -> bool:
        return self._service_running

    def _bridge(self) -> _FakeBridge:
        return self._bridge_obj

    def _service_id(self) -> str:
        return "svcA"

    def viewer(self) -> None:
        return None

    def _schema_enum_items(self, schema: Any) -> list[str]:
        del schema
        return []

    def _schema_numeric_range(self, schema: Any) -> tuple[float | None, float | None]:
        del schema
        return None, None


class _FakeBackendNode:
    def __init__(self) -> None:
        self.spec = None

    def effective_commands(self) -> list[Any]:
        return [_FakeCommand("Run", "Run command", True, [])]


class _FakeCommand:
    def __init__(self, name: str, description: str, show_on_node: bool, params: list[Any]) -> None:
        self.name = name
        self.description = description
        self.showOnNode = show_on_node
        self.params = list(params)


class _InvokeNodeItem:
    def __init__(self, *, service_running: bool) -> None:
        self._service_running = service_running
        self._bridge_obj = _FakeBridge()

    def _bridge(self) -> _FakeBridge:
        return self._bridge_obj

    def _service_id(self) -> str:
        return "svcA"

    def _is_service_running(self) -> bool:
        return self._service_running

    def _backend_node(self) -> None:
        return None


def _ensure_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is not None:
        return app
    return QtWidgets.QApplication([])


def test_snapshot_and_restore_selected_ids() -> None:
    _ensure_app()

    a = _FakeNode("A", selected=False)
    b = _FakeNode("B", selected=True)
    graph = _FakeGraph([a, b])
    node_item = _FakeNodeItem(graph=graph)

    ids = _snapshot_selected_node_ids(node_item)
    assert ids == ["B"]

    _restore_selected_node_ids(node_item, ["A"])
    assert a.selected is True
    assert b.selected is False


def test_invoke_command_skips_when_service_not_running() -> None:
    node_item = _InvokeNodeItem(service_running=False)

    invoke_command(node_item, _FakeCommand("Run", "Run command", True, []))

    assert node_item._bridge_obj.calls == []
