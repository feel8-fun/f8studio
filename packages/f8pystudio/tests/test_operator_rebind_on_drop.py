from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from f8pystudio.nodegraph.node_graph import F8StudioGraph


class _FakeContainer:
    def __init__(self, node_id: str, service_class: str) -> None:
        self.id = node_id
        self.spec = SimpleNamespace(serviceClass=service_class)
        self._child_nodes: list[Any] = []
        self.view = SimpleNamespace(_child_views=[])

    def add_child(self, node: Any) -> None:
        if node not in self._child_nodes:
            self._child_nodes.append(node)
        if node.view not in self.view._child_views:
            self.view._child_views.append(node.view)
        node.view._container_item = self

    def remove_child(self, node: Any) -> None:
        self._child_nodes = [n for n in self._child_nodes if n is not node]
        self.view._child_views = [v for v in self.view._child_views if v is not node.view]
        if node.view._container_item is self:
            node.view._container_item = None


class _FakeOperator:
    def __init__(self, node_id: str, service_class: str, svc_id: str, *, x: float, y: float) -> None:
        self.id = node_id
        self.spec = SimpleNamespace(serviceClass=service_class)
        self.svcId = svc_id
        self.model = SimpleNamespace(properties={"svcId": svc_id}, custom_properties={}, pos=[x, y])
        self.view = SimpleNamespace(id=node_id, _container_item=None, xy_pos=[x, y])
        self._properties: dict[str, Any] = {"svcId": svc_id}

    def set_property(self, name: str, value: Any, push_undo: bool = True) -> None:
        _ = push_undo
        self._properties[str(name)] = value

    def input_ports(self) -> list[Any]:
        return []

    def output_ports(self) -> list[Any]:
        return []


def _new_graph() -> F8StudioGraph:
    graph = F8StudioGraph.__new__(F8StudioGraph)
    graph._is_operator_node = lambda node: isinstance(node, _FakeOperator)  # type: ignore[method-assign]
    graph._is_container_node = lambda node: isinstance(node, _FakeContainer)  # type: ignore[method-assign]
    graph._disconnect_invalid_connections_for_operator = lambda op: 0  # type: ignore[method-assign]
    graph._notification_parent = lambda: None  # type: ignore[method-assign]
    return graph


def test_on_operator_drop_rebinds_operator_to_new_same_class_container(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("f8pystudio.nodegraph.node_graph.show_warning", lambda *args, **kwargs: warnings.append("w"))

    old_container = _FakeContainer("svc_old", "f8.pyengine")
    new_container = _FakeContainer("svc_new", "f8.pyengine")
    operator = _FakeOperator("op1", "f8.pyengine", "svc_old", x=10.0, y=10.0)
    old_container.add_child(operator)

    graph = _new_graph()
    nodes = {"op1": operator, "svc_old": old_container, "svc_new": new_container}
    graph.get_node_by_id = lambda node_id: nodes.get(str(node_id))  # type: ignore[method-assign]
    graph._container_at_node = lambda node: new_container  # type: ignore[method-assign]

    ok, msg = graph.on_operator_drop(node_id="op1", start_pos=(10.0, 10.0), start_container_id="svc_old")

    assert ok is True
    assert msg == ""
    assert operator.svcId == "svc_new"
    assert operator._properties["svcId"] == "svc_new"
    assert operator in new_container._child_nodes
    assert operator not in old_container._child_nodes
    assert warnings == []


def test_on_operator_drop_reverts_when_target_container_service_class_mismatch(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("f8pystudio.nodegraph.node_graph.show_warning", lambda *args, **kwargs: warnings.append("w"))

    old_container = _FakeContainer("svc_old", "f8.pyengine")
    wrong_container = _FakeContainer("svc_other", "f8.audio")
    operator = _FakeOperator("op1", "f8.pyengine", "svc_old", x=33.0, y=44.0)
    old_container.add_child(operator)
    operator.view.xy_pos = [120.0, 220.0]
    operator.model.pos = [120.0, 220.0]

    graph = _new_graph()
    nodes = {"op1": operator, "svc_old": old_container, "svc_other": wrong_container}
    graph.get_node_by_id = lambda node_id: nodes.get(str(node_id))  # type: ignore[method-assign]
    graph._container_at_node = lambda node: wrong_container  # type: ignore[method-assign]

    ok, _msg = graph.on_operator_drop(node_id="op1", start_pos=(33.0, 44.0), start_container_id="svc_old")

    assert ok is False
    assert operator.svcId == "svc_old"
    assert operator.view.xy_pos == [33.0, 44.0]
    assert operator.model.pos == [33.0, 44.0]
    assert operator in old_container._child_nodes
    assert warnings == ["w"]


def test_on_operator_drop_reverts_when_not_dropped_inside_container(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("f8pystudio.nodegraph.node_graph.show_warning", lambda *args, **kwargs: warnings.append("w"))

    old_container = _FakeContainer("svc_old", "f8.pyengine")
    operator = _FakeOperator("op1", "f8.pyengine", "svc_old", x=5.0, y=8.0)
    old_container.add_child(operator)
    operator.view.xy_pos = [300.0, 400.0]
    operator.model.pos = [300.0, 400.0]

    graph = _new_graph()
    nodes = {"op1": operator, "svc_old": old_container}
    graph.get_node_by_id = lambda node_id: nodes.get(str(node_id))  # type: ignore[method-assign]
    graph._container_at_node = lambda node: None  # type: ignore[method-assign]

    ok, _msg = graph.on_operator_drop(node_id="op1", start_pos=(5.0, 8.0), start_container_id="svc_old")

    assert ok is False
    assert operator.svcId == "svc_old"
    assert operator.view.xy_pos == [5.0, 8.0]
    assert operator.model.pos == [5.0, 8.0]
    assert warnings == ["w"]
