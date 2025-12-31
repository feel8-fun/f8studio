from __future__ import annotations

from typing import Any

from f8pysdk import F8EdgeKindEmum, F8EdgeSpec
from Qt import QtCore

from ..graph.operator_graph import OperatorGraph
from ..graph.operator_instance import OperatorInstance
from ..renderers.generic import OperatorNodeBase

from ..renderers.renderer_registry import OperatorRendererRegistry
from ..operators.operator_registry import OperatorSpecRegistry
from .spec_node_class_registry import SpecNodeClassRegistry
from NodeGraphQt import NodeGraph, BaseNode


class OperatorGraphEditor:
    """
    Thin wrapper that mirrors OperatorGraph contents into a NodeGraphQt scene.

    The view is renderer-driven: each OperatorSpec chooses a rendererClass that resolves
    through OperatorRendererRegistry. A generic renderer is provided by default.
    """

    def __init__(
        self,
        *,
        graph: OperatorGraph | None = None,
    ) -> None:
        
        self.graph = graph or OperatorGraph()

        self.node_graph = NodeGraph()
        
        self._node_index: dict[str, BaseNode] = {}
        self._counter = 0

        SpecNodeClassRegistry.instance().apply(self.node_graph)
        # self._node_type_by_operator_class: dict[str, str] = {}
        try:
            self.node_graph.property_changed.connect(self._on_node_property_changed)
        except Exception:
            pass
        try:
            self.node_graph.viewer().moved_nodes.connect(self._on_nodes_moved)
        except Exception:
            pass
        try:
            self.node_graph.node_created.connect(self._on_node_created)
        except Exception:
            pass
        try:
            self.node_graph.nodes_deleted.connect(self._on_nodes_deleted)
        except Exception:
            pass

    # Public API ----------------------------------------------------------
    def spawn_instance(
        self,
        operator_class: str,
        *,
        pos: tuple[float, float] | None = None,
        instance_id: str | None = None,
    ) -> OperatorInstance:
        
        spec = OperatorSpecRegistry.instance().get(operator_class)
        node_id = instance_id or self._unique_node_id(operator_class)
        instance = OperatorInstance.from_spec(spec, id=node_id)
        if pos:
            self._set_node_renderer_pos(instance, pos)
        self.graph.add_node(instance)
        self._build_node(instance, pos=pos)
        return instance

    def rebuild(self) -> None:
        """Rebuild the entire NodeGraphQt scene from the backing OperatorGraph."""
        cleared = False
        if hasattr(self.node_graph, "clear_session"):
            try:
                self.node_graph.clear_session()
                cleared = True
            except Exception:
                pass
        if not cleared and hasattr(self.node_graph, "clear"):
            try:
                self.node_graph.clear()
            except Exception:
                pass
        self._node_index.clear()
        for instance in self.graph.nodes.values():
            self._build_node(instance)
        for edge in [*self.graph.exec_edges, *self.graph.data_edges, *self.graph.state_edges]:
            self._connect_edge(edge)

    def show(self) -> None:
        """Show the NodeGraphQt widget."""
        if hasattr(self.node_graph, "widget"):
            self.node_graph.widget.show()
        elif hasattr(self.node_graph, "show"):
            self.node_graph.show()

    def widget(self) -> Any:
        """Expose the underlying Qt widget for host applications."""
        return getattr(self.node_graph, "widget", self.node_graph)

    def _on_node_created(self, node: Any) -> None:
        if not isinstance(node, BaseNode):
            return
        if not isinstance(node, OperatorNodeBase):
            return

        instance = getattr(node, "instance", None)
        if not isinstance(instance, OperatorInstance):
            return

        if instance.id not in self.graph.nodes:
            try:
                self.graph.add_node(instance)
            except Exception:
                pass

        self._node_index[instance.id] = node

        try:
            pos = node.pos()
            self._set_node_renderer_pos(instance, (float(pos[0]), float(pos[1])))
        except Exception:
            pass

    def _on_nodes_deleted(self, node_ids: list[str]) -> None:
        for node_id in node_ids or []:
            self._node_index.pop(node_id, None)
            try:
                self.graph.remove_node(node_id)
            except Exception:
                pass

    @staticmethod
    def _coerce_pos(value: Any) -> tuple[float, float] | None:
        """
        Best-effort conversion for stored node positions.

        Accepts tuples/lists and Qt-like point objects that expose x()/y() methods.
        """
        if value is None:
            return None

        if isinstance(value, (tuple, list)) and len(value) == 2:
            try:
                return (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                return None

        x_attr = getattr(value, "x", None)
        y_attr = getattr(value, "y", None)
        if callable(x_attr) and callable(y_attr):
            try:
                return (float(x_attr()), float(y_attr()))
            except (TypeError, ValueError):
                return None

        return None

    @staticmethod
    def _set_node_renderer_pos(instance: OperatorInstance, pos: tuple[float, float]) -> None:
        props = instance.spec.rendererProps or {}
        props["pos"] = [float(pos[0]), float(pos[1])]
        instance.spec.rendererProps = props

    @staticmethod
    def _get_node_renderer_pos(instance: OperatorInstance) -> tuple[float, float] | None:
        props = instance.spec.rendererProps or {}
        return OperatorGraphEditor._coerce_pos(props.get("pos"))

    def _on_node_property_changed(self, node: Any, name: str, value: object) -> None:
        if name != "pos":
            return
        node_id = getattr(node, "id", None)
        if not isinstance(node_id, str):
            return
        instance = self.graph.nodes.get(node_id)
        if instance is None:
            return
        pos = self._coerce_pos(value)
        if pos is None:
            return
        self._set_node_renderer_pos(instance, pos)

    def _on_nodes_moved(self, node_data: Any) -> None:
        """
        Track interactive node moves from the NodeGraphQt viewer.

        The viewer emits a dict of {<node_view>: <previous_pos>}.
        """
        if not isinstance(node_data, dict):
            return
        for node_view in node_data.keys():
            node_id = getattr(node_view, "id", None)
            if not isinstance(node_id, str):
                continue
            instance = self.graph.nodes.get(node_id)
            if instance is None:
                continue
            pos = self._coerce_pos(getattr(node_view, "xy_pos", None))
            if pos is None:
                continue
            self._set_node_renderer_pos(instance, pos)

    def _build_node(self, instance: OperatorInstance, *, pos: tuple[float, float] | None = None) -> BaseNode:
        if pos is None:
            pos = self._get_node_renderer_pos(instance)

        # renderer_key = instance.spec.rendererClass or "default"
        renderer_cls = SpecNodeClassRegistry.instance().get(instance.spec.operatorClass)
        node = renderer_cls(instance)

        # Use stable ids so we can map NodeGraphQt events back to OperatorInstance.
        try:
            node.model.id = instance.id
            node.view.id = instance.id
        except Exception:
            pass

        # Add nodes at their stored position to avoid overlap when rebuilding.
        if pos is None:
            self.node_graph.add_node(node, selected=False, push_undo=False)
        else:
            node_pos = [float(pos[0]), float(pos[1])]
            self.node_graph.add_node(node, pos=node_pos, selected=False, push_undo=False)
            # Ensure the underlying node model stores the position without polluting undo history.
            try:
                node.set_property("pos", node_pos, push_undo=False)
            except Exception:
                pass

        self._node_index[instance.id] = node
        return node

    def _connect_edge(self, edge: F8EdgeSpec) -> None:
        try:
            source_node = self._node_index[edge.from_]
            target_node = self._node_index[edge.to]
        except KeyError:
            return

        if edge.kind == F8EdgeKindEmum.exec:
            out_port = source_node.port("exec", "out", edge.fromPort)
            in_port = target_node.port("exec", "in", edge.toPort)
        elif edge.kind == F8EdgeKindEmum.data:
            out_port = source_node.port("data", "out", edge.fromPort)
            in_port = target_node.port("data", "in", edge.toPort)
        elif edge.kind == F8EdgeKindEmum.state:
            out_port = source_node.port("state", "out", edge.fromPort)
            in_port = target_node.port("state", "in", edge.toPort)
        else:
            return

        if out_port is None or in_port is None:
            return
        try:
            self.node_graph.connect_ports(out_port, in_port)
        except Exception:
            pass

    def _unique_node_id(self, operator_class: str) -> str:
        self._counter += 1
        short_name = operator_class.split(".")[-1]
        return f"{short_name}_{self._counter}"
