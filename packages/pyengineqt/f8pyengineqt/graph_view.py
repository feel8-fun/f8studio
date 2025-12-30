from __future__ import annotations

from typing import Any

from f8pysdk import F8EdgeKindEmum, F8EdgeSpec

from .operator_graph import OperatorGraph
from .operator_instance import OperatorInstance

from .renderers.renderer_registry import OperatorRendererRegistry
from .operators.operator_registry import OperatorSpecRegistry

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
        spec_registry: OperatorSpecRegistry,
        renderer_registry: OperatorRendererRegistry,
        graph: OperatorGraph | None = None,
    ) -> None:
        self.spec_registry = spec_registry
        self.renderer_registry = renderer_registry

        self.graph = graph or OperatorGraph()

        self.node_graph = NodeGraph()
        self._node_index: dict[str, BaseNode] = {}
        self._counter = 0

    # Public API ----------------------------------------------------------
    def spawn_instance(
        self,
        operator_class: str,
        *,
        pos: tuple[float, float] | None = None,
        instance_id: str | None = None,
    ) -> OperatorInstance:
        template = self.spec_registry.get(operator_class)
        node_id = instance_id or self._unique_node_id(operator_class)
        instance = OperatorInstance.from_spec(template, id=node_id)
        if pos:
            instance.ctx["pos"] = pos
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

    # Internal helpers ----------------------------------------------------
    def _build_node(self, instance: OperatorInstance, *, pos: tuple[float, float] | None = None) -> BaseNode:
        renderer_key = instance.spec.rendererClass or "default"
        renderer_cls = self.renderer_registry.get(renderer_key)
        node = renderer_cls(instance)

        # node = renderer.build_node(self.node_graph, instance, pos=node_pos)
        self._node_index[instance.id] = node
        if pos:
            self._node_index[instance.id].set_pos(*pos)
        self.node_graph.add_node(node)
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
