from __future__ import annotations

from typing import Any, Iterable

from NodeGraphQt import BaseNode, NodeGraph

from ..graph.operator_graph import OperatorGraph
from ..graph.operator_instance import OperatorInstance
from ..operators.operator_registry import OperatorSpecRegistry
from ..renderers.generic import GenericNode
from ..renderers.renderer_registry import OperatorRendererRegistry
from ..schema.compat import schema_is_superset, schema_signature
from .spec_node_class_registry import SpecNodeClassRegistry
from f8pysdk import F8OperatorSpec, F8PrimitiveTypeEnum
from .f8_node_viewer import F8NodeViewer
from .engine_manager import EngineManager
from ..services.service_registry import ServiceSpecRegistry
from ..services.builtin import engine_service_spec, ENGINE_SERVICE_CLASS
from ..renderers.service_engine import EngineServiceNode

from pathlib import Path

BASE_PATH = Path(__file__).parent


class OperatorGraphEditor:
    """
    Editor-only graph wrapper.

    - During editing: NodeGraphQt is the source of truth.
    - During deployment: export the NodeGraphQt session to an OperatorGraph.
    """

    def __init__(self) -> None:
        self.node_graph = NodeGraph(viewer=F8NodeViewer())
        self.engine = EngineManager(self)

        hotkey_path = BASE_PATH / "hotkeys" / "hotkeys.json"
        self.node_graph.set_context_menu_from_file(str(hotkey_path), "graph")

        # Ensure singletons are initialized so the spec node registry can use them.
        OperatorSpecRegistry.instance()
        OperatorRendererRegistry.instance()
        ServiceSpecRegistry.instance().register(engine_service_spec(), overwrite=True)

        self.node_graph._node_factory.clear_registered_nodes()
        SpecNodeClassRegistry.instance().apply(self.node_graph)
        try:
            self.node_graph.register_node(EngineServiceNode, alias=ENGINE_SERVICE_CLASS)
        except Exception:
            pass

        # First-version engine: KV sync + topology cache.
        self.engine.start()

    def show(self) -> None:
        if hasattr(self.node_graph, "widget"):
            self.node_graph.widget.show()
        elif hasattr(self.node_graph, "show"):
            self.node_graph.show()

    def widget(self) -> Any:
        return getattr(self.node_graph, "widget", self.node_graph)

    def create_node(
        self,
        operator_class: str,
        *,
        pos: tuple[float, float] | None = None,
        name: str | None = None,
    ) -> GenericNode:
        kwargs: dict[str, Any] = {"push_undo": False, "selected": False}
        if pos is not None:
            kwargs["pos"] = [float(pos[0]), float(pos[1])]
        if name is not None:
            kwargs["name"] = name
        node = self.node_graph.create_node(operator_class, **kwargs)
        if not isinstance(node, GenericNode):
            raise TypeError(f"Expected GenericNode for {operator_class}, got {type(node)}")
        return node

    def connect(
        self,
        source: GenericNode,
        *,
        kind: str,
        out_port: str,
        target: GenericNode,
        in_port: str,
    ) -> None:
        if kind in ("data", "state"):
            try:
                if kind == "data":
                    out_schema = next(p.valueSchema for p in (source.spec.dataOutPorts or []) if p.name == out_port)
                    in_schema = next(p.valueSchema for p in (target.spec.dataInPorts or []) if p.name == in_port)
                else:
                    out_schema = next(s.valueSchema for s in (source.spec.states or []) if s.name == out_port)
                    in_schema = next(s.valueSchema for s in (target.spec.states or []) if s.name == in_port)
                if not schema_is_superset(schema_signature(out_schema), schema_signature(in_schema)):
                    return
            except Exception:
                pass

        src_handle = source.port(kind, "out", out_port)
        dst_handle = target.port(kind, "in", in_port)
        if src_handle is None or dst_handle is None:
            return
        try:
            src_handle.connect_to(dst_handle)
        except Exception:
            pass

    def nodes(self) -> list[BaseNode]:
        return [n for n in self.node_graph.all_nodes() if isinstance(n, BaseNode)]

    def operator_nodes(self) -> list[GenericNode]:
        return [n for n in self.node_graph.all_nodes() if isinstance(n, GenericNode)]

    def to_operator_graph(self) -> OperatorGraph:
        """
        Export the current NodeGraphQt graph to an OperatorGraph.

        This is a one-way conversion for deployment/execution.
        """
        graph = OperatorGraph()

        for node in self.operator_nodes():
            state: dict[str, Any] = {}
            spec = node.spec
            if not isinstance(spec, F8OperatorSpec):
                try:
                    spec = OperatorSpecRegistry.instance().get(spec.operatorClass)
                except Exception:
                    continue

            for field in spec.states or []:
                try:
                    value = node.get_property(field.name)
                    schema_type = getattr(field.valueSchema, "type", None) if field.valueSchema else None
                    if schema_type == F8PrimitiveTypeEnum.integer:
                        state[field.name] = int(value) if value is not None and value != "" else None
                    elif schema_type == F8PrimitiveTypeEnum.number:
                        state[field.name] = float(value) if value is not None and value != "" else None
                    elif schema_type == F8PrimitiveTypeEnum.boolean:
                        state[field.name] = bool(value)
                    else:
                        state[field.name] = value
                except Exception:
                    pass

            instance = OperatorInstance.from_spec(spec, id=node.id, state=state)

            try:
                pos = node.pos()
                renderer_props = instance.spec.rendererProps or {}
                renderer_props["pos"] = [float(pos[0]), float(pos[1])]
                instance.spec.rendererProps = renderer_props
            except Exception:
                pass

            graph.add_node(instance)

        def raw_name_for_port(target_node: GenericNode, kind: str, direction: str, port_obj: Any) -> str | None:
            mapping = {
                ("exec", "in"): target_node.port_handles.exec_in,
                ("exec", "out"): target_node.port_handles.exec_out,
                ("data", "in"): target_node.port_handles.data_in,
                ("data", "out"): target_node.port_handles.data_out,
                ("state", "in"): target_node.port_handles.state_in,
                ("state", "out"): target_node.port_handles.state_out,
            }.get((kind, direction))
            if not mapping:
                return None
            for raw_name, handle in mapping.items():
                if handle is port_obj:
                    return raw_name
            return None

        for source in self.operator_nodes():
            for raw_out, out_port in source.port_handles.exec_out.items():
                for in_port in out_port.connected_ports():
                    target = in_port.node()
                    if not isinstance(target, GenericNode):
                        continue
                    raw_in = raw_name_for_port(target, "exec", "in", in_port)
                    if raw_in is None:
                        continue
                    try:
                        graph.connect_exec(source.id, raw_out, target.id, raw_in)
                    except Exception:
                        pass

            for raw_out, out_port in source.port_handles.data_out.items():
                for in_port in out_port.connected_ports():
                    target = in_port.node()
                    if not isinstance(target, GenericNode):
                        continue
                    raw_in = raw_name_for_port(target, "data", "in", in_port)
                    if raw_in is None:
                        continue
                    try:
                        graph.connect_data(source.id, raw_out, target.id, raw_in)
                    except Exception:
                        pass

            for raw_out, out_port in source.port_handles.state_out.items():
                for in_port in out_port.connected_ports():
                    target = in_port.node()
                    if not isinstance(target, GenericNode):
                        continue
                    raw_in = raw_name_for_port(target, "state", "in", in_port)
                    if raw_in is None:
                        continue
                    try:
                        graph.connect_state(source.id, raw_out, target.id, raw_in)
                    except Exception:
                        pass

        return graph
