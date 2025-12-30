from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Type

from f8pysdk import F8PrimitiveTypeEnum, F8StateFieldAccess
from f8pysdk.generated.data import Schema

from .operator_instance import OperatorInstance

try:  # Optional import so the module can be imported without Qt installed.
    from NodeGraphQt import BaseNode, NodeGraph
except Exception:  # noqa: BLE001
    BaseNode = None  # type: ignore[assignment]
    NodeGraph = None  # type: ignore[assignment]


class OperatorRendererRegistry:
    """Registry for renderer classes keyed by rendererClass."""

    def __init__(self) -> None:
        self._renderers: dict[str, Type['BaseOperatorRenderer']] = {}

    def register(
        self, renderer_key: str, renderer_cls: Type['BaseOperatorRenderer'], *, overwrite: bool = False
    ) -> None:
        if renderer_key in self._renderers and not overwrite:
            raise ValueError(f'renderer "{renderer_key}" already registered')
        self._renderers[renderer_key] = renderer_cls

    def unregister(self, renderer_key: str) -> None:
        self._renderers.pop(renderer_key, None)

    def get(self, renderer_key: str) -> Type['BaseOperatorRenderer']:
        try:
            return self._renderers[renderer_key]
        except KeyError as exc:  # noqa: PERF203
            raise KeyError(f'renderer "{renderer_key}" not found') from exc

    def keys(self) -> list[str]:
        return list(self._renderers.keys())


NodeBase = BaseNode or object


@dataclass
class PortHandles:
    exec_in: dict[str, Any] = field(default_factory=dict)
    exec_out: dict[str, Any] = field(default_factory=dict)
    data_in: dict[str, Any] = field(default_factory=dict)
    data_out: dict[str, Any] = field(default_factory=dict)
    state_in: dict[str, Any] = field(default_factory=dict)
    state_out: dict[str, Any] = field(default_factory=dict)


class OperatorNode(NodeBase):  # type: ignore[misc]
    """
    Generic NodeGraphQt node that renders ports from an OperatorInstance spec.

    The class is intentionally lightweight so renderer implementations can subclass or
    wrap it to customize visuals without changing graph wiring code.
    """

    __identifier__ = 'feel8.operator'
    NODE_NAME = 'Operator'

    def __init__(self) -> None:
        if BaseNode is None:
            raise ImportError('NodeGraphQt is required to instantiate OperatorNode')
        super().__init__()
        self.instance: OperatorInstance | None = None
        self.port_handles = PortHandles()

    # Public API ----------------------------------------------------------
    def apply_instance(self, instance: OperatorInstance) -> None:
        """Attach an OperatorInstance and rebuild ports/properties."""
        self.instance = instance
        if hasattr(self, 'set_name'):
            self.set_name(instance.spec.label or instance.id)  # type: ignore[attr-defined]
        tooltip = instance.spec.description or instance.operator_class
        if hasattr(self, 'set_tooltip'):
            self.set_tooltip(tooltip)  # type: ignore[attr-defined]
        self._build_ports()
        self._apply_state_properties()

    def port(self, kind: str, direction: str, name: str) -> Any:
        mapping = {
            ('exec', 'in'): self.port_handles.exec_in,
            ('exec', 'out'): self.port_handles.exec_out,
            ('data', 'in'): self.port_handles.data_in,
            ('data', 'out'): self.port_handles.data_out,
            ('state', 'in'): self.port_handles.state_in,
            ('state', 'out'): self.port_handles.state_out,
        }.get((kind, direction))
        if mapping is None:
            return None
        return mapping.get(name)

    # Builders ------------------------------------------------------------
    def _build_ports(self) -> None:
        assert self.instance is not None
        self._clear_ports()
        self._build_exec_ports()
        self._build_data_ports()
        self._build_state_ports()

    def _clear_ports(self) -> None:
        self.port_handles = PortHandles()
        if hasattr(self, 'clear_ports'):
            try:
                self.clear_ports()  # type: ignore[attr-defined]
            except Exception:
                pass

    def _build_exec_ports(self) -> None:
        assert self.instance is not None
        for port in self.instance.spec.execInPorts or []:
            handle = self.add_input(port, color=(200, 200, 50))  # type: ignore[attr-defined]
            self.port_handles.exec_in[port] = handle
        for port in self.instance.spec.execOutPorts or []:
            handle = self.add_output(port, color=(200, 200, 50))  # type: ignore[attr-defined]
            self.port_handles.exec_out[port] = handle

    def _build_data_ports(self) -> None:
        assert self.instance is not None
        for port in self.instance.spec.dataInPorts or []:
            handle = self.add_input(port.name, color=_color_for_schema(port.valueSchema))  # type: ignore[attr-defined]
            self.port_handles.data_in[port.name] = handle
        for port in self.instance.spec.dataOutPorts or []:
            handle = self.add_output(port.name, color=_color_for_schema(port.valueSchema))  # type: ignore[attr-defined]
            self.port_handles.data_out[port.name] = handle

    def _build_state_ports(self) -> None:
        assert self.instance is not None
        for field in self.instance.spec.states or []:
            access = field.access or F8StateFieldAccess.ro
            if access in (F8StateFieldAccess.wo, F8StateFieldAccess.rw):
                handle = self.add_input(f'{field.name} [in]', color=(120, 200, 200))  # type: ignore[attr-defined]
                self.port_handles.state_in[field.name] = handle
            if access in (F8StateFieldAccess.ro, F8StateFieldAccess.rw, F8StateFieldAccess.init):
                handle = self.add_output(f'{field.name} [out]', color=(120, 200, 200))  # type: ignore[attr-defined]
                self.port_handles.state_out[field.name] = handle

    def _apply_state_properties(self) -> None:
        assert self.instance is not None
        if not hasattr(self, 'create_property'):
            return
        for field in self.instance.spec.states or []:
            value = self.instance.state.get(field.name, _schema_default(field.valueSchema))
            try:
                self.create_property(field.name, value)  # type: ignore[attr-defined]
            except Exception:
                # Fall back to basic property assignment when NodeGraphQt API changes.
                setattr(self, field.name, value)


class BaseOperatorRenderer:
    """Base renderer interface for building NodeGraphQt nodes."""

    node_cls: Type[OperatorNode] = OperatorNode

    def build_node(
        self,
        graph: 'NodeGraph',
        instance: OperatorInstance,
        *,
        pos: tuple[float, float] | None = None,
    ) -> OperatorNode:
        raise NotImplementedError


class GenericOperatorRenderer(BaseOperatorRenderer):
    """Default renderer that maps OperatorSpec ports to NodeGraphQt ports."""

    def _ensure_registered(self, graph: 'NodeGraph') -> None:
        try:
            graph.register_node(self.node_cls)
        except Exception:
            # NodeGraphQt raises if already registered; safe to ignore.
            pass

    def _node_type_name(self) -> str:
        return f'{self.node_cls.__identifier__}.{self.node_cls.__name__}'

    def build_node(
        self,
        graph: 'NodeGraph',
        instance: OperatorInstance,
        *,
        pos: tuple[float, float] | None = None,
    ) -> OperatorNode:
        self._ensure_registered(graph)
        node_name = instance.spec.label or instance.id
        try:
            node = graph.create_node(self._node_type_name(), name=node_name, pos=pos)
        except Exception:
            node = self.node_cls()
            if hasattr(graph, 'add_node'):
                graph.add_node(node)
            if pos and hasattr(node, 'set_pos'):
                node.set_pos(pos)  # type: ignore[attr-defined]
            if hasattr(node, 'set_name'):
                node.set_name(node_name)  # type: ignore[attr-defined]

        node.apply_instance(instance)
        return node


def _schema_default(schema: Schema | None) -> Any | None:
    if not schema:
        return None
    return getattr(schema.root, "default", None)


def _schema_type(schema: Schema | None) -> str | None:
    if not schema:
        return None
    return getattr(schema.root, "type", None)


def _color_for_schema(schema: Schema | None) -> tuple[int, int, int]:
    type_name = _schema_type(schema)
    palette = {
        F8PrimitiveTypeEnum.boolean: (170, 170, 170),
        F8PrimitiveTypeEnum.integer: (220, 140, 60),
        F8PrimitiveTypeEnum.number: (90, 160, 255),
        F8PrimitiveTypeEnum.string: (120, 200, 120),
        "array": (160, 120, 200),
        "object": (180, 120, 200),
        "any": (200, 200, 200),
    }
    if isinstance(type_name, F8PrimitiveTypeEnum):
        return palette.get(type_name, (200, 200, 200))
    if isinstance(type_name, str):
        return palette.get(type_name, (200, 200, 200))
    return (200, 200, 200)
