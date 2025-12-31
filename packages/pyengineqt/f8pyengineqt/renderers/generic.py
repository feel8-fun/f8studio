from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from f8pysdk import F8PrimitiveTypeEnum, F8StateFieldAccess, F8DataTypeSchema

from ..graph.operator_instance import OperatorInstance
from ..operators.operator_registry import OperatorSpecRegistry

from NodeGraphQt import BaseNode


@dataclass
class PortHandles:
    exec_in: dict[str, Any] = field(default_factory=dict)
    exec_out: dict[str, Any] = field(default_factory=dict)
    data_in: dict[str, Any] = field(default_factory=dict)
    data_out: dict[str, Any] = field(default_factory=dict)
    state_in: dict[str, Any] = field(default_factory=dict)
    state_out: dict[str, Any] = field(default_factory=dict)


class OperatorNodeBase(BaseNode):  # type: ignore[misc]
    """
    Base NodeGraphQt node that renders ports from an OperatorInstance spec.

    Supports two construction modes:
    - `__init__(instance)` for programmatic creation (our engine-side graph model).
    - `__init__()` for NodeGraphQt's internal factory (eg. NodesPaletteWidget drag/drop),
      where `OPERATOR_CLASS` must be set on the concrete subclass.
    """

    __identifier__ = "fun.feel8.op.renderer.base"
    NODE_NAME = "Operator"
    instance: OperatorInstance

    OPERATOR_CLASS: str = ""

    def __init__(self, instance: OperatorInstance | None = None) -> None:
        super().__init__()
        if instance is None:
            operator_class = getattr(self, "OPERATOR_CLASS", "") or ""
            if not operator_class:
                raise RuntimeError("OPERATOR_CLASS not set on OperatorNodeBase subclass")

            template = OperatorSpecRegistry.instance().get(operator_class)
            instance = OperatorInstance.from_spec(template, id=self.id)

        self.instance = instance
        self.port_handles = PortHandles()

        self.set_name(instance.spec.label or instance.id)  # type: ignore[attr-defined]
        self._build_ports()
        self._apply_state_properties()

    def port(self, kind: str, direction: str, name: str) -> Any:
        mapping = {
            ("exec", "in"): self.port_handles.exec_in,
            ("exec", "out"): self.port_handles.exec_out,
            ("data", "in"): self.port_handles.data_in,
            ("data", "out"): self.port_handles.data_out,
            ("state", "in"): self.port_handles.state_in,
            ("state", "out"): self.port_handles.state_out,
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
        if hasattr(self, "clear_ports"):
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
                handle = self.add_input(f"{field.name} [in]", color=(120, 200, 200))  # type: ignore[attr-defined]
                self.port_handles.state_in[field.name] = handle
            if access in (F8StateFieldAccess.ro, F8StateFieldAccess.rw, F8StateFieldAccess.init):
                handle = self.add_output(f"{field.name} [out]", color=(120, 200, 200))  # type: ignore[attr-defined]
                self.port_handles.state_out[field.name] = handle

    def _apply_state_properties(self) -> None:
        assert self.instance is not None
        if not hasattr(self, "create_property"):
            return
        for field in self.instance.spec.states or []:
            value = self.instance.state.get(field.name, _schema_default(field.valueSchema))
            try:
                self.create_property(field.name, value)  # type: ignore[attr-defined]
            except Exception:
                # Fall back to basic property assignment when NodeGraphQt API changes.
                setattr(self, field.name, value)


class GenericOperatorNode(OperatorNodeBase):  # type: ignore[misc]
    """
    Default renderer node variant.

    Used for both programmatic builds and NodeGraphQt internal registry creation.
    """

    __identifier__ = "fun.feel8.op.renderer.generic"


class UiOperatorNode(OperatorNodeBase):  # type: ignore[misc]
    """
    UI-focused renderer node variant (placeholder for style differences).

    Keeping it separate allows rendererClass/rendererId to select node visuals.
    """

    __identifier__ = "fun.feel8.op.renderer.ui"


def _schema_default(schema: F8DataTypeSchema | None) -> Any | None:
    if not schema:
        return None
    return getattr(schema.root, "default", None)


def _schema_type(schema: F8DataTypeSchema | None) -> str | None:
    if not schema:
        return None
    return getattr(schema.root, "type", None)


def _color_for_schema(schema: F8DataTypeSchema | None) -> tuple[int, int, int]:
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
