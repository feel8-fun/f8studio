from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from f8pysdk import (
    F8PrimitiveTypeEnum,
    F8StateFieldAccess,
    F8DataTypeSchema,
    F8OperatorSpec,
    F8ServiceSpec,
    schema_default,
    schema_type,
)

from ..operators.operator_registry import OperatorSpecRegistry

from NodeGraphQt import BaseNode


from NodeGraphQt.qgraphics.node_base import NodeItem
from NodeGraphQt.constants import NodePropWidgetEnum
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from .port_painter import draw_exec_port, draw_nothing, draw_square_port

from qtpy import QtCore, QtWidgets

EMPTY_PORT_COLOR = (0, 0, 0, 0)
EXEC_PORT_COLOR = (230, 230, 230)
DATA_PORT_COLOR = (150, 150, 150)
STATE_PORT_COLOR = (200, 200, 50)


@dataclass
class PortHandles:
    exec_in: dict[str, Any] = field(default_factory=dict)
    exec_out: dict[str, Any] = field(default_factory=dict)
    data_in: dict[str, Any] = field(default_factory=dict)
    data_out: dict[str, Any] = field(default_factory=dict)
    state_in: dict[str, Any] = field(default_factory=dict)
    state_out: dict[str, Any] = field(default_factory=dict)


class GenericNode(BaseNode):  # type: ignore[misc]
    """
    Base NodeGraphQt node for editing OperatorSpecs in NodeGraphQt.

    This node is editor-only: it builds ports + properties from an OperatorSpec
    template fetched from `OperatorSpecRegistry` using `OPERATOR_CLASS`.
    """

    __identifier__ = "feel8.renderer"
    NODE_NAME = "Generic"

    SPEC_KEY: str = ""

    spec: F8OperatorSpec | F8ServiceSpec

    def __init__(self) -> None:
        super().__init__()

        spec_key = self.SPEC_KEY
        # Lookup order: OperatorSpecRegistry -> ServiceSpecRegistry -> UserSpecsLibrary.

        if OperatorSpecRegistry.instance().has(spec_key):
            self.spec = OperatorSpecRegistry.instance().get(spec_key)
        else:
            raise ValueError(f"Spec [{spec_key}] not found.")

        self.port_handles = PortHandles()

        self.set_name(self.spec.label)  # type: ignore[attr-defined]
        self._build_ports()
        self._apply_state_properties()

    def _add_spacer_port(self, is_input: bool, name="") -> None:
        if is_input:

            self.add_input(
                name=name, color=EMPTY_PORT_COLOR, display_name=False, painter_func=draw_nothing, locked=True
            )
        else:
            self.add_output(
                name=name, color=EMPTY_PORT_COLOR, display_name=False, painter_func=draw_nothing, locked=True
            )

    def _align_port_rows(self) -> None:
        in_count = len(self.inputs())
        out_count = len(self.outputs())

        diff = in_count - out_count

        for i in range(abs(diff)):
            self._add_spacer_port(is_input=diff < 0)

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
        self._clear_ports()
        self._build_exec_ports()
        self._align_port_rows()
        self._build_data_ports()
        self._align_port_rows()
        self._build_state_ports()

    def _clear_ports(self) -> None:
        for p in self.inputs():
            self.delete_input(p)
        for p in self.outputs():
            self.delete_output(p)

        self.port_handles = PortHandles()

    def _build_exec_ports(self) -> None:
        for port in self.spec.execInPorts or []:
            handle = self.add_input(f"[E]{port}", color=EXEC_PORT_COLOR, painter_func=draw_exec_port)  # type: ignore[attr-defined]
            self.port_handles.exec_in[port] = handle
        for port in self.spec.execOutPorts or []:
            handle = self.add_output(f"{port}[E]", color=EXEC_PORT_COLOR, painter_func=draw_exec_port)  # type: ignore[attr-defined]
            self.port_handles.exec_out[port] = handle

    def _build_data_ports(self) -> None:
        for port in self.spec.dataInPorts or []:
            handle = self.add_input(f"[D]{port.name}", color=DATA_PORT_COLOR)  # type: ignore[attr-defined]
            self.port_handles.data_in[port.name] = handle
        for port in self.spec.dataOutPorts or []:
            handle = self.add_output(f"{port.name}[D]", color=DATA_PORT_COLOR)  # type: ignore[attr-defined]
            self.port_handles.data_out[port.name] = handle

    def _build_state_ports(self) -> None:
        for field in self.spec.states or []:
            access = field.access or F8StateFieldAccess.ro

            if access == F8StateFieldAccess.ro:
                self._add_spacer_port(is_input=True, name=f"[S]{field.name}")
                handle = self.add_output(
                    name=f"{field.name}[S]",
                    color=STATE_PORT_COLOR,
                    display_name=True,
                    painter_func=draw_square_port,
                )  # type: ignore[attr-defined]
                self.port_handles.state_out[field.name] = handle
            elif access == F8StateFieldAccess.wo:
                handle = self.add_input(
                    name=f"[S]{field.name}",
                    color=STATE_PORT_COLOR,
                    display_name=True,
                    painter_func=draw_square_port,
                )  # type: ignore[attr-defined]
                self.port_handles.state_in[field.name] = handle
                self._add_spacer_port(is_input=False, name=f"{field.name}[S]")
            else:
                handle = self.add_input(
                    name=f"[S]{field.name}",
                    color=STATE_PORT_COLOR,
                    display_name=True,
                    painter_func=draw_square_port,
                )  # type: ignore[attr-defined]
                self.port_handles.state_in[field.name] = handle
                handle = self.add_output(
                    name=f"{field.name}[S]",
                    color=STATE_PORT_COLOR,
                    display_name=True,
                    painter_func=draw_square_port,
                )  # type: ignore[attr-defined]
                self.port_handles.state_out[field.name] = handle

    def _apply_state_properties(self) -> None:
        # Create embedded widgets for state fields and wire them to node properties.

        for field in self.spec.states or []:
            schema = field.valueSchema
            # access = field.access
            # label = field.name
            default_value = schema_default(schema)

            field_type = schema_type(schema)

            if field_type == F8PrimitiveTypeEnum.boolean:
                self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QCHECK_BOX.value)
            elif field_type == F8PrimitiveTypeEnum.integer:
                minimum = getattr(schema, "minimum", None)
                maximum = getattr(schema, "maximum", None)
                if minimum is not None and maximum is not None:
                    self.create_property(
                        field.name,
                        default_value,
                        widget_type=NodePropWidgetEnum.SLIDER.value,
                        range=(int(minimum), int(maximum)),
                    )
                else:
                    self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)
            elif field_type == F8PrimitiveTypeEnum.number:

                minimum = getattr(schema, "minimum", None)
                maximum = getattr(schema, "maximum", None)
                if minimum is not None and maximum is not None:
                    self.create_property(
                        field.name,
                        default_value,
                        widget_type=NodePropWidgetEnum.DOUBLE_SLIDER.value,
                        range=(float(minimum), float(maximum)),
                    )
                else:
                    self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)

            elif field_type == F8PrimitiveTypeEnum.string:
                enum_values = getattr(schema, "enum", None) if schema else None
                code_language = getattr(schema, "language", None) if schema else None
                if enum_values:
                    items = [str(v) for v in enum_values]
                    self.create_property(
                        field.name,
                        default_value,
                        widget_type=NodePropWidgetEnum.QCOMBO_BOX.value,
                        items=items,
                    )
                elif code_language:
                    self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QTEXT_EDIT.value)
                else:
                    self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)


class UiOperatorNode(GenericNode):  # type: ignore[misc]
    """
    UI-focused renderer node variant (placeholder for style differences).

    Keeping it separate allows rendererClass/rendererId to select node visuals.
    """

    __identifier__ = "fun.feel8.op.renderer.ui"
