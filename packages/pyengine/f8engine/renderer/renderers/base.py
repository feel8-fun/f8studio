from typing import Union, Literal

from dataclasses import dataclass
from ...operator import Type, Access, StateField, OperatorInstance
import dearpygui.dearpygui as dpg

@dataclass
class PortUserData:
    node_id: Union[str, int]
    port: str
    kind: Literal["exec", "data", "state"]
    direction: Literal["in", "out", "field"]

class BaseOpRenderer:
    """Base renderer with helper methods."""

    node_id: Union[str, int]
    instance: OperatorInstance

    def __init__(self, node_id: Union[str, int], instance: OperatorInstance) -> None:
        self.node_id = node_id
        self.instance = instance
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, parent=self.node_id):
            dpg.add_text("--  EXEC --")
        self._add_exec_pins()
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, parent=self.node_id):
            dpg.add_text("--  DATA --")
        self._add_data_pins()
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, parent=self.node_id):
            dpg.add_text("-- STATE --")
        self._add_state_pins()

    def _add_exec_pins(self) -> None:
        for in_exec in self.instance.spec.execInPorts:
            with dpg.node_attribute(
                tag=f"{self.node_id}_in_exec_{in_exec}",
                parent=self.node_id,
                attribute_type=dpg.mvNode_Attr_Input,
                shape=dpg.mvNode_PinShape_TriangleFilled,
                user_data=PortUserData(node_id=self.node_id, port=in_exec, kind="exec", direction="in"
            )
            ):
                dpg.add_text(in_exec)

        for out_exec in self.instance.spec.execOutPorts:
            with dpg.node_attribute(
                tag=f"{self.node_id}_out_exec_{out_exec}",
                parent=self.node_id,
                attribute_type=dpg.mvNode_Attr_Output,
                shape=dpg.mvNode_PinShape_TriangleFilled,
                user_data=PortUserData(node_id=self.node_id, port=out_exec, kind="exec", direction="out"),
            ):
                dpg.add_text(out_exec)

    def _add_data_pins(self) -> None:
        for in_data in self.instance.spec.dataInPorts:
            with dpg.node_attribute(
                tag=f"{self.node_id}_in_data_{in_data.name}",
                parent=self.node_id,
                attribute_type=dpg.mvNode_Attr_Input,
                shape=dpg.mvNode_PinShape_CircleFilled,
                user_data=PortUserData(node_id=self.node_id, port=in_data.name, kind="data", direction="in"),
            ):
                dpg.add_text(in_data.name)

        for out_data in self.instance.spec.dataOutPorts:
            with dpg.node_attribute(
                tag=f"{self.node_id}_out_data_{out_data.name}",
                parent=self.node_id,
                attribute_type=dpg.mvNode_Attr_Output,
                shape=dpg.mvNode_PinShape_CircleFilled,
                user_data=PortUserData(node_id=self.node_id, port=out_data.name, kind="data", direction="out"),
            ):
                dpg.add_text(out_data.name)

    def _add_state_field(self, state_field: StateField) -> None:
        user_data = PortUserData(node_id=self.node_id, port=state_field.name, kind="state", direction="field")
        if state_field.type == Type.bool:
            dpg.add_checkbox(
                default_value=False, tag=f"{self.node_id}_state_field_{state_field.name}", user_data=user_data
            )
        elif state_field.type == Type.int:
            dpg.add_input_int(
                default_value=0,
                width=100,
                tag=f"{self.node_id}_state_field_{state_field.name}",
                min_value=state_field.minimum if state_field.minimum is not None else 0,
                max_value=state_field.maximum if state_field.maximum is not None else 100,
                step=state_field.step if state_field.step is not None else 1,
                user_data=user_data,
            )
        elif state_field.type == Type.float:
            dpg.add_input_float(
                default_value=0.0,
                width=100,
                tag=f"{self.node_id}_state_field_{state_field.name}",
                min_value=state_field.minimum if state_field.minimum is not None else 0.0,
                max_value=state_field.maximum if state_field.maximum is not None else 100.0,
                step=state_field.step if state_field.step is not None else 1.0,
                user_data=user_data,
            )
        elif state_field.type == Type.string:
            if state_field.enumValues and len(state_field.enumValues) > 0:
                dpg.add_combo(
                    items=state_field.enumValues,
                    default_value=state_field.default if state_field.default else state_field.enumValues[0],
                    width=100,
                    tag=f"{self.node_id}_state_field_{state_field.name}",
                    user_data=user_data,
                )
            else:
                dpg.add_input_text(
                    default_value="",
                    width=100,
                    tag=f"{self.node_id}_state_field_{state_field.name}",
                    user_data=user_data,
                )
        elif state_field.type == Type.object:
            dpg.add_button(label="Edit", tag=f"{self.node_id}_state_field_{state_field.name}", user_data=user_data)

    def _add_state_pins(self) -> None:
        for state_field in self.instance.spec.states or []:
            access = state_field.access or Access.ro
            if access in (Access.wo, Access.rw):
                with dpg.node_attribute(
                    tag=f"{self.node_id}_in_state_{state_field.name}",
                    user_data=PortUserData(node_id=self.node_id, port=state_field.name, kind="state", direction="in"),
                    parent=self.node_id,
                    attribute_type=dpg.mvNode_Attr_Input,
                    shape=dpg.mvNode_PinShape_QuadFilled,
                ):
                    dpg.add_text(state_field.name)
                    self._add_state_field(state_field)

            if access in (Access.ro, Access.rw, Access.init):
                with dpg.node_attribute(
                    tag=f"{self.node_id}_out_state_{state_field.name}",
                    user_data=PortUserData(node_id=self.node_id, port=state_field.name, kind="state", direction="out"),
                    parent=self.node_id,
                    attribute_type=dpg.mvNode_Attr_Output,
                    shape=dpg.mvNode_PinShape_QuadFilled,
                ):
                    dpg.add_text(state_field.name)
                    if access != Access.rw:
                        self._add_state_field(state_field)
