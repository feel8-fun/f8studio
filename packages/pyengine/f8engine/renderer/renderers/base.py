from typing import Union

from ...operator import Access, OperatorInstance
import dearpygui.dearpygui as dpg


class BaseOpRenderer:
    """Base renderer with helper methods."""

    node_id: Union[str, int]
    instance: OperatorInstance

    def __init__(self, node_id: Union[str, int], instance: OperatorInstance) -> None:
        self.node_id = node_id
        self.instance = instance

        self._add_exec_pins()
        self._add_data_pins()
        self._add_state_pins()

    def _add_exec_pins(self) -> None:
        for in_exec in self.instance.spec.execInPorts:
            with dpg.node_attribute(
                tag=f"{self.node_id}_in_exec_{in_exec}",
                parent=self.node_id,
                attribute_type=dpg.mvNode_Attr_Input,
                shape=dpg.mvNode_PinShape_Triangle,
            ):
                dpg.add_text(in_exec)

        for out_exec in self.instance.spec.execOutPorts:
            with dpg.node_attribute(
                tag=f"{self.node_id}_out_exec_{out_exec}",
                parent=self.node_id,
                attribute_type=dpg.mvNode_Attr_Output,
                shape=dpg.mvNode_PinShape_Triangle,
            ):
                dpg.add_text(out_exec)

    def _add_data_pins(self) -> None:
        for in_data in self.instance.spec.dataInPorts:
            with dpg.node_attribute(
                tag=f"{self.node_id}_in_data_{in_data.name}",
                parent=self.node_id,
                attribute_type=dpg.mvNode_Attr_Input,
                shape=dpg.mvNode_PinShape_Circle,
            ):
                dpg.add_text(in_data.name)

        for out_data in self.instance.spec.dataOutPorts:
            with dpg.node_attribute(
                tag=f"{self.node_id}_out_data_{out_data.name}",
                parent=self.node_id,
                attribute_type=dpg.mvNode_Attr_Output,
                shape=dpg.mvNode_PinShape_Circle,
            ):
                dpg.add_text(out_data.name)

    def _add_state_pins(self) -> None:
        for state_field in self.instance.spec.states or []:
            access = state_field.access or Access.ro
            if access in (Access.wo, Access.rw):
                with dpg.node_attribute(
                    tag=f"{self.node_id}_in_state_{state_field.name}",
                    parent=self.node_id,
                    attribute_type=dpg.mvNode_Attr_Input,
                    shape=dpg.mvNode_PinShape_Quad,
                ):
                    dpg.add_text(state_field.name)
            if access in (Access.ro, Access.rw, Access.init):
                with dpg.node_attribute(
                    tag=f"{self.node_id}_out_state_{state_field.name}",
                    parent=self.node_id,
                    attribute_type=dpg.mvNode_Attr_Output,
                    shape=dpg.mvNode_PinShape_Quad,
                ):
                    dpg.add_text(state_field.name)
