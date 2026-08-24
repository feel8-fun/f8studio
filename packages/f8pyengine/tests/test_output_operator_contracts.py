from __future__ import annotations

import msgspec

from f8pyengine.operators.buttplug_out import ButtplugOutRuntimeNode
from f8pyengine.operators.handy_out import HandyOutRuntimeNode
from f8pyengine.operators.lovense_out import LovenseOutRuntimeNode
from f8pyengine.operators.serial_out import SerialOutRuntimeNode
from f8pyengine.operators.tcode import AXES, TCodeRuntimeNode
from f8pysdk.generated import F8NumberTypeSchema


def _data_input_spec(runtime_node: type, port_name: str):
    for port in list(runtime_node.SPEC.dataInPorts or []):
        if str(port.name) == port_name:
            return port
    raise AssertionError(f"missing data input port: {runtime_node.__name__}.{port_name}")


def _state_spec(runtime_node: type, field_name: str):
    for state in list(runtime_node.SPEC.stateFields or []):
        if str(state.name) == field_name:
            return state
    raise AssertionError(f"missing state field: {runtime_node.__name__}.{field_name}")


def _assert_normalized_number_port(runtime_node: type, port_name: str) -> None:
    port = _data_input_spec(runtime_node, port_name)
    schema = port.valueSchema
    assert isinstance(schema, F8NumberTypeSchema)
    assert float(schema.minimum) >= 0.0
    assert float(schema.maximum) <= 1.0
    assert float(schema.minimum) < float(schema.maximum)


def test_physical_position_outputs_accept_normalized_values() -> None:
    _assert_normalized_number_port(LovenseOutRuntimeNode, "position")
    _assert_normalized_number_port(ButtplugOutRuntimeNode, "position")
    _assert_normalized_number_port(HandyOutRuntimeNode, "value")


def test_tcode_axes_accept_normalized_values() -> None:
    assert TCodeRuntimeNode.SPEC.paletteCategory == "f8.pyengine.signal"
    for axis in AXES:
        _assert_normalized_number_port(TCodeRuntimeNode, axis)


def test_tcode_exposes_all_sr6_axes_on_the_graph_node() -> None:
    visible = {str(port.name) for port in TCodeRuntimeNode.SPEC.dataInPorts or [] if port.showOnNode is True}
    assert {"L0", "L1", "L2", "R0", "R1", "R2"} <= visible
    assert {"V0", "V1", "A0", "A1"}.isdisjoint(visible)


def test_sensitive_output_configuration_is_redacted_on_publish() -> None:
    assert _state_spec(LovenseOutRuntimeNode, "commandUrl").redactOnPublish is True
    assert _state_spec(ButtplugOutRuntimeNode, "wsUrl").redactOnPublish is True
    assert _state_spec(HandyOutRuntimeNode, "connectionKey").redactOnPublish is True
    assert _state_spec(HandyOutRuntimeNode, "baseUrl").redactOnPublish is True


def test_every_physical_output_has_explicit_enabled_state() -> None:
    for runtime_node in (LovenseOutRuntimeNode, ButtplugOutRuntimeNode, HandyOutRuntimeNode, SerialOutRuntimeNode):
        enabled = _state_spec(runtime_node, "enabled")
        assert enabled.required is True
        assert not isinstance(enabled.valueSchema, msgspec.UnsetType)


def test_serial_output_is_disarmed_by_default() -> None:
    enabled = _state_spec(SerialOutRuntimeNode, "enabled")
    assert enabled.valueSchema.default is False
