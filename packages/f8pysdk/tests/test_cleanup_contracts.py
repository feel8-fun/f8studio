from __future__ import annotations

import msgspec
import numpy as np
import pytest

from f8pysdk.codec import copy_model, dump_json
from f8pysdk.specs import (
    F8DataPortSpec, F8StateSpec, F8StateAccess, F8ServiceSpec, F8OperatorSpec,
    array_schema, number_schema, video_frame_port,
)


@pytest.mark.parametrize("old_field,value", [("payloadKind", "json"), ("delivery", "fifo"), ("valueSchema", {"type": "any"})])
def test_data_port_rejects_stale_mirrors(old_field: str, value: object) -> None:
    payload = msgspec.to_builtins(video_frame_port(name="video"))
    payload[old_field] = value
    with pytest.raises(msgspec.ValidationError, match="unknown field"):
        msgspec.convert(payload, type=F8DataPortSpec)


def test_data_port_requires_explicit_payload() -> None:
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"name": "old", "valueSchema": {"type": "any"}}, type=F8DataPortSpec)


def test_authoring_rejects_old_control_and_exec_ports() -> None:
    with pytest.raises(msgspec.ValidationError, match="unknown field"):
        msgspec.convert({"name": "x", "access": "rw", "valueSchema": {"type": "string"}, "uiControl": "textarea"}, type=F8StateSpec)
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert({"serviceClass": "demo", "operatorClass": "op", "label": "Op", "execInPorts": ["exec"]}, type=F8OperatorSpec)


def test_model_copy_is_deep_when_requested() -> None:
    original = F8ServiceSpec(serviceClass="demo", label="Demo", stateFields=[
        F8StateSpec(name="values", access=F8StateAccess.rw, valueSchema=array_schema(items=number_schema(), default=[1])),
    ])
    copied = copy_model(original, deep=True)
    assert copied.stateFields is not original.stateFields
    assert copied.stateFields[0].valueSchema.default is not original.stateFields[0].valueSchema.default
    copied.stateFields[0].valueSchema.default.append(2)
    assert original.stateFields[0].valueSchema.default == [1]


def test_json_conversion_rejects_unknown_and_cyclic_values() -> None:
    with pytest.raises(TypeError, match="unsupported JSON"):
        dump_json(object())
    value: list[object] = []
    value.append(value)
    with pytest.raises(ValueError, match="cyclic"):
        dump_json(value)
    assert dump_json({"integer": np.int64(3), "float": np.float32(2.5), "missing": msgspec.UNSET, "null": None}) == {"integer": 3, "float": 2.5, "null": None}


def test_json_conversion_preserves_schema_tags() -> None:
    original = video_frame_port(name="video")
    payload = dump_json(original)
    assert payload["payload"]["metadataSchema"]["type"] == "object"
    assert msgspec.convert(payload, type=F8DataPortSpec) == original
