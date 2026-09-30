from __future__ import annotations
from f8pysdk.specs import F8DataPayloadSpec, F8DataPortPayloadKind

import pytest

from f8pysdk.specs import F8DataPortSpec, F8StateAccess, F8StateSpec, any_schema, string_schema
from f8pysdk.editor_assist_protocol import validate_editor_assist_spec


def _editor_assist_payload() -> dict[str, object]:
    return {
        "version": 1,
        "language": "python",
        "python": {
            "support_files": {"f8_script_api.pyi": "class F8Ctx:\n    ...\n"},
            "overlay_prefix": "from f8_script_api import *\n",
            "dynamic_bindings": {
                "inputs": {
                    "enabled": True,
                    "source": "data_in_ports",
                    "type_name": "F8Inputs",
                    "module_name": "f8_dynamic_inputs",
                    "schema_mode": "basic_recursive",
                    "access_mode": "object_and_mapping",
                },
                "states": {
                    "enabled": True,
                    "source": "state_fields",
                    "type_name": "F8States",
                    "module_name": "f8_dynamic_states",
                    "schema_mode": "basic_recursive",
                    "access_mode": "object_and_mapping",
                },
                "outputs": {
                    "enabled": True,
                    "source": "data_out_ports",
                    "type_name": "F8Outputs",
                    "module_name": "f8_dynamic_outputs",
                    "schema_mode": "basic_recursive",
                    "access_mode": "object_and_mapping",
                },
            },
        },
    }


def test_validate_editor_assist_spec_accepts_valid_payload() -> None:
    spec = validate_editor_assist_spec(_editor_assist_payload())
    assert int(spec.version) == 1
    assert str(spec.language or "") == "python"
    assert str(spec.python.overlay_prefix or "").startswith("from f8_script_api import")


def test_state_spec_accepts_editor_assist_field() -> None:
    state = F8StateSpec(
        name="code",
        label="Code",
        description="",
        valueSchema=string_schema(default=""),
        access=F8StateAccess.rw,
        redactOnPublish=True,
        editorAssist=validate_editor_assist_spec(_editor_assist_payload()),
    )
    assert state.editorAssist is not None
    assert int(state.editorAssist.version) == 1
    assert bool(state.redactOnPublish) is True


def test_state_spec_still_forbids_unknown_fields() -> None:
    with pytest.raises(Exception):
        _ = F8StateSpec(
            name="code",
            label="Code",
            description="",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            unknownX=1,  # type: ignore[call-arg]
        )


def test_data_port_spec_rejects_editor_assist_field() -> None:
    with pytest.raises(Exception):
        _ = F8DataPortSpec(
            name="x",
            payload=F8DataPayloadSpec(kind=F8DataPortPayloadKind.json, valueSchema=any_schema()),
            editorAssist=validate_editor_assist_spec(_editor_assist_payload()),  # type: ignore[call-arg]
        )


def test_validate_editor_assist_spec_rejects_invalid_states_source() -> None:
    payload = _editor_assist_payload()
    python_payload = payload.get("python")
    assert isinstance(python_payload, dict)
    dynamic_bindings = python_payload.get("dynamic_bindings")
    assert isinstance(dynamic_bindings, dict)
    states_binding = dynamic_bindings.get("states")
    assert isinstance(states_binding, dict)
    states_binding["source"] = "data_in_ports"

    with pytest.raises(Exception):
        _ = validate_editor_assist_spec(payload)


def test_validate_editor_assist_spec_rejects_invalid_outputs_source() -> None:
    payload = _editor_assist_payload()
    python_payload = payload.get("python")
    assert isinstance(python_payload, dict)
    dynamic_bindings = python_payload.get("dynamic_bindings")
    assert isinstance(dynamic_bindings, dict)
    outputs_binding = dynamic_bindings.get("outputs")
    assert isinstance(outputs_binding, dict)
    outputs_binding["source"] = "state_fields"

    with pytest.raises(Exception):
        _ = validate_editor_assist_spec(payload)
