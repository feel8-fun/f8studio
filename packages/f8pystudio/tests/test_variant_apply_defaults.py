from __future__ import annotations

from typing import Any

from f8pysdk.generated import F8OperatorSchemaVersion, F8OperatorSpec, F8StateAccess, F8StateSpec
from f8pysdk.schema_helpers import string_schema

from f8pystudio.nodegraph.node_graph import F8StudioGraph


class _NodeModelStub:
    def __init__(self) -> None:
        self.properties: dict[str, Any] = {}
        self.custom_properties: dict[str, Any] = {}
        self.f8_sys: dict[str, Any] = {}


class _NodeStub:
    def __init__(self, spec: F8OperatorSpec) -> None:
        self._spec = spec
        self.model = _NodeModelStub()
        self.model.custom_properties["code"] = "BASE_CODE"
        self.model.custom_properties["svcId"] = "seed_svc"

    @property
    def spec(self) -> F8OperatorSpec:
        return self._spec

    @spec.setter
    def spec(self, value: F8OperatorSpec) -> None:
        self._spec = value

    def set_ui_overrides(self, _value: dict[str, object], *, rebuild: bool = True) -> None:
        _ = rebuild

    def sync_from_spec(self) -> None:
        for state_spec in list(self._spec.stateFields or []):
            name = str(state_spec.name or "").strip()
            if not name:
                continue
            if name not in self.model.custom_properties and name not in self.model.properties:
                self.model.custom_properties[name] = None

    def set_property(self, name: str, value: Any, *, push_undo: bool = True) -> None:
        _ = push_undo
        if name in self.model.properties:
            self.model.properties[name] = value
            return
        self.model.custom_properties[name] = value


def test_apply_variant_to_node_uses_variant_state_defaults_for_writable_fields() -> None:
    base_spec = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass="f8.pyengine",
        operatorClass="f8.python_script",
        version="0.0.1",
        label="Python Script",
        stateFields=[
            F8StateSpec(name="code", valueSchema=string_schema(default="BASE"), access=F8StateAccess.rw),
            F8StateSpec(name="svcId", valueSchema=string_schema(default=""), access=F8StateAccess.ro),
        ],
    )
    variant_spec_json = base_spec.model_copy(
        update={
            "stateFields": [
                F8StateSpec(name="code", valueSchema=string_schema(default="VARIANT_CODE"), access=F8StateAccess.rw),
                F8StateSpec(name="svcId", valueSchema=string_schema(default="variant_svc"), access=F8StateAccess.ro),
            ]
        }
    ).model_dump(mode="json")
    node = _NodeStub(base_spec)
    graph = F8StudioGraph.__new__(F8StudioGraph)

    graph._apply_variant_to_node(
        node=node,  # type: ignore[arg-type]
        variant_id="v_123",
        variant_name="Variant",
        variant_spec_json=variant_spec_json,
    )

    assert node.model.custom_properties["code"] == "VARIANT_CODE"
    assert node.model.custom_properties["svcId"] == "seed_svc"
