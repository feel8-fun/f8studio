from __future__ import annotations

from typing import Any

from f8pysdk.generated import F8OperatorSchemaVersion, F8OperatorSpec, F8StateAccess, F8StateSpec
from f8pysdk.schema_helpers import integer_schema, string_schema

from f8pystudio.variants.variant_compose import build_variant_record_from_node


class _NodeModelStub:
    def __init__(self, *, properties: dict[str, Any], custom_properties: dict[str, Any]) -> None:
        self.properties = dict(properties)
        self.custom_properties = dict(custom_properties)

    def get_property(self, name: str) -> Any:
        if name in self.properties:
            return self.properties[name]
        if name in self.custom_properties:
            return self.custom_properties[name]
        raise KeyError(name)


class _NodeStub:
    def __init__(
        self,
        *,
        spec: F8OperatorSpec,
        node_type: str,
        properties: dict[str, Any],
        custom_properties: dict[str, Any],
    ) -> None:
        self.spec = spec
        self.type_ = node_type
        self.model = _NodeModelStub(properties=properties, custom_properties=custom_properties)

    @staticmethod
    def ui_overrides() -> dict[str, Any]:
        return {}


def test_build_variant_record_reads_state_values_from_properties_and_custom_properties() -> None:
    spec = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass="f8.pyengine",
        operatorClass="f8.python_script",
        version="0.0.1",
        label="Python Script",
        stateFields=[
            F8StateSpec(
                name="code",
                valueSchema=string_schema(default=""),
                access=F8StateAccess.rw,
            ),
            F8StateSpec(
                name="gain",
                valueSchema=integer_schema(default=0),
                access=F8StateAccess.rw,
            ),
        ],
    )
    node = _NodeStub(
        spec=spec,
        node_type="f8.pyengine.f8.python_script",
        properties={"code": "print('hello')"},
        custom_properties={"gain": 7},
    )

    record = build_variant_record_from_node(node=node, name="My Script", description="", tags=[])
    state_fields = list(record.spec.get("stateFields") or [])
    by_name = {str(item.get("name")): item for item in state_fields if isinstance(item, dict)}

    code_schema = by_name["code"]["valueSchema"]
    gain_schema = by_name["gain"]["valueSchema"]
    assert code_schema["default"] == "print('hello')"
    assert gain_schema["default"] == 7
