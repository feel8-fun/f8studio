from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import msgspec

from f8studio_server.schema_generation import model_schemas


class Child(msgspec.Struct, rename="camel", tag="child", tag_field="kind"):
    value: int = 4
    maybe_value: str | None = None
    absent: Annotated[str | msgspec.UnsetType, msgspec.Meta(description="May be omitted")] = msgspec.UNSET
    explicit_unset: str | msgspec.UnsetType = "default"


class Parent(msgspec.Struct):
    children: tuple[Child, ...] = ()
    position: tuple[float, float] = (0, 0)


class Compact(msgspec.Struct, omit_defaults=True):
    value: int = 0


@dataclass
class Wrapper:
    value: Parent


def test_schemas_distinguish_accepted_input_from_serialized_output() -> None:
    inputs, outputs, schemas = model_schemas((Wrapper, Compact))
    assert inputs[0]["$ref"] == "#/$defs/WrapperInput"
    assert outputs[0]["$ref"] == "#/$defs/Wrapper"
    assert schemas["WrapperInput"]["properties"]["value"]["$ref"] == "#/$defs/ParentInput"
    assert schemas["ParentInput"]["properties"]["children"]["items"]["$ref"] == "#/$defs/ChildInput"
    assert schemas["ChildInput"]["required"] == ["kind"]
    assert set(schemas["Child"]["required"]) == {"kind", "value", "maybeValue"}
    assert set(schemas["Parent"]["required"]) == {"children", "position"}
    assert schemas["Compact"].get("required", []) == []
    wire = msgspec.json.decode(msgspec.json.encode(Child()))
    assert set(schemas["Child"]["required"]) <= wire.keys()
    assert wire["maybeValue"] is None
    assert "absent" not in wire
    assert "explicitUnset" in wire
    assert "explicitUnset" not in msgspec.json.decode(msgspec.json.encode(Child(explicit_unset=msgspec.UNSET)))


class Tree(msgspec.Struct):
    children: tuple["Tree", ...] = ()


def test_input_and_output_recursive_references_keep_their_own_direction() -> None:
    _, _, schemas = model_schemas((Tree,))
    assert schemas["TreeInput"]["properties"]["children"]["items"]["$ref"] == "#/$defs/TreeInput"
    assert schemas["Tree"]["properties"]["children"]["items"]["$ref"] == "#/$defs/Tree"


class Other(msgspec.Struct, tag="other", tag_field="kind"):
    label: str = ""


def test_input_discriminator_mapping_uses_input_definitions() -> None:
    inputs, outputs, _ = model_schemas((Child | Other,))
    assert inputs[0]["discriminator"]["mapping"] == {
        "child": "#/$defs/ChildInput",
        "other": "#/$defs/OtherInput",
    }
    assert outputs[0]["discriminator"]["mapping"]["child"] == "#/$defs/Child"


class SchemaKeywordsAsData(msgspec.Struct, kw_only=True):
    reference: str = msgspec.field(name="$ref", default="unchanged")
    payload: Annotated[dict[str, str], msgspec.Meta(extra_json_schema={"default": {"$ref": "user-data"}})]


def test_schema_keywords_inside_field_names_and_defaults_remain_data() -> None:
    _, _, schemas = model_schemas((SchemaKeywordsAsData,))
    properties = schemas["SchemaKeywordsAsDataInput"]["properties"]
    assert properties["$ref"]["default"] == "unchanged"
    assert properties["payload"]["default"] == {"$ref": "user-data"}
