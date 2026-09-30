"""Schema generation using msgspec serialization metadata for input/output types."""

from __future__ import annotations

import dataclasses
import types
from typing import Annotated, Any, Union, cast, get_args, get_origin, get_type_hints
import msgspec


def _allows_unset(value: Any) -> bool:
    if value is msgspec.UnsetType:
        return True
    origin = get_origin(value)
    if origin is Annotated:
        return _allows_unset(get_args(value)[0])
    if origin in (Union, types.UnionType):
        return any(_allows_unset(item) for item in get_args(value))
    return False


def output_required_fields(roots: tuple[Any, ...]) -> dict[str, tuple[str, ...]]:
    """Fields emitted by msgspec, including defaults but excluding UNSET.

    This is generic serializer metadata traversal, not application dispatch.
    Input schemas retain msgspec's decoding/default semantics independently.
    """
    result: dict[str, tuple[str, ...]] = {}
    seen: set[Any] = set()

    def visit(value: Any) -> None:
        origin = get_origin(value)
        if origin is not None:
            for argument in get_args(value):
                if isinstance(argument, (type, types.UnionType)) or get_origin(argument) is not None:
                    visit(argument)
            return
        if not isinstance(value, type) or value in seen:
            return
        seen.add(value)
        if issubclass(value, msgspec.Struct):
            if value.__name__ in result:
                raise ValueError(f"Ambiguous schema model name: {value.__name__}")
            config = value.__struct_config__
            required: list[str] = [config.tag_field] if config.tag is not None and config.tag_field is not None else []
            for field in msgspec.structs.fields(value):
                visit(field.type)
                if not _allows_unset(field.type) and not config.omit_defaults:
                    required.append(field.encode_name)
            result[value.__name__] = tuple(required)
        elif dataclasses.is_dataclass(cast(Any, value)):
            fields = dataclasses.fields(cast(Any, value))
            result[value.__name__] = tuple(field.name for field in fields)
            for field_type in get_type_hints(cast(Any, value), include_extras=True).values():
                visit(field_type)

    for root in roots:
        visit(root)
    return result


def model_schemas(
    roots: tuple[Any, ...], *, ref_template: str = "#/$defs/{name}"
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    outputs, raw_components = msgspec.json.schema_components(roots, ref_template=ref_template)
    emitted = output_required_fields(roots)
    components: dict[str, Any] = {}

    def as_input(value: Any) -> Any:
        if isinstance(value, dict):
            incoming: dict[str, Any] = {}
            for key, item in cast(dict[str, Any], value).items():
                if key in ("default", "examples", "enum", "const"):
                    incoming[key] = item
                elif key in ("properties", "patternProperties", "$defs"):
                    incoming[key] = {name: as_input(child) for name, child in item.items()}
                elif key == "$ref":
                    incoming[key] = item + "Input"
                else:
                    incoming[key] = as_input(item)
            discriminator = cast(dict[str, Any], value).get("discriminator")
            if isinstance(discriminator, dict) and "mapping" in discriminator:
                typed_discriminator = cast(dict[str, Any], discriminator)
                incoming["discriminator"] = {
                    **typed_discriminator,
                    "mapping": {tag: ref + "Input" for tag, ref in typed_discriminator["mapping"].items()},
                }
            return incoming
        if isinstance(value, list):
            return [as_input(item) for item in cast(list[Any], value)]
        return value

    for name, schema in raw_components.items():
        if name + "Input" in raw_components:
            raise ValueError(f"Generated input name conflicts with model: {name}Input")
        components[name + "Input"] = as_input(schema)
        if name in emitted and "properties" in schema:
            schema = {**schema, "required": sorted(set(schema.get("required", ())) | set(emitted[name]))}
        components[name] = schema
    inputs = tuple(cast(dict[str, Any], as_input(root)) for root in outputs)
    return inputs, outputs, components
