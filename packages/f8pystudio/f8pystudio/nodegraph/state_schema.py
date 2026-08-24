from __future__ import annotations

import enum
from typing import Any

from f8pysdk.specs import F8StateAccess
from f8pysdk.specs import (
    F8AnyTypeSchema,
    F8ArrayTypeSchema,
    F8BooleanTypeSchema,
    F8ComplexObjectTypeSchema,
    F8IntegerTypeSchema,
    F8NullTypeSchema,
    F8NumberTypeSchema,
    F8StringTypeSchema,
)
from .ui_state_mutations import state_field_global_hotkey as _state_field_global_hotkey


def effective_state_fields(node: Any) -> list[Any]:
    try:
        fields = node.effective_state_fields()
    except AttributeError:
        fields = []
    return list(fields or [])


def state_field_schema(node: Any, prop_name: str) -> Any | None:
    prop = str(prop_name or "").strip()
    if not prop:
        return None
    for field in effective_state_fields(node):
        try:
            name = str(field.name or "").strip()
        except AttributeError:
            continue
        if name == prop:
            try:
                return field.valueSchema
            except AttributeError:
                return None
    return None


def state_field_access(node: Any, prop_name: str) -> F8StateAccess | None:
    prop = str(prop_name or "").strip()
    if not prop:
        return None
    for field in effective_state_fields(node):
        try:
            name = str(field.name or "").strip()
        except AttributeError:
            continue
        if name != prop:
            continue
        try:
            access = field.access
        except AttributeError:
            return None
        if isinstance(access, F8StateAccess):
            return access
        return None
    return None


def state_field_ui_control(node: Any, prop_name: str) -> str:
    prop = str(prop_name or "").strip()
    if not prop:
        return ""
    for field in effective_state_fields(node):
        try:
            name = str(field.name or "").strip()
        except AttributeError:
            continue
        if name != prop:
            continue
        try:
            return str(field.uiControl or "").strip()
        except AttributeError:
            return ""
    return ""


def state_field_label(node: Any, prop_name: str) -> str:
    prop = str(prop_name or "").strip()
    if not prop:
        return ""
    for field in effective_state_fields(node):
        try:
            name = str(field.name or "").strip()
        except AttributeError:
            continue
        if name != prop:
            continue
        try:
            return str(field.label or "").strip()
        except AttributeError:
            return ""
    return ""


def state_field_global_hotkey(node: Any, prop_name: str) -> str:
    try:
        return _state_field_global_hotkey(node, prop_name)
    except Exception:
        return ""


def schema_type_any(schema: Any) -> str:
    if schema is None:
        return ""
    if isinstance(schema, dict):
        raw = schema.get("type")
        if isinstance(raw, enum.Enum):
            return str(raw.value)
        return str(raw or "")
    inner = schema
    if isinstance(inner, F8StringTypeSchema):
        return "string"
    if isinstance(inner, F8NumberTypeSchema):
        return "number"
    if isinstance(inner, F8IntegerTypeSchema):
        return "integer"
    if isinstance(inner, F8BooleanTypeSchema):
        return "boolean"
    if isinstance(inner, F8NullTypeSchema):
        return "null"
    if isinstance(inner, F8ComplexObjectTypeSchema):
        return "object"
    if isinstance(inner, F8ArrayTypeSchema):
        return "array"
    if isinstance(inner, F8AnyTypeSchema):
        return "any"
    return ""


def schema_enum_items(schema: Any) -> list[str]:
    if schema is None:
        return []
    if isinstance(schema, dict):
        values_raw = schema.get("enum")
        if isinstance(values_raw, list):
            return [str(item) for item in values_raw]
        if schema.get("type") == "array":
            items_raw = schema.get("items")
            if isinstance(items_raw, dict):
                item_values_raw = items_raw.get("enum")
                if isinstance(item_values_raw, list):
                    return [str(item) for item in item_values_raw]
        return []
    if isinstance(schema, F8ArrayTypeSchema):
        return schema_enum_items(schema.items)
    if isinstance(
        schema,
        (
            F8StringTypeSchema,
            F8NumberTypeSchema,
            F8IntegerTypeSchema,
            F8BooleanTypeSchema,
            F8NullTypeSchema,
        ),
    ):
        values_raw = schema.enum
        if isinstance(values_raw, list):
            return [str(item) for item in values_raw]
        return []
    return []


def schema_numeric_range(schema: Any) -> tuple[float | None, float | None]:
    if schema is None:
        return None, None
    if isinstance(schema, dict) and schema.get("type") == "array":
        return schema_numeric_range(schema.get("items"))
    if isinstance(schema, F8ArrayTypeSchema):
        return schema_numeric_range(schema.items)
    mins: list[float] = []
    maxs: list[float] = []

    def _append_min(raw: Any) -> None:
        if raw is None:
            return
        try:
            mins.append(float(raw))
        except (TypeError, ValueError):
            return

    def _append_max(raw: Any) -> None:
        if raw is None:
            return
        try:
            maxs.append(float(raw))
        except (TypeError, ValueError):
            return

    try:
        _append_min(schema.minimum)
    except AttributeError:
        _append_min(None)
    try:
        _append_min(schema.exclusiveMinimum)
    except AttributeError:
        _append_min(None)
    try:
        _append_max(schema.maximum)
    except AttributeError:
        _append_max(None)
    try:
        _append_max(schema.exclusiveMaximum)
    except AttributeError:
        _append_max(None)

    lo = min(mins) if mins else None
    hi = max(maxs) if maxs else None
    return lo, hi


def schema_array_item_type(schema: Any) -> str:
    if isinstance(schema, dict) and schema.get("type") == "array":
        return schema_type_any(schema.get("items"))
    if isinstance(schema, F8ArrayTypeSchema):
        return schema_type_any(schema.items)
    return ""
