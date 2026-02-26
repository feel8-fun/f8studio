from __future__ import annotations

import enum
from typing import Any

from f8pysdk import F8StateAccess


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


def state_field_ui_language(node: Any, prop_name: str) -> str:
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
            return str(field.uiLanguage or "").strip().lower()
        except AttributeError:
            return ""
    return ""


def schema_type_any(schema: Any) -> str:
    if schema is None:
        return ""
    try:
        inner = schema.root
    except AttributeError:
        inner = schema
    try:
        raw_type = inner.type
    except AttributeError:
        raw_type = None
    if isinstance(raw_type, enum.Enum):
        return str(raw_type.value)
    return str(raw_type or "")


def schema_enum_items(schema: Any) -> list[str]:
    if schema is None:
        return []
    try:
        root = schema.root
    except AttributeError:
        return []
    try:
        values = list(root.enum or [])
    except AttributeError:
        return []
    return [str(item) for item in values]


def schema_numeric_range(schema: Any) -> tuple[float | None, float | None]:
    if schema is None:
        return None, None
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

    try:
        root = schema.root
    except AttributeError:
        root = None
    if root is not None:
        try:
            _append_min(root.minimum)
        except AttributeError:
            _append_min(None)
        try:
            _append_min(root.exclusiveMinimum)
        except AttributeError:
            _append_min(None)
        try:
            _append_max(root.maximum)
        except AttributeError:
            _append_max(None)
        try:
            _append_max(root.exclusiveMaximum)
        except AttributeError:
            _append_max(None)

    lo = min(mins) if mins else None
    hi = max(maxs) if maxs else None
    return lo, hi
