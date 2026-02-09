from __future__ import annotations

import enum
from typing import Any, Callable

from NodeGraphQt.custom_widgets.properties_bin.node_property_factory import NodePropertyWidgetFactory

from f8pysdk import F8DataTypeSchema, F8StateAccess

from .f8_editor_widgets import (
    F8PropBoolSwitch,
    F8PropImageB64,
    F8PropOptionCombo,
    F8PropValueBar,
    parse_select_pool,
)
from .f8_prop_value_widgets import F8DoubleSpinBoxPropWidget, F8IntSpinBoxPropWidget, F8NumberPropLineEdit
from .f8_prop_value_widgets import F8CodeButtonPropWidget


def effective_state_fields(node: Any) -> list[Any]:
    try:
        return list(node.effective_state_fields() or [])
    except Exception:
        return []


def state_field_schema(node: Any, prop_name: str) -> Any | None:
    fields = effective_state_fields(node)
    if not fields:
        try:
            spec = node.spec
        except Exception:
            spec = None
        if spec is not None:
            try:
                fields = list(spec.stateFields or [])
            except Exception:
                fields = []
    for f in fields:
        try:
            if str(f.name or "").strip() == str(prop_name or "").strip():
                return f.valueSchema
        except Exception:
            continue
    return None


def state_field_access(node: Any, prop_name: str) -> F8StateAccess | None:
    fields = effective_state_fields(node)
    if not fields:
        try:
            spec = node.spec
        except Exception:
            spec = None
        if spec is not None:
            try:
                fields = list(spec.stateFields or [])
            except Exception:
                fields = []
    for f in fields:
        try:
            if str(f.name or "").strip() == str(prop_name or "").strip():
                a = f.access
                return a if isinstance(a, F8StateAccess) else None
        except Exception:
            continue
    return None


def state_field_ui_control(node: Any, prop_name: str) -> str:
    fields = effective_state_fields(node)
    if not fields:
        try:
            spec = node.spec
        except Exception:
            spec = None
        if spec is not None:
            try:
                fields = list(spec.stateFields or [])
            except Exception:
                fields = []
    for f in fields:
        try:
            if str(f.name or "").strip() == str(prop_name or "").strip():
                return str(f.uiControl or "").strip().lower()
        except Exception:
            continue
    return ""


def schema_type_any(schema: Any) -> str:
    try:
        try:
            inner = schema.root
        except Exception:
            inner = schema
        try:
            t = inner.type
        except Exception:
            t = None
        if isinstance(t, enum.Enum):
            return str(t.value)
        return str(t or "")
    except Exception:
        return ""


def schema_enum_items(schema: Any) -> list[str]:
    if schema is None:
        return []
    try:
        root = schema.root
    except Exception:
        return []
    try:
        enum_items = list(root.enum or [])
    except Exception:
        return []
    return [str(x) for x in enum_items]


def schema_numeric_range(schema: Any) -> tuple[float | None, float | None]:
    if schema is None:
        return None, None
    mins: list[float] = []
    maxs: list[float] = []
    try:
        root = schema.root
    except Exception:
        root = None

    def _maybe_add_min(v: Any) -> None:
        if v is None:
            return
        try:
            mins.append(float(v))
        except Exception:
            pass

    def _maybe_add_max(v: Any) -> None:
        if v is None:
            return
        try:
            maxs.append(float(v))
        except Exception:
            pass

    try:
        _maybe_add_min(schema.minimum)
    except Exception:
        pass
    try:
        _maybe_add_min(schema.exclusiveMinimum)
    except Exception:
        pass
    try:
        _maybe_add_max(schema.maximum)
    except Exception:
        pass
    try:
        _maybe_add_max(schema.exclusiveMaximum)
    except Exception:
        pass

    if root is not None:
        try:
            _maybe_add_min(root.minimum)
        except Exception:
            pass
        try:
            _maybe_add_min(root.exclusiveMinimum)
        except Exception:
            pass
        try:
            _maybe_add_max(root.maximum)
        except Exception:
            pass
        try:
            _maybe_add_max(root.exclusiveMaximum)
        except Exception:
            pass
    lo = min(mins) if mins else None
    hi = max(maxs) if maxs else None
    return lo, hi


def build_state_value_widget(
    *,
    node: Any,
    prop_name: str,
    widget_type: int,
    widget_factory: NodePropertyWidgetFactory,
    register_option_pool_dependent: Callable[[str, Any], None] | None = None,
) -> Any:
    """
    Build a UI-control-driven widget for a state field (used in PropertiesBin).
    """
    schema = state_field_schema(node, prop_name)
    schema_t = schema_type_any(schema) if schema is not None else ""
    ui_control = state_field_ui_control(node, prop_name)
    enum_items = schema_enum_items(schema) if schema is not None else []
    lo, hi = schema_numeric_range(schema) if schema is not None else (None, None)
    pool_field = parse_select_pool(ui_control)

    is_image_b64 = schema_t == "string" and (
        ui_control in {"image", "image_b64", "img"} or "b64" in str(prop_name).lower()
    )
    if is_image_b64:
        widget = F8PropImageB64()
        widget.set_name(prop_name)
        return widget

    if ui_control in {"code"}:
        try:
            title = f"{node.name()} — {prop_name}"
        except Exception:
            title = f"Edit {prop_name}"
        widget = F8CodeButtonPropWidget(title=title)
        widget.set_name(prop_name)
        return widget

    if enum_items or pool_field or ui_control in {"select", "dropdown", "dropbox", "combo", "combobox"}:
        widget = F8PropOptionCombo()
        widget.set_name(prop_name)
        if pool_field:
            def _pool_resolver(field: str) -> list[str]:
                try:
                    v = node.get_property(str(field))
                except Exception:
                    return []
                if isinstance(v, (list, tuple)):
                    return [str(x) for x in v]
                if isinstance(v, str):
                    # Allow pools stored as JSON strings (eg. "[]", ["a","b"]).
                    try:
                        import json

                        parsed = json.loads(v)
                    except Exception:
                        return []
                    if isinstance(parsed, (list, tuple)):
                        out: list[str] = []
                        for x in parsed:
                            if isinstance(x, str):
                                s = x.strip()
                                if s:
                                    out.append(s)
                                continue
                            if isinstance(x, dict):
                                s = str(x.get("id") or "").strip()
                                if s:
                                    out.append(s)
                                continue
                            s = str(x).strip()
                            if s:
                                out.append(s)
                        return out
                return []

            widget.set_pool(pool_field, _pool_resolver)
            if register_option_pool_dependent is not None:
                register_option_pool_dependent(pool_field, widget)
        else:
            widget.set_items(enum_items)
        return widget

    if schema is not None and (schema_t == "boolean" or ui_control in {"switch", "toggle"}):
        widget = F8PropBoolSwitch()
        widget.set_name(prop_name)
        return widget

    if schema is not None and schema_t in {"integer", "number"} and ui_control == "slider":
        widget = F8PropValueBar(data_type=int if schema_t == "integer" else float)
        widget.set_name(prop_name)
        if lo is not None:
            widget.set_min(lo)
        if hi is not None:
            widget.set_max(hi)
        return widget

    if schema is not None and schema_t == "integer" and ui_control in {"spinbox", "int"}:
        widget = F8IntSpinBoxPropWidget()
        widget.set_name(prop_name)
        if lo is not None:
            widget.set_min(lo)
        if hi is not None:
            widget.set_max(hi)
        return widget

    if schema is not None and schema_t == "number" and ui_control in {"doublespinbox", "float"}:
        widget = F8DoubleSpinBoxPropWidget()
        widget.set_name(prop_name)
        if lo is not None:
            widget.set_min(lo)
        if hi is not None:
            widget.set_max(hi)
        return widget

    if schema is not None and schema_t in {"integer", "number"}:
        widget = F8NumberPropLineEdit(data_type=int if schema_t == "integer" else float)
        widget.set_name(prop_name)
        if lo is not None:
            widget.set_min(lo)
        if hi is not None:
            widget.set_max(hi)
        return widget

    widget = widget_factory.get_widget(widget_type)
    widget.set_name(prop_name)
    return widget
