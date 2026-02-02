from __future__ import annotations

from typing import Any, Callable

from NodeGraphQt.custom_widgets.properties_bin.node_property_factory import NodePropertyWidgetFactory

from f8pysdk import F8DataTypeSchema, F8StateAccess

from .f8_editor_widgets import (
    F8PropBoolToggle,
    F8PropImageB64,
    F8PropOptionToggle,
    F8PropValueBar,
    parse_select_pool,
)
from .f8_prop_value_widgets import F8DoubleSpinBoxPropWidget, F8IntSpinBoxPropWidget, F8NumberPropLineEdit


def effective_state_fields(node: Any) -> list[Any]:
    fn = getattr(node, "effective_state_fields", None)
    return list(fn() or []) if callable(fn) else []


def state_field_schema(node: Any, prop_name: str) -> Any | None:
    fields = effective_state_fields(node)
    if not fields:
        spec = getattr(node, "spec", None)
        fields = list(getattr(spec, "stateFields", None) or [])
    for f in fields:
        if str(getattr(f, "name", "") or "").strip() == str(prop_name or "").strip():
            return getattr(f, "valueSchema", None)
    return None


def state_field_access(node: Any, prop_name: str) -> F8StateAccess | None:
    fields = effective_state_fields(node)
    if not fields:
        spec = getattr(node, "spec", None)
        fields = list(getattr(spec, "stateFields", None) or [])
    for f in fields:
        if str(getattr(f, "name", "") or "").strip() == str(prop_name or "").strip():
            a = getattr(f, "access", None)
            return a if isinstance(a, F8StateAccess) else None
    return None


def state_field_ui_control(node: Any, prop_name: str) -> str:
    fields = effective_state_fields(node)
    if not fields:
        spec = getattr(node, "spec", None)
        fields = list(getattr(spec, "stateFields", None) or [])
    for f in fields:
        if str(getattr(f, "name", "") or "").strip() == str(prop_name or "").strip():
            return str(getattr(f, "uiControl", "") or "").strip().lower()
    return ""


def schema_type_any(schema: Any) -> str:
    try:
        inner = getattr(schema, "root", schema)
        t = getattr(inner, "type", None)
        if hasattr(t, "value"):
            return str(t.value)
        return str(t)
    except Exception:
        return ""


def schema_enum_items(schema: Any) -> list[str]:
    if schema is None:
        return []
    root = getattr(schema, "root", None)
    enum = getattr(root, "enum", None)
    if not enum:
        return []
    return [str(x) for x in list(enum)]


def schema_numeric_range(schema: Any) -> tuple[float | None, float | None]:
    if schema is None:
        return None, None
    mins: list[float] = []
    maxs: list[float] = []

    def _pick(attr: str) -> Any:
        v = getattr(schema, attr, None)
        if v is not None:
            return v
        root = getattr(schema, "root", None)
        return getattr(root, attr, None) if root is not None else None

    for k in ("minimum", "exclusiveMinimum"):
        try:
            v = _pick(k)
            if v is not None:
                mins.append(float(v))
        except Exception:
            pass
    for k in ("maximum", "exclusiveMaximum"):
        try:
            v = _pick(k)
            if v is not None:
                maxs.append(float(v))
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

    if enum_items or pool_field or ui_control in {"select", "dropdown", "dropbox", "combo", "combobox"}:
        widget = F8PropOptionToggle()
        widget.set_name(prop_name)
        if pool_field:
            def _pool_resolver(field: str) -> list[str]:
                try:
                    v = node.get_property(str(field))
                except Exception:
                    return []
                if isinstance(v, (list, tuple)):
                    return [str(x) for x in v]
                return []

            widget.set_pool(pool_field, _pool_resolver)
            if register_option_pool_dependent is not None:
                register_option_pool_dependent(pool_field, widget)
        else:
            widget.set_items(enum_items)
        return widget

    if schema is not None and (schema_t == "boolean" or ui_control in {"switch", "toggle"}):
        widget = F8PropBoolToggle()
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
