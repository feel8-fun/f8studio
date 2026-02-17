from __future__ import annotations

from typing import Any

from f8pysdk import F8StateAccess

from ..f8_editor_widgets import (
    F8PropBoolSwitch,
    F8PropImageB64,
    F8PropMultiSelect,
    F8PropOptionCombo,
    F8PropValueBar,
)
from ..f8_prop_value_widgets import F8CodeButtonPropWidget, F8InlineCodePropWidget, F8NumberPropLineEdit, F8WrapLinePropWidget
from .descriptors import ControlBuildContext
from .pool_resolver import build_node_pool_resolver, parse_multiselect_pool, parse_select_pool
from .schema_introspect import (
    schema_enum_items,
    schema_numeric_range,
    schema_type_any,
    state_field_schema,
    state_field_ui_control,
    state_field_ui_language,
)


def build_state_value_widget(context: ControlBuildContext) -> Any:
    node = context.node
    prop_name = str(context.prop_name or "")
    schema = state_field_schema(node, prop_name)
    schema_t = schema_type_any(schema) if schema is not None else ""
    ui_control = state_field_ui_control(node, prop_name)
    ui_control_l = str(ui_control).strip().lower()
    ui_language = state_field_ui_language(node, prop_name)
    enum_items = schema_enum_items(schema) if schema is not None else []
    lo, hi = schema_numeric_range(schema) if schema is not None else (None, None)

    pool_field = parse_select_pool(ui_control)
    multi_pool_field = parse_multiselect_pool(ui_control)
    pool_resolver = build_node_pool_resolver(node)

    is_image_b64 = schema_t == "string" and (
        ui_control_l in {"image", "image_b64", "img"} or "b64" in str(prop_name).lower()
    )
    if is_image_b64:
        widget = F8PropImageB64()
        widget.set_name(prop_name)
        return widget

    if ui_control_l in {"code"}:
        try:
            title = f"{node.name()} - {prop_name}"
        except AttributeError:
            title = f"Edit {prop_name}"
        widget = F8CodeButtonPropWidget(title=title, language=ui_language or "plaintext")
        widget.set_name(prop_name)
        return widget

    if ui_control_l in {"wrapline"}:
        widget = F8WrapLinePropWidget(language=ui_language or "plaintext")
        widget.set_name(prop_name)
        return widget

    if ui_control_l in {"code_inline", "multiline"}:
        widget = F8InlineCodePropWidget(language=ui_language or "plaintext")
        widget.set_name(prop_name)
        return widget

    if multi_pool_field or ui_control_l in {"multiselect", "multi_select", "multi-select"}:
        widget = F8PropMultiSelect()
        widget.set_name(prop_name)
        if multi_pool_field:
            widget.set_pool(multi_pool_field, pool_resolver)
            if context.register_option_pool_dependent is not None:
                context.register_option_pool_dependent(multi_pool_field, widget)
        else:
            widget.set_items(enum_items)
        return widget

    if enum_items or pool_field or ui_control_l in {"select", "dropdown", "dropbox", "combo", "combobox"}:
        widget = F8PropOptionCombo()
        widget.set_name(prop_name)
        if pool_field:
            widget.set_pool(pool_field, pool_resolver)
            if context.register_option_pool_dependent is not None:
                context.register_option_pool_dependent(pool_field, widget)
        else:
            widget.set_items(enum_items)
        return widget

    if schema is not None and (schema_t == "boolean" or ui_control_l in {"switch", "toggle"}):
        widget = F8PropBoolSwitch()
        widget.set_name(prop_name)
        return widget

    if schema is not None and schema_t in {"integer", "number"} and ui_control_l == "slider":
        widget = F8PropValueBar(data_type=int if schema_t == "integer" else float)
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

    widget = context.widget_factory.get_widget(context.widget_type)
    widget.set_name(prop_name)
    return widget


def state_field_is_readonly(access: F8StateAccess | None) -> bool:
    if access is None:
        return False
    return bool(access == F8StateAccess.ro)
