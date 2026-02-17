from __future__ import annotations

from typing import Any, Callable

from NodeGraphQt.custom_widgets.properties_bin.node_property_factory import NodePropertyWidgetFactory

from f8pysdk import F8StateAccess

from .state_controls.descriptors import ControlBuildContext
from .state_controls.factory import build_state_value_widget as _build_state_value_widget
from .state_controls.schema_introspect import (
    effective_state_fields,
    schema_enum_items,
    schema_numeric_range,
    schema_type_any,
    state_field_access,
    state_field_schema,
    state_field_ui_control,
    state_field_ui_language,
)


def build_state_value_widget(
    *,
    node: Any,
    prop_name: str,
    widget_type: int,
    widget_factory: NodePropertyWidgetFactory,
    register_option_pool_dependent: Callable[[str, Any], None] | None = None,
) -> Any:
    context = ControlBuildContext(
        node=node,
        prop_name=str(prop_name or ""),
        widget_type=int(widget_type),
        widget_factory=widget_factory,
        register_option_pool_dependent=register_option_pool_dependent,
    )
    return _build_state_value_widget(context)


__all__ = [
    "build_state_value_widget",
    "effective_state_fields",
    "schema_enum_items",
    "schema_numeric_range",
    "schema_type_any",
    "state_field_access",
    "state_field_schema",
    "state_field_ui_control",
    "state_field_ui_language",
    "F8StateAccess",
]
