from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from f8pysdk.specs import F8StateAccess
from NodeGraphQt.custom_widgets.properties_bin.node_property_factory import NodePropertyWidgetFactory

from .state_builders import StateControlSpec, build_panel_control_binding, set_control_read_only
from ...editor_assist.protocol import editor_assist_context_for_field
from ...editor_assist.workspace import EditorAssistContext
from ...nodegraph.node_text_fields import node_text_editor_binding
from ...nodegraph.state_pool_resolver import build_node_pool_resolver, parse_multiselect_pool, parse_select_pool
from ...nodegraph.state_schema import (
    effective_state_fields,
    schema_array_item_type,
    schema_enum_items,
    schema_numeric_range,
    schema_type_any,
    state_field_access,
    state_field_label,
    state_field_schema,
    state_field_ui_control,
)
from ...ui.components.state_editors import F8CodeButtonEditor
from ...ui.support.ui_control import parse_ui_control


@dataclass(frozen=True)
class ControlBuildContext:
    node: Any
    prop_name: str
    widget_type: int
    widget_factory: Any
    register_option_pool_dependent: Callable[[str, Any], None] | None = None


@dataclass(frozen=True)
class ControlBuildResult:
    widget: Any
    readonly: bool = False
    disabled_reason: str = ""


@dataclass(frozen=True)
class StateFieldDescriptor:
    name: str
    access: str
    ui_control: str
    value_schema: Any


def _editor_assist_context_for_field(node: Any, prop_name: str, language: str) -> EditorAssistContext | None:
    ui_control = state_field_ui_control(node, prop_name)
    if parse_ui_control(ui_control).control_name != "code":
        return None
    field_name = str(prop_name or "").strip()
    lang = str(language or "").strip().lower()
    if not field_name or not lang:
        return None
    try:
        spec = node.spec
    except Exception:
        return None
    return editor_assist_context_for_field(spec, field_kind="state", field_key=field_name, language=lang, node=node)


def _node_graph_and_id(node: Any) -> tuple[Any | None, str]:
    try:
        graph = node.graph
        node_id = str(node.id or "").strip()
    except AttributeError:
        return None, ""
    return graph, node_id


def build_state_panel_control(
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
    return _build_state_panel_control(context)


def _build_state_panel_control(context: ControlBuildContext) -> Any:
    node = context.node
    prop_name = str(context.prop_name or "")
    schema = state_field_schema(node, prop_name)
    schema_t = schema_type_any(schema) if schema is not None else ""
    ui_control = state_field_ui_control(node, prop_name)
    parsed_ui = parse_ui_control(ui_control)
    ui_control_name = parsed_ui.control_name
    ui_language = parsed_ui.ui_language
    field_label = state_field_label(node, prop_name) or prop_name
    enum_items = schema_enum_items(schema) if schema is not None else []
    minimum, maximum = schema_numeric_range(schema) if schema is not None else (None, None)

    select_pool_field = parse_select_pool(ui_control)
    multiselect_pool_field = parse_multiselect_pool(ui_control)
    pool_resolver = build_node_pool_resolver(node)
    spec = StateControlSpec(
        name=prop_name,
        label=field_label,
        ui_control=ui_control,
        ui_language=ui_language or "plaintext",
        schema_type=schema_t,
        enum_items=enum_items,
        minimum=minimum,
        maximum=maximum,
        select_pool_field=select_pool_field,
        multiselect_pool_field=multiselect_pool_field,
        is_image_b64=schema_t == "string"
        and (ui_control_name in {"image", "image_b64", "img"} or "b64" in prop_name.lower()),
        range_integer=schema_array_item_type(schema) == "integer",
    )
    try:
        title = f"{node.name()} - {prop_name}"
    except AttributeError:
        title = f"Edit {prop_name}"
    binding = build_panel_control_binding(
        spec=spec,
        fallback_widget=context.widget_factory.get_widget(context.widget_type),
        pool_resolver=pool_resolver,
        editor_title=title,
        assist_context=_editor_assist_context_for_field(node, prop_name, ui_language),
        assist_context_provider=lambda current_node=node,
        current_prop=prop_name,
        current_lang=ui_language: _editor_assist_context_for_field(
            current_node,
            current_prop,
            current_lang,
        ),
        editor_session_key=None,
    )
    if isinstance(binding.widget, F8CodeButtonEditor):
        graph, node_id = _node_graph_and_id(node)
        text_binding = node_text_editor_binding(graph, node_id, prop_name)
        if text_binding is not None:
            binding.widget.set_persisted_value_getter(text_binding.value_getter)
            binding.widget.set_persisted_value_setter(text_binding.value_setter)
            binding.widget.set_persisted_target_exists_provider(text_binding.target_exists)
            binding.widget.set_editor_session_key(text_binding.session_key)
    if binding.refresh_options is not None and context.register_option_pool_dependent is not None:
        if multiselect_pool_field:
            context.register_option_pool_dependent(multiselect_pool_field, binding.widget)
        if select_pool_field:
            context.register_option_pool_dependent(select_pool_field, binding.widget)
    return binding.widget


def set_widget_read_only(widget: Any, *, read_only: bool) -> None:
    set_control_read_only(widget, read_only=read_only)


def state_field_is_readonly(access: F8StateAccess | None) -> bool:
    if access is None:
        return False
    return bool(access == F8StateAccess.ro)


__all__ = [
    "ControlBuildContext",
    "ControlBuildResult",
    "StateFieldDescriptor",
    "F8StateAccess",
    "build_state_panel_control",
    "effective_state_fields",
    "parse_multiselect_pool",
    "parse_select_pool",
    "schema_enum_items",
    "schema_array_item_type",
    "schema_numeric_range",
    "schema_type_any",
    "set_widget_read_only",
    "state_field_access",
    "state_field_is_readonly",
    "state_field_label",
    "state_field_schema",
    "state_field_ui_control",
]
