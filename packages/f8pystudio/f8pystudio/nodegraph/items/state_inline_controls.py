from __future__ import annotations

"""
Inline node-hosted controls for state values.

This module binds state-value controls into the nodegraph item environment,
handling node callbacks, option-pool refresh, styling, and graph sync.
"""

import enum
import json
import logging
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

from f8pysdk.specs import schema_type

from ...ui.components.controls import F8Dial, F8MultiSelect, F8OptionCombo
from ...ui.components.state_editors import (
    F8BoolSwitchEditor,
    F8CodeButtonEditor,
    F8DialEditor,
    F8IncrementButtonEditor,
    F8MultiSelectEditor,
    F8OptionComboEditor,
    F8ValueBarEditor,
)
from ...ui.components.controls import parse_multiselect_pool, parse_select_pool
from ...ui.support.ui_control import parse_ui_control
from ...ui.support.state_builders import (
    StateControlSpec,
    build_inline_control_binding,
    set_control_read_only,
)
from ...ui.support.studio_theme import (
    inline_control_qss,
    inline_header_button_qss,
    studio_dark_theme,
    transparent_header_qss,
    transparent_widget_qss,
)
from ...ui.components.wave import (
    WAVE_PATTERN_EDITOR_DEPENDENCY_FIELDS,
    WAVE_PREVIEW_DEPENDENCY_FIELDS,
)
from ...editor_assist.protocol import editor_assist_context_for_field
from ...editor_assist.workspace import EditorAssistContext
from ...nodegraph.state_pool_resolver import resolve_pool_items
from ...nodegraph.state_schema import schema_array_item_type
from ...nodegraph.node_text_fields import node_text_editor_binding, resolve_node
from ...nodegraph.ui_state_mutations import set_state_inline_expanded, state_inline_expanded
from .node_item_core import StateFieldInfo, state_field_info
from .proxy_widget_utils import dispose_detached_proxy_widget
from .service_toolbar_host import F8ElideToolButton, F8ForceGlobalToolTipFilter

logger = logging.getLogger(__name__)

INLINE_HEADER_BUTTON_STYLE = inline_header_button_qss()
_NODE_ACCESS_ERRORS = (AttributeError, RuntimeError, TypeError)
_NODE_VALUE_ACCESS_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_QT_PROXY_ACCESS_ERRORS = (AttributeError, RuntimeError, TypeError)
_STATE_SPEC_ERRORS = (AttributeError, TypeError, ValueError)
_INLINE_STATE_UPDATER_ERRORS = (Exception,)


def _node_item_id(node_item: Any) -> str:
    try:
        return str(node_item.id or "").strip()
    except _NODE_VALUE_ACCESS_ERRORS:
        return ""


def _backend_node_id(node: Any) -> str:
    try:
        return str(node.id or "").strip()
    except _NODE_VALUE_ACCESS_ERRORS:
        return ""


def _node_spec(node: Any) -> Any | None:
    try:
        return node.spec
    except _NODE_ACCESS_ERRORS:
        return None


def _json_safe_schema_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_json_safe_schema_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_schema_value(item) for key, item in value.items()}
    return str(value)


def state_inline_control_serial(node_item: Any, info: StateFieldInfo) -> str:
    """
    Signature used to decide whether an inline state control must be rebuilt.

    Include schema details that affect the control widget itself, not just
    cosmetic metadata like label/description.
    """
    try:
        value_schema = info.value_schema
        enum_items = node_item._schema_enum_items(value_schema)
        minimum, maximum = node_item._schema_numeric_range(value_schema)
        default_value = None
        if value_schema is not None:
            try:
                default_value = value_schema.default
            except AttributeError:
                default_value = None
        return json.dumps(
            {
                "access": info.access_str,
                "required": info.required,
                "uiControl": info.ui_control,
                "schemaType": str(schema_type(value_schema) or ""),
                "enum": [str(item) for item in enum_items],
                "minimum": minimum,
                "maximum": maximum,
                "default": _json_safe_schema_value(default_value),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except (AttributeError, TypeError, ValueError):
        return ""


def _refresh_embedded_text_palette(widget: QtWidgets.QWidget) -> None:
    palette = widget.palette()
    theme_palette = studio_dark_theme().palette
    text_color = QtGui.QColor(theme_palette.text_primary)
    placeholder_color = QtGui.QColor(theme_palette.text_muted)
    placeholder_color.setAlpha(150)

    for group in (
        QtGui.QPalette.ColorGroup.Active,
        QtGui.QPalette.ColorGroup.Inactive,
        QtGui.QPalette.ColorGroup.Disabled,
    ):
        palette.setColor(group, QtGui.QPalette.ColorRole.Text, text_color)
        palette.setColor(group, QtGui.QPalette.ColorRole.WindowText, text_color)
        palette.setColor(group, QtGui.QPalette.ColorRole.ButtonText, text_color)
        palette.setColor(group, QtGui.QPalette.ColorRole.BrightText, text_color)
        try:
            palette.setColor(group, QtGui.QPalette.ColorRole.PlaceholderText, placeholder_color)
        except AttributeError:
            pass

    try:
        palette.setBrush(QtGui.QPalette.ColorRole.PlaceholderText, placeholder_color)
    except (AttributeError, TypeError):
        pass

    widget.setPalette(palette)
    if isinstance(widget, QtWidgets.QAbstractScrollArea):
        viewport = widget.viewport()
        if viewport is not None:
            viewport.setPalette(palette)
    try:
        widget.update()
    except (AttributeError, RuntimeError, TypeError):
        pass


def _apply_text_palette(widget: QtWidgets.QWidget) -> None:
    _refresh_embedded_text_palette(widget)


def build_inline_header_button(
    *,
    label: str,
    tooltip: str,
    expandable: bool,
    expanded: bool = False,
    parent: QtWidgets.QWidget | None = None,
) -> tuple[QtWidgets.QWidget, F8ElideToolButton]:
    header = QtWidgets.QWidget(parent)
    header_lay = QtWidgets.QHBoxLayout(header)
    header_lay.setContentsMargins(0, 0, 0, 0)
    header_lay.setSpacing(6)
    header.setAttribute(QtCore.Qt.WA_StyledBackground, True)
    header.setStyleSheet(transparent_header_qss())

    btn = F8ElideToolButton(header)
    btn.setCheckable(bool(expandable))
    btn.setChecked(bool(expanded) if expandable else False)
    btn.setAutoRaise(True)
    btn.setProperty("_f8_preview_interaction_exempt", bool(expandable))
    if expandable:
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
    else:
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        btn.setArrowType(QtCore.Qt.NoArrow)
    btn.set_full_text(label)
    btn.setToolTip(tooltip)
    btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    btn.setStyleSheet(INLINE_HEADER_BUTTON_STYLE)

    header_lay.addWidget(btn, 1)
    return header, btn


def _refresh_state_inline_control_metadata(node_item: Any, control: QtWidgets.QWidget, info: StateFieldInfo) -> None:
    name = info.name
    label = info.label or name
    field_tooltip = info.tooltip if info.tooltip != name else ""

    if isinstance(control, F8IncrementButtonEditor):
        control.set_button_text(label)
        control.set_context_tooltip(field_tooltip)
        return

    if isinstance(control, F8CodeButtonEditor):
        control.set_title(f"{node_item.name} - {label}")
        control.setToolTip(field_tooltip)
        return

    if isinstance(
        control,
        (
            F8OptionComboEditor,
            F8MultiSelectEditor,
            F8BoolSwitchEditor,
            F8ValueBarEditor,
            F8DialEditor,
        ),
    ):
        control.set_context_tooltip(field_tooltip)
        return

    if isinstance(control, (F8OptionCombo, F8MultiSelect, F8Dial)):
        control.set_context_tooltip(field_tooltip)
        return

    control.setToolTip(field_tooltip)


def _editor_assist_context(
    graph: Any,
    *,
    node_id: str,
    state_field_name: str,
    ui_control: str,
    language: str,
) -> EditorAssistContext | None:
    if parse_ui_control(ui_control).control_name != "code":
        return None
    field_name = str(state_field_name or "").strip()
    lang = str(language or "").strip().lower()
    if not field_name or not lang:
        return None

    node = resolve_node(graph, node_id)
    if node is None:
        return None
    spec = None
    spec = _node_spec(node)
    if spec is None:
        return None

    return editor_assist_context_for_field(spec, field_kind="state", field_key=field_name, language=lang, node=node)


def is_state_inline_input_connected(node_item: Any, field_name: str) -> bool:
    """
    True if the state field is upstream-driven via a state-edge.
    """
    name = str(field_name or "").strip()
    if not name:
        return False
    node = node_item._backend_node()
    if node is None:
        return False
    port = node.get_input(f"[S]{name}")
    if port is None:
        return False
    return bool(port.connected_ports())


def set_state_inline_control_read_only(control: QtWidgets.QWidget, *, read_only: bool) -> None:
    """
    Best-effort toggle for inline state controls hosted in the node item.
    """
    set_control_read_only(control, read_only=read_only)


def _preview_force_read_only(node_item: Any) -> bool:
    try:
        return bool(node_item._f8_preview_read_only)
    except _NODE_VALUE_ACCESS_ERRORS:
        return False


def refresh_state_inline_control_read_only(node_item: Any) -> None:
    """
    Refresh readonly state for already-built inline state controls.
    """
    node = node_item._backend_node()
    if node is None:
        return
    try:
        fields = list(node.effective_state_fields() or [])
    except _NODE_VALUE_ACCESS_ERRORS:
        fields = []
    for field in fields:
        info = state_field_info(field)
        if info is None or not info.show_on_node:
            continue
        name = info.name
        read_only = (
            _preview_force_read_only(node_item)
            or info.access_str == "ro"
            or is_state_inline_input_connected(node_item, name)
        )
        try:
            binding = node_item._state_inline_bindings.get(name)
        except AttributeError:
            binding = None
        if binding is not None:
            binding.set_read_only(bool(read_only))
            continue
        ctrl = node_item._state_inline_controls.get(name)
        if ctrl is not None:
            set_state_inline_control_read_only(ctrl, read_only=bool(read_only))


def sync_state_inline_controls_from_graph_property(node_item: Any, node: Any, name: str, value: Any) -> None:
    """
    Keep inline state widgets in sync with NodeGraphQt properties.

    The inspector already tracks these through NodeGraphQt's own property
    widgets; since inline widgets are custom QWidgets, mirror updates here to
    get the same "two-way binding" behavior.
    """
    try:
        if str(node.id or "") != str(node_item.id or ""):
            return
    except (AttributeError, TypeError):
        return
    key = str(name or "").strip()
    if not key:
        return
    preview_updater = None
    if key in WAVE_PREVIEW_DEPENDENCY_FIELDS:
        preview_updater = node_item._state_inline_updaters.get("preview")
    pattern_updater = None
    if key in WAVE_PATTERN_EDITOR_DEPENDENCY_FIELDS:
        pattern_updater = node_item._state_inline_updaters.get("points")

    updater = node_item._state_inline_updaters.get(key)
    if updater is not None:
        try:
            updater(value)
        except _INLINE_STATE_UPDATER_ERRORS:
            logger.exception("inline state updater failed nodeId=%s key=%s", _node_item_id(node_item), key)

    if preview_updater is not None and preview_updater is not updater:
        try:
            preview_value = node.get_property("preview")
        except KeyError:
            preview_value = None
        try:
            preview_updater(preview_value)
        except _INLINE_STATE_UPDATER_ERRORS:
            logger.exception("inline wave preview updater failed nodeId=%s key=%s", _node_item_id(node_item), key)

    if pattern_updater is not None and pattern_updater is not updater:
        try:
            points_value = node.get_property("points")
        except KeyError:
            points_value = None
        try:
            pattern_updater(points_value)
        except _INLINE_STATE_UPDATER_ERRORS:
            logger.exception("inline wave pattern updater failed nodeId=%s key=%s", _node_item_id(node_item), key)

    refresh_state_inline_option_pools(node_item, key)


def refresh_state_inline_option_pools(node_item: Any, changed_field: str) -> None:
    """
    If `changed_field` is used as an option-pool, refresh all dependent option controls.
    """
    pool = str(changed_field or "").strip()
    if not has_state_inline_option_pool_dependents(node_item, pool):
        return
    node = node_item._backend_node()
    if node is None:
        return
    try:
        bindings = node_item._state_inline_bindings
    except AttributeError:
        bindings = {}
    for field, pool_name in list(node_item._state_inline_option_pools.items()):
        if pool_name != pool:
            continue
        binding = bindings.get(field)
        if binding is not None and binding.refresh_options is not None:
            try:
                binding.refresh_options()
            except (RuntimeError, TypeError):
                continue
            continue
        ctrl = node_item._state_inline_controls.get(field)
        if ctrl is None:
            continue
        try:
            pool_value = node.get_property(pool)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pool_value = None
        items = resolve_pool_items(pool_value)
        try:
            selected_value = node.get_property(field)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            try:
                selected_value = ctrl.value()
            except _NODE_VALUE_ACCESS_ERRORS:
                selected_value = None
        try:
            ctrl.set_options(items, labels=items)
            ctrl.set_value(selected_value)
        except (AttributeError, RuntimeError, TypeError):
            continue


def has_state_inline_option_pool_dependents(node_item: Any, changed_field: str) -> bool:
    pool = str(changed_field or "").strip()
    if not pool:
        return False
    try:
        option_pools = node_item._state_inline_option_pools
    except AttributeError:
        return False
    for pool_name in option_pools.values():
        if str(pool_name or "").strip() == pool:
            return True
    return False


def toggle_state_inline_section(node_item: Any, name: str, expanded: bool) -> None:
    state_name = str(name)
    old_scene_rect = None
    try:
        old_scene_rect = node_item.mapToScene(node_item.boundingRect()).boundingRect()
    except RuntimeError:
        old_scene_rect = None

    node_item._state_inline_expanded[state_name] = bool(expanded)
    node = node_item._backend_node()
    if node is not None:
        try:
            set_state_inline_expanded(node, state_name=state_name, expanded=bool(expanded))
        except AttributeError:
            logger.exception("node missing ui_state/set_ui_state; cannot persist expand state")
    btn = node_item._state_inline_toggles.get(state_name)
    if btn is not None:
        try:
            btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        except RuntimeError:
            pass
    body = node_item._state_inline_bodies.get(state_name)
    if body is not None:
        try:
            body.setVisible(bool(expanded))
        except RuntimeError:
            pass

    def _redraw_and_invalidate() -> None:
        node_item.draw_node()
        new_scene_rect = node_item.mapToScene(node_item.boundingRect()).boundingRect()
        rect = new_scene_rect
        if old_scene_rect is not None:
            rect = old_scene_rect.united(new_scene_rect)
        rect = rect.adjusted(-6, -6, 6, 6)
        scene = node_item.scene()
        if scene is not None:
            scene.update(rect)
        viewer = node_item.viewer()
        if viewer is not None:
            viewer.viewport().update()

    try:
        QtCore.QTimer.singleShot(0, _redraw_and_invalidate)
    except RuntimeError:
        _redraw_and_invalidate()


def build_state_inline_control(
    node_item: Any,
    state_field: StateFieldInfo,
    *,
    widget_parent: QtWidgets.QWidget | None = None,
) -> QtWidgets.QWidget:
    name = state_field.name
    ui_raw = state_field.ui_control
    parsed_ui = parse_ui_control(ui_raw)
    ui = parsed_ui.control_name
    schema = state_field.value_schema
    access_s = state_field.access_str
    schema_type_value = (schema_type(schema) or "") if schema is not None else ""

    enum_items = node_item._schema_enum_items(schema)
    lo, hi = node_item._schema_numeric_range(schema)
    select_pool_field = parse_select_pool(ui_raw)
    multiselect_pool_field = parse_multiselect_pool(ui_raw)
    field_tooltip = state_field.tooltip if state_field.tooltip != name else ""

    def _common_style(widget: QtWidgets.QWidget) -> None:
        widget.setStyleSheet(inline_control_qss())

    def _install_global_tooltip_filter(widget: QtWidgets.QWidget) -> None:
        tooltip_filter = F8ForceGlobalToolTipFilter(widget)
        widget.installEventFilter(tooltip_filter)
        node_item._tooltip_filters.append(tooltip_filter)

    def _set_node_value(value: Any, *, push_undo: bool) -> None:
        node = node_item._backend_node()
        if node is None or not name:
            return
        try:
            node.set_property(name, value, push_undo=push_undo)
        except TypeError:
            node.set_property(name, value)

    def _get_node_value() -> Any:
        node = node_item._backend_node()
        if node is None or not name:
            return None
        try:
            return node.get_property(name)
        except KeyError:
            return None

    def _get_node_property(field_name: str) -> Any:
        node = node_item._backend_node()
        if node is None:
            return None
        try:
            return node.get_property(field_name)
        except KeyError:
            return None

    def _pool_items(pool_field: str | None) -> list[str]:
        if not pool_field:
            return []
        node = node_item._backend_node()
        if node is None:
            return []
        try:
            value = node.get_property(pool_field)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            return []
        return resolve_pool_items(value)

    read_only = (
        _preview_force_read_only(node_item) or access_s == "ro" or node_item._is_state_inline_input_connected(name)
    )
    spec = StateControlSpec(
        name=name,
        label=state_field.label or name,
        ui_control=ui_raw,
        ui_language=state_field.ui_language or "plaintext",
        schema_type=schema_type_value,
        enum_items=enum_items,
        minimum=lo,
        maximum=hi,
        field_tooltip=field_tooltip,
        select_pool_field=select_pool_field,
        multiselect_pool_field=multiselect_pool_field,
        is_image_b64=schema_type_value == "string" and (ui in {"image", "image_b64", "img"} or "b64" in name.lower()),
        range_integer=schema_array_item_type(schema) == "integer",
    )
    try:
        graph = node_item._graph()
    except _NODE_ACCESS_ERRORS:
        graph = None
    node = node_item._backend_node()
    node_id = _backend_node_id(node) if node is not None else ""

    try:
        viewer = node_item.viewer()
    except _NODE_ACCESS_ERRORS:
        viewer = None
    warning_parent = None
    if viewer is not None:
        try:
            warning_parent = viewer.window() if viewer.window() is not None else viewer
        except (AttributeError, RuntimeError, TypeError):
            warning_parent = viewer

    text_binding = node_text_editor_binding(graph, node_id, name, warning_parent=warning_parent)

    def _get_fallback_code_value() -> str:
        current = _get_node_value()
        return "" if current is None else str(current)

    def _set_fallback_code_value(updated: str) -> bool:
        _set_node_value(updated, push_undo=True)
        return node_item._backend_node() is not None

    def _fallback_code_target_exists() -> bool:
        return node_item._backend_node() is not None

    binding = build_inline_control_binding(
        spec=spec,
        read_only=read_only,
        widget_parent=widget_parent,
        value_getter=_get_node_value,
        value_setter=_set_node_value,
        property_value_getter=_get_node_property,
        pool_resolver=lambda pool_field: _pool_items(pool_field),
        code_title=f"{node_item.name} - {spec.label}",
        code_value_getter=text_binding.value_getter if text_binding is not None else _get_fallback_code_value,
        code_value_setter=text_binding.value_setter if text_binding is not None else _set_fallback_code_value,
        code_target_exists_provider=text_binding.target_exists
        if text_binding is not None
        else _fallback_code_target_exists,
        assist_context=_editor_assist_context(
            graph,
            node_id=node_id,
            state_field_name=name,
            ui_control=ui_raw,
            language=parsed_ui.ui_language or "plaintext",
        ),
        assist_context_provider=lambda: _editor_assist_context(
            graph,
            node_id=node_id,
            state_field_name=name,
            ui_control=ui_raw,
            language=parsed_ui.ui_language or "plaintext",
        ),
        editor_session_key=text_binding.session_key if text_binding is not None else None,
        style_applier=_common_style,
        text_palette_applier=_apply_text_palette,
        tooltip_filter_installer=_install_global_tooltip_filter,
    )
    try:
        bindings = node_item._state_inline_bindings
    except AttributeError:
        bindings = {}
        node_item._state_inline_bindings = bindings
    bindings[name] = binding
    node_item._state_inline_updaters[name] = binding.apply_value
    if select_pool_field:
        node_item._state_inline_option_pools[name] = select_pool_field
    if multiselect_pool_field:
        node_item._state_inline_option_pools[name] = multiselect_pool_field
    return binding.widget


def ensure_state_inline_controls(node_item: Any) -> None:
    node_item._ensure_graph_property_hook()
    node = node_item._backend_node()
    if node is None:
        return
    try:
        fields = list(node.effective_state_fields() or [])
    except _NODE_VALUE_ACCESS_ERRORS:
        spec = _node_spec(node)
        if spec is None:
            fields = []
        else:
            try:
                fields = list(spec.stateFields or [])
            except _STATE_SPEC_ERRORS:
                fields = []

    show: list[StateFieldInfo] = []
    for field in fields:
        info = state_field_info(field)
        if info is None or not info.show_on_node:
            continue
        show.append(info)

    desired = [info.name for info in show]

    # delete stale widgets.
    for name in list(node_item._state_inline_proxies.keys()):
        if name in desired:
            continue
        proxy = node_item._state_inline_proxies.pop(name, None)
        node_item._state_inline_controls.pop(name, None)
        node_item._state_inline_bindings.pop(name, None)
        node_item._state_inline_updaters.pop(name, None)
        node_item._state_inline_toggles.pop(name, None)
        node_item._state_inline_headers.pop(name, None)
        node_item._state_inline_bodies.pop(name, None)
        node_item._state_inline_expanded.pop(name, None)
        node_item._state_inline_option_pools.pop(name, None)
        node_item._state_inline_ctrl_serial.pop(name, None)
        if proxy is None:
            continue
        old = None
        try:
            old = proxy.widget()
        except _QT_PROXY_ACCESS_ERRORS:
            old = None
        try:
            proxy.setWidget(None)
        except RuntimeError:
            pass
        if old is not None:
            dispose_detached_proxy_widget(old, context=f"inline-state-remove:{name}")
        try:
            proxy.setParentItem(None)
            if node_item.scene() is not None:
                node_item.scene().removeItem(proxy)
        except RuntimeError:
            pass

    for info in show:
        # Always keep label/tooltip up to date without rebuilding.
        name = info.name
        label = info.label or name
        tip = info.tooltip or name
        btn_existing = node_item._state_inline_toggles.get(name)
        if btn_existing is not None:
            try:
                btn_existing.set_full_text(label)
            except RuntimeError:
                pass
            try:
                btn_existing.setToolTip(tip)
            except RuntimeError:
                pass
        control_existing = node_item._state_inline_controls.get(name)
        if control_existing is not None:
            try:
                _refresh_state_inline_control_metadata(node_item, control_existing, info)
            except RuntimeError:
                pass

        ctrl_sig = state_inline_control_serial(node_item, info)
        if (
            name in node_item._state_inline_proxies
            and ctrl_sig
            and ctrl_sig == node_item._state_inline_ctrl_serial.get(name, "")
        ):
            continue

        # Default collapsed; restore persisted expand state from ui overrides.
        expanded = False
        persisted_expanded = state_inline_expanded(node, name)
        if persisted_expanded is not None:
            expanded = bool(persisted_expanded)
        expanded = bool(node_item._state_inline_expanded.get(name, expanded))
        panel = QtWidgets.QWidget()
        header, btn = build_inline_header_button(
            label=label,
            tooltip=tip,
            expandable=True,
            expanded=expanded,
            parent=panel,
        )

        # Body: control widget (collapsed by default).
        body = QtWidgets.QWidget(panel)
        control = node_item._build_state_inline_control(info, widget_parent=body)
        body_lay = QtWidgets.QVBoxLayout(body)
        body_lay.setContentsMargins(8, 0, 8, 6)
        body_lay.setSpacing(0)
        body_lay.addWidget(control)
        body.setVisible(expanded)
        body.setStyleSheet(transparent_widget_qss())

        panel_lay = QtWidgets.QVBoxLayout(panel)
        panel_lay.setContentsMargins(0, 0, 0, 0)
        panel_lay.setSpacing(0)
        panel_lay.addWidget(header)
        panel_lay.addWidget(body)
        panel.setProperty("_f8_state_panel", True)
        panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        # panel.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        panel.setStyleSheet(transparent_header_qss())

        # Connect toggle.
        btn.toggled.connect(lambda v, _n=name: node_item._toggle_state_inline_section(_n, bool(v)))  # type: ignore[attr-defined]
        btn.pressed.connect(node_item._select_node_from_embedded_widget)  # type: ignore[attr-defined]

        # Install/replace proxy.
        proxy = node_item._state_inline_proxies.get(name)
        if proxy is None:
            proxy = QtWidgets.QGraphicsProxyWidget(node_item)
            node_item._state_inline_proxies[name] = proxy

        old = None
        try:
            old = proxy.widget()
        except _QT_PROXY_ACCESS_ERRORS:
            old = None
        proxy.setWidget(panel)

        if old is not None and old is not panel:
            dispose_detached_proxy_widget(old, context=f"inline-state-replace:{name}")

        node_item._state_inline_controls[name] = control
        node_item._state_inline_toggles[name] = btn
        node_item._state_inline_headers[name] = header
        node_item._state_inline_bodies[name] = body
        node_item._state_inline_expanded[name] = expanded
        if ctrl_sig:
            node_item._state_inline_ctrl_serial[name] = ctrl_sig
        try:
            node_item._invalidate_layout_metrics()
            node_item._prepare_layout_metrics()
            node_item.sync_proxy_mode(force=True)
        except (AttributeError, RuntimeError, TypeError):
            pass

    reordered_proxies: dict[str, QtWidgets.QGraphicsProxyWidget] = {}
    reordered_controls: dict[str, QtWidgets.QWidget] = {}
    reordered_bindings: dict[str, Any] = {}
    reordered_updaters: dict[str, Any] = {}
    reordered_toggles: dict[str, Any] = {}
    reordered_headers: dict[str, QtWidgets.QWidget] = {}
    reordered_bodies: dict[str, QtWidgets.QWidget] = {}
    reordered_expanded: dict[str, bool] = {}
    reordered_option_pools: dict[str, str] = {}
    reordered_ctrl_serial: dict[str, str] = {}

    for name in desired:
        proxy = node_item._state_inline_proxies.get(name)
        control = node_item._state_inline_controls.get(name)
        binding = node_item._state_inline_bindings.get(name)
        updater = node_item._state_inline_updaters.get(name)
        toggle = node_item._state_inline_toggles.get(name)
        header = node_item._state_inline_headers.get(name)
        body = node_item._state_inline_bodies.get(name)
        if proxy is not None:
            reordered_proxies[name] = proxy
        if control is not None:
            reordered_controls[name] = control
        if binding is not None:
            reordered_bindings[name] = binding
        if updater is not None:
            reordered_updaters[name] = updater
        if toggle is not None:
            reordered_toggles[name] = toggle
        if header is not None:
            reordered_headers[name] = header
        if body is not None:
            reordered_bodies[name] = body
        if name in node_item._state_inline_expanded:
            reordered_expanded[name] = bool(node_item._state_inline_expanded.get(name, False))
        if name in node_item._state_inline_option_pools:
            reordered_option_pools[name] = str(node_item._state_inline_option_pools.get(name, "") or "")
        if name in node_item._state_inline_ctrl_serial:
            reordered_ctrl_serial[name] = str(node_item._state_inline_ctrl_serial.get(name, "") or "")

    node_item._state_inline_proxies.clear()
    node_item._state_inline_controls.clear()
    node_item._state_inline_bindings.clear()
    node_item._state_inline_updaters.clear()
    node_item._state_inline_toggles.clear()
    node_item._state_inline_headers.clear()
    node_item._state_inline_bodies.clear()
    node_item._state_inline_expanded.clear()
    node_item._state_inline_option_pools.clear()
    node_item._state_inline_ctrl_serial.clear()

    node_item._state_inline_proxies.update(reordered_proxies)
    node_item._state_inline_controls.update(reordered_controls)
    node_item._state_inline_bindings.update(reordered_bindings)
    node_item._state_inline_updaters.update(reordered_updaters)
    node_item._state_inline_toggles.update(reordered_toggles)
    node_item._state_inline_headers.update(reordered_headers)
    node_item._state_inline_bodies.update(reordered_bodies)
    node_item._state_inline_expanded.update(reordered_expanded)
    node_item._state_inline_option_pools.update(reordered_option_pools)
    node_item._state_inline_ctrl_serial.update(reordered_ctrl_serial)
