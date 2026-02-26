from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class StateFieldInfo:
    name: str
    label: str
    tooltip: str
    show_on_node: bool
    access: Any
    access_str: str
    required: bool
    ui_control: str
    ui_language: str
    value_schema: Any


def port_name(port: Any) -> str:
    """
    NodeGraphQt Port exposes `name()` (method).
    """
    try:
        return str(port.name() or "")
    except (AttributeError, RuntimeError, TypeError):
        pass
    try:
        return str(port.name or "")
    except (AttributeError, RuntimeError, TypeError):
        return ""

def state_field_info(field: Any) -> StateFieldInfo | None:
    try:
        name = str(field.name or "").strip()
    except Exception:
        return None
    if not name:
        return None

    try:
        show_on_node = bool(field.showOnNode)
    except Exception:
        show_on_node = False
    try:
        label = str(field.label or "").strip() or name
    except Exception:
        label = name
    try:
        tooltip = str(field.description or "").strip() or name
    except Exception:
        tooltip = name
    try:
        ui_control = str(field.uiControl or "").strip()
    except Exception:
        ui_control = ""
    try:
        ui_language = str(field.uiLanguage or "")
    except Exception:
        ui_language = ""
    try:
        value_schema = field.valueSchema
    except Exception:
        value_schema = None
    try:
        access = field.access
    except Exception:
        access = None
    try:
        required = bool(field.required)
    except Exception:
        required = False

    if isinstance(access, enum.Enum):
        access_value = access.value
    else:
        access_value = access if access is not None else ""
    access_str = str(access_value or "").strip().lower()

    return StateFieldInfo(
        name=name,
        label=label,
        tooltip=tooltip,
        show_on_node=show_on_node,
        access=access,
        access_str=access_str,
        required=required,
        ui_control=ui_control,
        ui_language=ui_language,
        value_schema=value_schema,
    )
