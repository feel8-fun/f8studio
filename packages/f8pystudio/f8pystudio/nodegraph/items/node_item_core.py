from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from f8pysdk import F8ServiceSpec


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


def model_extra(obj: Any) -> dict[str, Any]:
    """
    Best-effort access to pydantic v2 extra fields without RTTI (`getattr`).
    """
    try:
        extra = obj.model_extra
        return extra if isinstance(extra, dict) else {}
    except (AttributeError, RuntimeError, TypeError):
        pass
    try:
        extra = obj.__pydantic_extra__
        return extra if isinstance(extra, dict) else {}
    except (AttributeError, RuntimeError, TypeError):
        return {}


def service_exec_ports(spec: F8ServiceSpec) -> tuple[list[str], list[str]]:
    """
    F8ServiceSpec doesn't declare exec ports, but it allows extra fields.
    """
    extra = model_extra(spec)
    in_raw = extra.get("execInPorts")
    out_raw = extra.get("execOutPorts")
    exec_in = [str(x) for x in list(in_raw or [])] if isinstance(in_raw, (list, tuple)) else []
    exec_out = [str(x) for x in list(out_raw or [])] if isinstance(out_raw, (list, tuple)) else []
    return exec_in, exec_out


def state_field_info(field: Any) -> StateFieldInfo | None:
    if isinstance(field, dict):
        name = str(field.get("name") or "").strip()
    else:
        try:
            name = str(field.name or "").strip()
        except Exception:
            return None
    if not name:
        return None

    if isinstance(field, dict):
        show_on_node = bool(field.get("showOnNode") or False)
    else:
        try:
            show_on_node = bool(field.showOnNode)
        except Exception:
            show_on_node = False

    if isinstance(field, dict):
        label = str(field.get("label") or "").strip() or name
        tooltip = str(field.get("description") or "").strip() or name
        ui_control = str(field.get("uiControl") or "").strip()
        ui_language = str(field.get("uiLanguage") or "")
        value_schema = field.get("valueSchema")
        access = field.get("access")
        required = bool(field.get("required") or False)
    else:
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
