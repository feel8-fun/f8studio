from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class StateFieldDescriptor:
    name: str
    access: str
    ui_control: str
    ui_language: str
    value_schema: Any


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
