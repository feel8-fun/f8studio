from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class RendererRegistration:
    renderer_class: str
    node_class: type[Any]


@dataclass(frozen=True)
class StateControlRegistration:
    ui_control: str
    factory: Callable[..., Any]


@dataclass(frozen=True)
class CommandHandlerRegistration:
    operator_class: str
    handler_class: type[Any]


@dataclass(frozen=True)
class StudioPluginManifest:
    plugin_id: str
    plugin_name: str
    plugin_version: str
    renderers: tuple[RendererRegistration, ...] = field(default_factory=tuple)
    state_controls: tuple[StateControlRegistration, ...] = field(default_factory=tuple)
    command_handlers: tuple[CommandHandlerRegistration, ...] = field(default_factory=tuple)
