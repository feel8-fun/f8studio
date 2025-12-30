from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..operator.instance import OperatorInstance

from .renderers.base import BaseOpRenderer

class RendererRegistry:
    """Registry for node renderers keyed by rendererClass."""

    def __init__(self) -> None:
        self._renderers: dict[str, BaseOpRenderer] = {}

    def register(self, renderer_class: str, renderer: 'BaseOpRenderer', *, overwrite: bool = False) -> None:
        if renderer_class in self._renderers and not overwrite:
            raise ValueError(f'rendererClass "{renderer_class}" already registered')
        self._renderers[renderer_class] = renderer

    def unregister(self, renderer_class: str) -> None:
        self._renderers.pop(renderer_class, None)

    def get(self, renderer_class: str) -> 'BaseOpRenderer':
        try:
            return self._renderers[renderer_class]
        except KeyError as exc:
            raise KeyError(f'rendererClass "{renderer_class}" not found') from exc


class NodeRenderContext:
    """
    Context passed to renderers to build node appearance.

    Provides helpers to add pins while keeping UI bookkeeping in sync.
    """

    def __init__(self, *, node_tag: int, instance: OperatorInstance, attr_meta: dict[int, dict[str, str]]) -> None:
        self.node_tag = node_tag
        self.instance = instance
        self._attr_meta = attr_meta

    def add_pin(
        self,
        port: str,
        *,
        kind: str,
        direction: str,
        shape: int,
        label: str | None = None,
    ) -> None:
        from dearpygui import dearpygui as dpg

        attr_type = dpg.mvNode_Attr_Output if direction == 'out' else dpg.mvNode_Attr_Input
        attr_id = dpg.add_node_attribute(
            label=label or port,
            parent=self.node_tag,
            attribute_type=attr_type,
            shape=shape,
        )
        dpg.add_spacer(parent=attr_id)

        self._attr_meta[attr_id] = {
            'node_id': self.instance.id,
            'kind': kind,
            'direction': direction,
            'port': port,
        }

    def add_state_pin(self, field_name: str, label_suffix: str, direction: str) -> None:
        field = next((f for f in self.instance.spec.states or [] if f.name == field_name), None)
        if not field:
            raise ValueError(f'state field {field_name} not found on {self.instance.id}')
        label = f'{field.label or field.name} [{label_suffix}]'
        from dearpygui import dearpygui as dpg

        shape = dpg.mvNode_PinShape_Quad if direction == 'out' else dpg.mvNode_PinShape_QuadFilled
        self.add_pin(field.name, kind='state', direction=direction, shape=shape, label=label)




