from typing import Type, Callable

from ..operator_instance import OperatorInstance
from NodeGraphQt import BaseNode
from .generic import GenericOperatorNode


type OperatorRenderer = Callable[[OperatorInstance], BaseNode]


class OperatorRendererRegistry:
    """Registry for renderer classes keyed by rendererClass."""

    def __init__(self) -> None:
        self._renderers: dict[str, OperatorRenderer] = {}
        self._renderers["default"] = GenericOperatorNode

    def register(self, renderer_key: str, renderer: OperatorRenderer, *, overwrite: bool = False) -> None:
        if renderer_key in self._renderers and not overwrite:
            raise ValueError(f'renderer "{renderer_key}" already registered')
        self._renderers[renderer_key] = renderer

    def unregister(self, renderer_key: str) -> None:
        self._renderers.pop(renderer_key, None)

    def get(self, renderer_key: str) -> OperatorRenderer:
        if renderer_key not in self._renderers:
            renderer_key = "default"
        return self._renderers[renderer_key]

    def keys(self) -> list[str]:
        return list(self._renderers.keys())
