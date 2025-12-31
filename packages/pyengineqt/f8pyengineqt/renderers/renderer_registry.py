from typing import Type

from .generic import GenericOperatorNode, OperatorNodeBase, UiOperatorNode


type OperatorRenderer = type[OperatorNodeBase]


class OperatorRendererRegistry:
    """Registry for renderer classes keyed by rendererClass."""

    @staticmethod
    def instance() -> "OperatorRendererRegistry":
        """Get the global singleton instance of the registry."""
        global _GLOBAL_RENDERER_REGISTRY
        try:
            return _GLOBAL_RENDERER_REGISTRY
        except NameError:
            _GLOBAL_RENDERER_REGISTRY = OperatorRendererRegistry()
            return _GLOBAL_RENDERER_REGISTRY

    def __init__(self) -> None:
        self._renderers: dict[str, OperatorRenderer] = {}
        self._renderers["default"] = GenericOperatorNode
        self._renderers["generic"] = GenericOperatorNode

    def register(self, renderer_key: str, renderer: type[OperatorNodeBase], *, overwrite: bool = False) -> None:
        if renderer_key in self._renderers and not overwrite:
            raise ValueError(f'renderer "{renderer_key}" already registered')
        if not issubclass(renderer, OperatorNodeBase):
            raise TypeError("renderer must subclass OperatorNodeBase")
        self._renderers[renderer_key] = renderer

    def unregister(self, renderer_key: str) -> None:
        self._renderers.pop(renderer_key, None)

    def get(self, renderer_key: str) -> type[OperatorNodeBase]:
        if renderer_key not in self._renderers:
            renderer_key = "default"
        return self._renderers[renderer_key]

    def keys(self) -> list[str]:
        return list(self._renderers.keys())
