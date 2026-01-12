from NodeGraphQt import NodeObject, BaseNode

from .generic import GenericRenderNode
from .service_engine import OperatorRunnerNode


class RenderNodeRegistry:
    """Registry for renderer classes keyed by rendererClass."""
    
    @staticmethod
    def instance() -> "RenderNodeRegistry":
        # Singleton instance accessor.
        if not hasattr(RenderNodeRegistry, "_instance"):
            RenderNodeRegistry._instance = RenderNodeRegistry()
        return RenderNodeRegistry._instance

    def __init__(self) -> None:
        self._renderers: dict[str, NodeObject] = {}
        self._renderers["default"] = GenericRenderNode
        self._renderers["f8.operator_runner"] = OperatorRunnerNode

    def register(self, renderer_key: str, renderer: type[NodeObject]) -> None:
        if renderer_key in self._renderers:
            raise ValueError(f'renderer "{renderer_key}" already registered')
        if not issubclass(renderer, NodeObject):
            raise TypeError("renderer must subclass NodeObject")
        self._renderers[renderer_key] = renderer

    def unregister(self, renderer_key: str) -> None:
        self._renderers.pop(renderer_key, None)

    def get(self, renderer_key: str, fallback_key: str = "default") -> type[NodeObject]:
        if renderer_key not in self._renderers and fallback_key:
            renderer_key = fallback_key
        return self._renderers[renderer_key]

    def keys(self) -> list[str]:
        return list(self._renderers.keys())
