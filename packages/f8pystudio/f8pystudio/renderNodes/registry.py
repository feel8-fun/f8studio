from NodeGraphQt import NodeObject, BaseNode

from .internal.base import F8BaseRenderNode
from .op_generic import GenericOpRenderNode
from .svc_container import ContainerSvcRenderNode


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
        self._renderers["default"] = F8BaseRenderNode
        self._renderers["default_svc"] = F8BaseRenderNode
        self._renderers["default_op"] = GenericOpRenderNode
        self._renderers["default_container"] = ContainerSvcRenderNode

        

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
