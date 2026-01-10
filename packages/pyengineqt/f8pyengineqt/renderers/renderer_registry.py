from .generic import GenericNode, UiOperatorNode
from .editor_log import EditorLogNode


type OperatorRenderer = type[GenericNode]


class RendererRegistry:
    """Registry for renderer classes keyed by rendererClass."""

    @staticmethod
    def instance() -> "RendererRegistry":
        """Get the global singleton instance of the registry."""
        global _GLOBAL_RENDERER_REGISTRY
        try:
            return _GLOBAL_RENDERER_REGISTRY
        except NameError:
            _GLOBAL_RENDERER_REGISTRY = RendererRegistry()
            return _GLOBAL_RENDERER_REGISTRY

    def __init__(self) -> None:
        self._renderers: dict[str, OperatorRenderer] = {}
        self._renderers["default"] = GenericNode
        self._renderers["generic"] = GenericNode
        self._renderers["ui"] = UiOperatorNode
        self._renderers["editor_log"] = EditorLogNode

    def register(self, renderer_key: str, renderer: type[GenericNode], *, overwrite: bool = False) -> None:
        if renderer_key in self._renderers and not overwrite:
            raise ValueError(f'renderer "{renderer_key}" already registered')
        if not issubclass(renderer, GenericNode):
            raise TypeError("renderer must subclass GenericNode")
        self._renderers[renderer_key] = renderer

    def unregister(self, renderer_key: str) -> None:
        self._renderers.pop(renderer_key, None)

    def get(self, renderer_key: str) -> type[GenericNode]:
        if renderer_key not in self._renderers:
            renderer_key = "default"
        return self._renderers[renderer_key]

    def keys(self) -> list[str]:
        return list(self._renderers.keys())
