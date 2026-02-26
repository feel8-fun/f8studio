import logging
from typing import ClassVar, Literal

from NodeGraphQt import NodeObject, BaseNode

# from .internal.base import F8BaseRenderNode

# from .op_generic import GenericOpRenderNode

from ..nodegraph.service_basenode import F8StudioServiceBaseNode
from ..nodegraph.container_basenode import F8StudioContainerBaseNode
from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from .viz_text import VizTextRenderNode
from .viz_wave import VizWaveRenderNode
from .viz_video import VizVideoRenderNode
from .viz_audio import VizAudioRenderNode
from .viz_track import VizTrackRenderNode
from .template_match_capture import TemplateMatchCaptureRenderNode
from .viz_three_d import VizThreeDRenderNode
from .viz_tcode import VizTCodeRenderNode


class RenderNodeRegistry:
    """Registry for renderer classes keyed by rendererClass."""

    _instance: ClassVar["RenderNodeRegistry | None"] = None

    @staticmethod
    def instance() -> "RenderNodeRegistry":
        # Singleton instance accessor.
        if RenderNodeRegistry._instance is None:
            RenderNodeRegistry._instance = RenderNodeRegistry()
        return RenderNodeRegistry._instance

    def __init__(self) -> None:
        self._renderers: dict[str, NodeObject] = {}
        self._fallback_warned: set[tuple[str, str]] = set()
        self._renderers["default_svc"] = F8StudioServiceBaseNode
        self._renderers["default_op"] = F8StudioOperatorBaseNode
        self._renderers["default_container"] = F8StudioContainerBaseNode
        self._renderers["template_match_capture"] = TemplateMatchCaptureRenderNode
        self._renderers["viz_text"] = VizTextRenderNode
        self._renderers["viz_wave"] = VizWaveRenderNode
        self._renderers["viz_video"] = VizVideoRenderNode
        self._renderers["viz_audio"] = VizAudioRenderNode
        self._renderers["viz_track"] = VizTrackRenderNode
        self._renderers["viz_three_d"] = VizThreeDRenderNode
        self._renderers["viz_tcode"] = VizTCodeRenderNode

    def register(self, renderer_key: str, renderer: type[NodeObject]) -> None:
        if renderer_key in self._renderers:
            raise ValueError(f'renderer "{renderer_key}" already registered')
        if not issubclass(renderer, NodeObject):
            raise TypeError("renderer must subclass NodeObject")
        self._renderers[renderer_key] = renderer

    def unregister(self, renderer_key: str) -> None:
        self._renderers.pop(renderer_key, None)

    def get(
        self,
        renderer_key: str,
        *,
        node_kind: Literal["service", "operator"] | None = None,
    ) -> type[NodeObject]:
        key = str(renderer_key or "").strip()
        if not key:
            key = "default_op" if node_kind == "operator" else "default_svc"
        renderer = self._renderers.get(key)
        if renderer is not None:
            return renderer

        fallback_key = "default_op" if node_kind == "operator" else "default_svc"
        fallback_renderer = self._renderers.get(fallback_key)
        if fallback_renderer is None:
            available = ", ".join(sorted(self._renderers.keys()))
            raise KeyError(
                f'unknown renderer key "{key}" and fallback "{fallback_key}" missing. '
                f"available: [{available}]"
            )
        warn_key = (key, fallback_key)
        if warn_key not in self._fallback_warned:
            self._fallback_warned.add(warn_key)
            logging.getLogger(__name__).warning(
                "Unknown renderer '%s'; fallback to '%s'.",
                key,
                fallback_key,
            )
        return fallback_renderer

    def keys(self) -> list[str]:
        return list(self._renderers.keys())
