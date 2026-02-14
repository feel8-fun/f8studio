from NodeGraphQt import NodeObject, BaseNode
from typing import ClassVar

# from .internal.base import F8BaseRenderNode

# from .op_generic import GenericOpRenderNode

from ..nodegraph.service_basenode import F8StudioServiceBaseNode
from ..nodegraph.container_basenode import F8StudioContainerBaseNode
from ..nodegraph.operator_basenode import F8StudioOperatorBaseNode
from .pystudio_print import PyStudioPrintNode
from .pystudio_timeseries import PyStudioTimeSeriesNode
from .pystudio_videoshm import PyStudioVideoShmNode
from .pystudio_audioshm import PyStudioAudioShmNode
from .pystudio_trackviz import PyStudioTrackVizNode
from .pystudio_template_tracker import PyStudioTemplateTrackerNode
from .pystudio_skeleton3d import PyStudioSkeleton3DNode


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
        self._renderers["default_svc"] = F8StudioServiceBaseNode
        self._renderers["default_op"] = F8StudioOperatorBaseNode
        self._renderers["default_container"] = F8StudioContainerBaseNode
        self._renderers["pystudio_print"] = PyStudioPrintNode
        self._renderers["pystudio_timeseries"] = PyStudioTimeSeriesNode
        self._renderers["pystudio_videoshm"] = PyStudioVideoShmNode
        self._renderers["pystudio_audioshm"] = PyStudioAudioShmNode
        self._renderers["pystudio_trackviz"] = PyStudioTrackVizNode
        self._renderers["pystudio_template_tracker"] = PyStudioTemplateTrackerNode
        self._renderers["pystudio_skeleton3d"] = PyStudioSkeleton3DNode

    def register(self, renderer_key: str, renderer: type[NodeObject]) -> None:
        if renderer_key in self._renderers:
            raise ValueError(f'renderer "{renderer_key}" already registered')
        if not issubclass(renderer, NodeObject):
            raise TypeError("renderer must subclass NodeObject")
        self._renderers[renderer_key] = renderer

    def unregister(self, renderer_key: str) -> None:
        self._renderers.pop(renderer_key, None)

    def get(self, renderer_key: str, fallback_key: str) -> type[NodeObject]:
        if renderer_key not in self._renderers and fallback_key:
            renderer_key = fallback_key
        return self._renderers[renderer_key]

    def keys(self) -> list[str]:
        return list(self._renderers.keys())
