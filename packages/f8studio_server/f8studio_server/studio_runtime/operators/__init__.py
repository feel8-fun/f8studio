from typing import Any, Protocol
from ._viz_base import StudioVizRuntimeNodeBase

from f8pysdk.registry import Registry, RuntimeNodeRegistry, create_runtime_node_registry
from f8pysdk.specs import F8RuntimeNode

from ..identifiers import SERVICE_CLASS
from ..presentation import PresentationOutlet

from .viz_text import VizTextRuntimeNode, register_operator as register_viz_text
from .viz_wave import VizWaveRuntimeNode, register_operator as register_viz_wave
from .viz_video import VizVideoRuntimeNode, register_operator as register_viz_video
from .viz_audio import VizAudioRuntimeNode, register_operator as register_viz_audio
from .control_panel import ControlPanelRuntimeNode, register_operator as register_control_panel
from .backdrop import BackdropRuntimeNode, register_operator as register_backdrop
from .note import NoteRuntimeNode, register_operator as register_note
from .patch_hub import PatchHubRuntimeNode, register_operator as register_patch_hub
from .value_stepper import ValueStepperRuntimeNode, register_operator as register_value_stepper
from .data_expr import DataExprRuntimeNode, register_operator as register_data_expr
from .state_expr import StateExprRuntimeNode, register_operator as register_state_expr
from .viz_track import VizTrackRuntimeNode, register_operator as register_viz_track
from .viz_three_d import VizThreeDRuntimeNode, register_operator as register_viz_three_d
from .viz_tcode import VizTCodeRuntimeNode, register_operator as register_viz_tcode

__all__ = [
    "VizTextRuntimeNode",
    "VizWaveRuntimeNode",
    "VizVideoRuntimeNode",
    "VizAudioRuntimeNode",
    "ControlPanelRuntimeNode",
    "BackdropRuntimeNode",
    "NoteRuntimeNode",
    "PatchHubRuntimeNode",
    "ValueStepperRuntimeNode",
    "DataExprRuntimeNode",
    "StateExprRuntimeNode",
    "VizTrackRuntimeNode",
    "VizThreeDRuntimeNode",
    "VizTCodeRuntimeNode",
    "create_operator_registry",
    "register_operator",
]


class _VizConstructor(Protocol):
    def __call__(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        initial_state: dict[str, Any] | None = None,
        presentation: PresentationOutlet | None = None,
    ) -> StudioVizRuntimeNodeBase: ...


def _register_presentation_factories(registry: Registry, presentation: PresentationOutlet) -> None:
    def factory(constructor: _VizConstructor):
        def create(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> StudioVizRuntimeNodeBase:
            return constructor(node_id=node_id, node=node, initial_state=initial_state, presentation=presentation)

        return create

    constructors: tuple[tuple[str, _VizConstructor], ...] = (
        ("f8.viz.text", VizTextRuntimeNode),
        ("f8.viz.wave", VizWaveRuntimeNode),
        ("f8.viz.video", VizVideoRuntimeNode),
        ("f8.viz.audio", VizAudioRuntimeNode),
        ("f8.viz.track", VizTrackRuntimeNode),
        ("f8.viz.three_d", VizThreeDRuntimeNode),
        ("f8.viz.tcode", VizTCodeRuntimeNode),
    )
    for operator_class, constructor in constructors:
        registry.register_operator_factory(SERVICE_CLASS, operator_class, factory(constructor), overwrite=True)


def register_operator(registry: Registry, *, presentation: PresentationOutlet) -> Registry:
    """
    Register all Studio in-process operators.
    """
    reg = register_viz_text(registry)
    reg = register_viz_wave(reg)
    reg = register_viz_video(reg)
    reg = register_viz_audio(reg)
    reg = register_control_panel(reg)
    reg = register_backdrop(reg)
    reg = register_note(reg)
    reg = register_patch_hub(reg)
    reg = register_value_stepper(reg)
    reg = register_data_expr(reg)
    reg = register_state_expr(reg)
    reg = register_viz_track(reg)
    reg = register_viz_three_d(reg)
    reg = register_viz_tcode(reg)
    _register_presentation_factories(reg, presentation)
    return reg


def create_operator_registry(*, presentation: PresentationOutlet) -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_operator(Registry.wrap(runtime_registry), presentation=presentation)
    return runtime_registry
