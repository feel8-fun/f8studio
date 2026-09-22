from typing import Any

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
    "create_operator_registry",
    "register_operator",
]


def _register_presentation_factories(registry: Registry, presentation: PresentationOutlet) -> None:
    def text_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> VizTextRuntimeNode:
        created = VizTextRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)
        created.presentation = presentation
        return created

    def wave_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> VizWaveRuntimeNode:
        created = VizWaveRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)
        created.presentation = presentation
        return created

    def video_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> VizVideoRuntimeNode:
        created = VizVideoRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)
        created.presentation = presentation
        return created

    def audio_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> VizAudioRuntimeNode:
        created = VizAudioRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)
        created.presentation = presentation
        return created

    def track_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> VizTrackRuntimeNode:
        created = VizTrackRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)
        created.presentation = presentation
        return created

    def three_d_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> VizThreeDRuntimeNode:
        created = VizThreeDRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)
        created.presentation = presentation
        return created

    registry.register_operator_factory(SERVICE_CLASS, "f8.viz.text", text_factory, overwrite=True)
    registry.register_operator_factory(SERVICE_CLASS, "f8.viz.wave", wave_factory, overwrite=True)
    registry.register_operator_factory(SERVICE_CLASS, "f8.viz.video", video_factory, overwrite=True)
    registry.register_operator_factory(SERVICE_CLASS, "f8.viz.audio", audio_factory, overwrite=True)
    registry.register_operator_factory(SERVICE_CLASS, "f8.viz.track", track_factory, overwrite=True)
    registry.register_operator_factory(SERVICE_CLASS, "f8.viz.three_d", three_d_factory, overwrite=True)


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
    _register_presentation_factories(reg, presentation)
    return reg


def create_operator_registry(*, presentation: PresentationOutlet) -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_operator(Registry.wrap(runtime_registry), presentation=presentation)
    return runtime_registry
