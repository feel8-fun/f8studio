from .print import PyStudioPrintRuntimeNode, register_operator as register_print
from .trackviz import PyStudioTrackVizRuntimeNode, register_operator as register_trackviz
from .timeseries import PyStudioTimeSeriesRuntimeNode, register_operator as register_timeseries
from .video_shm_view import PyStudioVideoShmViewRuntimeNode, register_operator as register_video_shm_view
from .audio_shm_view import PyStudioAudioShmViewRuntimeNode, register_operator as register_audio_shm_view
from .skeleton3d import PyStudioSkeleton3DRuntimeNode, register_operator as register_skeleton3d
from .tcode_viewer import PyStudioTCodeViewerRuntimeNode, register_operator as register_tcode_viewer

__all__ = [
    "PyStudioPrintRuntimeNode",
    "PyStudioTrackVizRuntimeNode",
    "PyStudioTimeSeriesRuntimeNode",
    "PyStudioVideoShmViewRuntimeNode",
    "PyStudioAudioShmViewRuntimeNode",
    "PyStudioSkeleton3DRuntimeNode",
    "PyStudioTCodeViewerRuntimeNode",
    "register_operator",
]


def register_operator(registry=None):
    """
    Register all Studio in-process operators.
    """
    reg = register_print(registry)
    reg = register_timeseries(reg)
    reg = register_trackviz(reg)
    reg = register_video_shm_view(reg)
    reg = register_audio_shm_view(reg)
    reg = register_skeleton3d(reg)
    reg = register_tcode_viewer(reg)
    return reg
