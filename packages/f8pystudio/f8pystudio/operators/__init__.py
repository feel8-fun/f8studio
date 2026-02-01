from .monitor_state import MonitorStateRuntimeNode, register_operator as register_monitor_state
from .print import PyStudioPrintRuntimeNode, register_operator as register_print
from .trackviz import PyStudioTrackVizRuntimeNode, register_operator as register_trackviz
from .timeseries import PyStudioTimeSeriesRuntimeNode, register_operator as register_timeseries
from .video_shm_view import PyStudioVideoShmViewRuntimeNode, register_operator as register_video_shm_view
from .audio_shm_view import PyStudioAudioShmViewRuntimeNode, register_operator as register_audio_shm_view

__all__ = [
    "PyStudioPrintRuntimeNode",
    "MonitorStateRuntimeNode",
    "PyStudioTrackVizRuntimeNode",
    "PyStudioTimeSeriesRuntimeNode",
    "PyStudioVideoShmViewRuntimeNode",
    "PyStudioAudioShmViewRuntimeNode",
    "register_operator",
]


def register_operator(registry=None):
    """
    Register all Studio in-process operators.
    """
    reg = register_print(registry)
    reg = register_timeseries(reg)
    reg = register_monitor_state(reg)
    reg = register_trackviz(reg)
    reg = register_video_shm_view(reg)
    reg = register_audio_shm_view(reg)
    return reg
