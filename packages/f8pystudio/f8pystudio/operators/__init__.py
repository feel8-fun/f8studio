from .monitor_state import MonitorStateRuntimeNode, register_operator as register_monitor_state
from .print import PyStudioPrintRuntimeNode, register_operator as register_print, set_preview_sink
from .timeseries import PyStudioTimeSeriesRuntimeNode, register_operator as register_timeseries

__all__ = [
    "PyStudioPrintRuntimeNode",
    "MonitorStateRuntimeNode",
    "PyStudioTimeSeriesRuntimeNode",
    "register_operator",
    "set_preview_sink",
]


def register_operator(registry=None):
    """
    Register all Studio in-process operators.
    """
    reg = register_print(registry)
    reg = register_timeseries(reg)
    reg = register_monitor_state(reg)
    return reg
