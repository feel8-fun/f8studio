from __future__ import annotations

from f8pysdk import (
    F8ServiceSpec,
    F8ServiceSchemaVersion,
)
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .constants import SERVICE_CLASS
from .operators.sequence import register_operator as register_sequence_operator
from .operators.signal import register_operators as register_signal_operators
from .operators.tick import register_operator as register_tick_operator


def register_pyengine_specs(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    """
    Register f8.pyengine service + operator specs and runtime factories.

    Specs live next to their runtime implementations (see `operators/*.py`).
    """
    reg = registry or RuntimeNodeRegistry.instance()

    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=SERVICE_CLASS,
            version="0.0.1",
            label="PyEngine",
            description="Python-based execution engine for Feel8 operators.",
            tags=["engine", "python", "py"],
            rendererClass="default_container",
            editableStateFields=False,
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableCommands=False,
        ),
        overwrite=True,
    )

    register_tick_operator(reg)
    register_sequence_operator(reg)
    register_signal_operators(reg)
    return reg


def register_pyengine_runtimes(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    # Backwards compatible name.
    return register_pyengine_specs(registry)
