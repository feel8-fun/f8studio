from __future__ import annotations

from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSpec,
    F8OperatorSchemaVersion,
    F8RuntimeNode,
    F8ServiceSpec,
    F8ServiceSchemaVersion,
    F8StateAccess,
    F8StateSpec,
    integer_schema,
    number_schema,
)
from f8pysdk.runtime import OperatorRuntimeNode, ServiceOperatorRuntimeRegistry

from .operators.signal_runtime import PrintRuntimeNode, SineRuntimeNode
from .operators.tick_runtime import TickRuntimeNode


def register_pyengine_runtimes(registry: ServiceOperatorRuntimeRegistry | None = None) -> ServiceOperatorRuntimeRegistry:
    """
    Register built-in f8.pyengine runtime implementations into the shared registry.
    """
    reg = registry or ServiceOperatorRuntimeRegistry.instance()

    def _tick_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorRuntimeNode:
        return TickRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    def _sine_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorRuntimeNode:
        return SineRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    def _print_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorRuntimeNode:
        return PrintRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register("f8.pyengine", "f8.tick", _tick_factory, overwrite=True)
    reg.register("f8.pyengine", "f8.sine", _sine_factory, overwrite=True)
    reg.register("f8.pyengine", "f8.print", _print_factory, overwrite=True)

    # Specs for discovery / `--describe`.
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass="f8.pyengine",
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

    reg.register_operator_spec(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pyengine",
            operatorClass="f8.tick",
            version="0.0.1",
            label="Tick",
            description="Source operator that generates periodic exec ticks.",
            tags=["execution", "timer", "start", "clock", "entrypoint"],
            stateFields=[
                F8StateSpec(
                    name="tickMs",
                    label="Tick (ms)",
                    description="Interval in milliseconds for emitting exec ticks.",
                    valueSchema=integer_schema(default=100, minimum=1, maximum=50000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
            ],
            execOutPorts=["exec"],
        ),
        overwrite=True,
    )

    reg.register_operator_spec(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pyengine",
            operatorClass="f8.sine",
            version="0.0.1",
            label="Sine",
            description="Exec-driven sine generator (pull-based output).",
            tags=["signal", "sin", "waveform", "generator", "oscillator"],
            execInPorts=["exec"],
            execOutPorts=["exec"],
            dataOutPorts=[F8DataPortSpec(name="value", description="sine output", valueSchema=number_schema())],
            stateFields=[
                F8StateSpec(
                    name="hz",
                    label="Hz",
                    description="Frequency in Hz.",
                    valueSchema=number_schema(default=1.0, minimum=0.0, maximum=100.0),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="amp",
                    label="Amp",
                    description="Amplitude.",
                    valueSchema=number_schema(default=1.0, minimum=0.0, maximum=1000.0),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
            ],
        ),
        overwrite=True,
    )

    reg.register_operator_spec(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass="f8.pyengine",
            operatorClass="f8.print",
            version="0.0.1",
            label="Print",
            description="Exec-driven printer (pulls `value` and prints).",
            tags=["debug", "console", "print"],
            execInPorts=["exec"],
            dataInPorts=[F8DataPortSpec(name="value", description="value to print", valueSchema=number_schema())],
        ),
        overwrite=True,
    )
    return reg
