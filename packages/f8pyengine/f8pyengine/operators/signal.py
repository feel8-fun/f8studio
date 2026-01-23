from __future__ import annotations

import math
import time
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    number_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS


SINE_OPERATOR_CLASS = "f8.sine"
TEMPEST_STROKE_OPERATOR_CLASS = "f8.tempest"


class SineRuntimeNode(RuntimeNode):
    """
    Exec-driven sine source (not a graph source): on exec, emits a numeric sample.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._exec_out_ports = list(getattr(node, "execOutPorts", None) or [])

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port) != "value":
            return None
        hz = await self.get_state("hz")
        if hz is None:
            hz = self._initial_state.get("hz", 1.0)
        amp = await self.get_state("amp")
        if amp is None:
            amp = self._initial_state.get("amp", 1.0)
        offset = await self.get_state("offset")
        if offset is None:
            offset = self._initial_state.get("offset", 0.0)
        phase = await self.get_state("phase")
        if phase is None:
            phase = self._initial_state.get("phase", 0.0)
        try:
            hz_f = float(hz)
        except Exception:
            hz_f = 1.0
        try:
            amp_f = float(amp)
        except Exception:
            amp_f = 1.0
        try:
            offset_f = float(offset)
        except Exception:
            offset_f = 0.0
        try:
            phase_f = float(phase)
        except Exception:
            phase_f = 0.0

        t = time.time()
        return offset_f + amp_f * math.sin(2.0 * math.pi * hz_f * t + (2.0 * math.pi * phase_f))


SineRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=SINE_OPERATOR_CLASS,
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
        F8StateSpec(
            name="offset",
            label="Offset",
            description="Vertical offset.",
            valueSchema=number_schema(default=0.0, minimum=-1000.0, maximum=1000.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="phase",
            label="Phase",
            description="Normalized phase (0.0 to 1.0).",
            valueSchema=number_schema(default=0.0, minimum=0, maximum=1),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _sine_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return SineRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, SINE_OPERATOR_CLASS, _sine_factory, overwrite=True)

    reg.register_operator_spec(SineRuntimeNode.SPEC, overwrite=True)
    return reg
