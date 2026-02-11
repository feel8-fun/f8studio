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
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS


COSINE_OPERATOR_CLASS = "f8.cosine"
TEMPEST_OPERATOR_CLASS = "f8.tempest"
PHASE_OPERATOR_CLASS = "f8.phase"

_TWO_PI = 2.0 * math.pi


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f):
        return None
    return f


class _PhaseAccumulator:
    """
    Monotonic-time phase accumulator (normalized 0..1).

    Used as a fallback when no external phase input is provided.
    """

    def __init__(self, *, initial_phase: float = 0.0, last_time_s: float | None = None) -> None:
        self._turns = float(initial_phase) % 1.0
        self._last_time_s: float | None = last_time_s

    def set_phase(self, phase: float) -> None:
        # Preserve integer turns; only override the fractional part.
        frac = float(phase) % 1.0
        self._turns = math.floor(float(self._turns)) + frac

    def set_turns(self, turns: float) -> None:
        self._turns = float(turns)

    def phase(self) -> float:
        return float(float(self._turns) % 1.0)

    def turns(self) -> float:
        return float(self._turns)

    def reset_time_base(self) -> None:
        self._last_time_s = None

    def step(self, *, hz: float) -> float:
        hz_f = float(hz)
        if math.isnan(hz_f) or hz_f < 0.0:
            hz_f = 0.0

        now_s = time.monotonic()
        if self._last_time_s is None:
            # First sample after (re)creation/attach: don't advance phase yet.
            delta_s = 0.0
        else:
            delta_s = max(0.0, now_s - self._last_time_s)
        self._last_time_s = now_s

        self._turns = float(self._turns + hz_f * delta_s)
        return float(self._turns)


class PhaseRuntimeNode(OperatorNode):
    """
    Phase source: advances and emits a normalized phase (0..1) plus unwrapped turns.

    This allows multiple oscillators (sine/tempest/etc.) to share one phase input.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

        init_phase = _coerce_number(self._initial_state.get("phase", 0.0))
        init_last = self._initial_state.get("__lastTimeS", None)
        self._acc = _PhaseAccumulator(
            initial_phase=float(init_phase if init_phase is not None else 0.0),
            last_time_s=_float_or(init_last, 0.0) if init_last is not None else None,
        )

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        # Freeze/resume without phase jump across deactivate/activate.
        self._acc.reset_time_base()

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        p = str(port)
        if p not in ("phase", "phaseTurns"):
            return None

        in_hz = _coerce_number(await self.pull("hz", ctx_id=ctx_id))
        in_phase = _coerce_number(await self.pull("phase", ctx_id=ctx_id))
        in_reset = await self.pull("reset", ctx_id=ctx_id)

        hz = in_hz
        if hz is None:
            hz = await self.get_state_value("hz")
            if hz is None:
                hz = self._initial_state.get("hz", 1.0)

        hz_f = _coerce_number(hz)
        if hz_f is None:
            hz_f = 1.0
        hz_f = max(0.0, hz_f)

        if bool(in_reset):
            self._acc.set_turns(0.0)
        if in_phase is not None:
            self._acc.set_phase(in_phase)
        turns = self._acc.step(hz=hz_f)

        if p == "phase":
            return float(float(turns) % 1.0)
        return float(turns)


PhaseRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=PHASE_OPERATOR_CLASS,
    version="0.0.1",
    label="Phase",
    description="Phase accumulator. Outputs normalized phase (0..1) and unwrapped phase turns.",
    tags=["signal", "phase", "waveform", "generator", "oscillator"],
    dataInPorts=[
        F8DataPortSpec(name="hz", description="Frequency override (Hz).", valueSchema=number_schema(), required=False),
        F8DataPortSpec(
            name="phase", description="Absolute phase override (0..1).", valueSchema=number_schema(), required=False
        ),
        F8DataPortSpec(
            name="reset", description="If true, reset phase to 0.", valueSchema={"type": "boolean"}, required=False
        ),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="phase", description="Normalized phase (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="phaseTurns", description="Unwrapped phase turns (cycles).", valueSchema=number_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="hz",
            label="Hz",
            description="Frequency in Hz.",
            valueSchema=number_schema(default=1.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
    ],
)


class CosineRuntimeNode(OperatorNode):
    """
    Sine phase transform: emits a numeric sample.

    This node does not generate phase internally. Provide `phase` (0..1).
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port) != "value":
            return None

        in_amp = _coerce_number(await self.pull("amp", ctx_id=ctx_id))
        in_dc = _coerce_number(await self.pull("dc", ctx_id=ctx_id))
        in_phase_in = _coerce_number(await self.pull("phase", ctx_id=ctx_id))
        in_phase = _coerce_number(await self.pull("phaseOffset", ctx_id=ctx_id))

        amp = in_amp
        if amp is None:
            amp = await self.get_state_value("amp")
            if amp is None:
                amp = self._initial_state.get("amp", 0.5)

        dc = in_dc
        if dc is None:
            dc = await self.get_state_value("dc")
            if dc is None:
                dc = self._initial_state.get("dc", 0.5)

        phase = in_phase
        if phase is None:
            phase = await self.get_state_value("phaseOffset")
            if phase is None:
                phase = self._initial_state.get("phaseOffset", 0.0)

        amp_f = _coerce_number(amp)
        if amp_f is None:
            amp_f = 0.5

        dc_f = _coerce_number(dc)
        if dc_f is None:
            dc_f = 0.5

        phase_offset_f = _coerce_number(phase)
        if phase_offset_f is None:
            phase_offset_f = 0.0

        phase_in_f = _coerce_number(in_phase_in)
        if phase_in_f is None:
            phase_in_f = 0.0
        else:
            phase_in_f = float(phase_in_f) % 1.0

        return dc_f + amp_f * math.cos(_TWO_PI * (phase_in_f + phase_offset_f))


CosineRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=COSINE_OPERATOR_CLASS,
    version="0.0.1",
    label="Cosine",
    description="Cosine phase transform. Provide `phase` (0..1) from an upstream phase driver (e.g. Phase node).",
    tags=["signal", "cos", "waveform", "generator", "oscillator"],
    dataInPorts=[
        F8DataPortSpec(
            name="phase",
            description="Phase input (0..1).",
            valueSchema=number_schema(),
            required=True,
            showOnNode=True,
        ),
        F8DataPortSpec(
            name="amp",
            description="Amplitude override.",
            valueSchema=number_schema(default=0.5),
            required=False,
            showOnNode=False,
        ),
        F8DataPortSpec(
            name="dc",
            description="DC offset override.",
            valueSchema=number_schema(default=0.5),
            required=False,
            showOnNode=False,
        ),
        F8DataPortSpec(
            name="phaseOffset",
            description="Phase offset override (0..1).",
            valueSchema=number_schema(),
            required=False,
            showOnNode=False,
        ),
    ],
    dataOutPorts=[F8DataPortSpec(name="value", description="cosine output", valueSchema=number_schema())],
    stateFields=[
        F8StateSpec(
            name="dc",
            label="DC",
            description="Default DC offset (used when `dc` input is not provided).",
            valueSchema=number_schema(default=0.5),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="amp",
            label="Amp",
            description="Amplitude.",
            valueSchema=number_schema(default=0.5),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="phaseOffset",
            label="Phase Offset",
            description="Normalized phase offset (0.0 to 1.0).",
            valueSchema=number_schema(default=0.0, minimum=0, maximum=1),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
    ],
)


class TempestRuntimeNode(OperatorNode):
    """
    Tempest phase transform: emits a numeric sample.

    Ported from `f8flow/web/.../nodes/tempest.ts`.

    This node does not generate phase internally. Provide `phase` (0..1).
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port) != "out":
            return None

        in_amplitude = _coerce_number(await self.pull("amp", ctx_id=ctx_id))
        in_phase_in = _coerce_number(await self.pull("phase", ctx_id=ctx_id))
        in_phase_offset = _coerce_number(await self.pull("phaseOffset", ctx_id=ctx_id))
        in_eccentricity = _coerce_number(await self.pull("eccentric", ctx_id=ctx_id))
        in_dc_offset = _coerce_number(await self.pull("dc", ctx_id=ctx_id))

        amplitude = in_amplitude
        if amplitude is None:
            amplitude = await self.get_state_value("amp")
            if amplitude is None:
                amplitude = self._initial_state.get("amp", 1.0)
        phase_offset = in_phase_offset
        if phase_offset is None:
            phase_offset = await self.get_state_value("phaseOffset")
            if phase_offset is None:
                phase_offset = self._initial_state.get("phaseOffset", 0.0)
        eccentricity = in_eccentricity
        if eccentricity is None:
            eccentricity = await self.get_state_value("eccentric")
            if eccentricity is None:
                eccentricity = self._initial_state.get("eccentric", 0.0)
        dc_offset = in_dc_offset
        if dc_offset is None:
            dc_offset = await self.get_state_value("dc")
            if dc_offset is None:
                dc_offset = self._initial_state.get("dc", 0.0)

        amplitude_f = _coerce_number(amplitude)
        if amplitude_f is None:
            amplitude_f = 1.0
        phase_offset_f = _coerce_number(phase_offset)
        if phase_offset_f is None:
            phase_offset_f = 0.0
        eccentricity_f = _coerce_number(eccentricity)
        if eccentricity_f is None:
            eccentricity_f = 0.0
        dc_offset_f = _coerce_number(dc_offset)
        if dc_offset_f is None:
            dc_offset_f = 0.0

        phase_in_f = _coerce_number(in_phase_in)
        if phase_in_f is None:
            phase_in_f = 0.0
        else:
            phase_in_f = float(phase_in_f) % 1.0

        theta_term = _TWO_PI * (phase_in_f + phase_offset_f)
        value = amplitude_f * math.cos(theta_term + (eccentricity_f * math.sin(theta_term))) + dc_offset_f

        return float(value)


TempestRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=TEMPEST_OPERATOR_CLASS,
    version="0.0.1",
    label="Tempest",
    description="Tempest phase transform (phase-modulated cosine). Provide `phase` (0..1) from an upstream phase driver (e.g. Phase node).",
    tags=["signal", "waveform", "generator", "oscillator", "tempest"],
    dataInPorts=[
        F8DataPortSpec(
            name="phase",
            description="Phase input (0..1).",
            valueSchema=number_schema(),
            required=True,
            showOnNode=True,
        ),
        F8DataPortSpec(
            name="amp",
            description="Amplitude override.",
            valueSchema=number_schema(),
            required=False,
            showOnNode=False,
        ),
        F8DataPortSpec(
            name="phaseOffset",
            description="Phase offset override (0..1).",
            valueSchema=number_schema(),
            required=False,
            showOnNode=False,
        ),
        F8DataPortSpec(
            name="eccentric",
            description="Eccentricity override",
            valueSchema=number_schema(),
            required=False,
            showOnNode=False,
        ),
        F8DataPortSpec(
            name="dc",
            description="DC offset override.",
            valueSchema=number_schema(),
            required=False,
            showOnNode=False,
        ),
    ],
    dataOutPorts=[F8DataPortSpec(name="out", description="tempest output", valueSchema=number_schema())],
    stateFields=[
        F8StateSpec(
            name="dc",
            label="DC",
            description="Default DC offset (used when `dc` input is not provided).",
            valueSchema=number_schema(default=0.5),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="amp",
            label="Amp",
            description="Default amplitude (used when `amp`/`amplitude` input is not provided).",
            valueSchema=number_schema(default=0.5),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="phaseOffset",
            label="Phase Offset",
            description="Fraction of a full cycle added to the phase (0..1).",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="eccentric",
            label="Eccentric",
            description="Controls curvature of the inner sine.",
            valueSchema=number_schema(default=0.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _cosine_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return CosineRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    def _tempest_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return TempestRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    def _phase_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return PhaseRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, COSINE_OPERATOR_CLASS, _cosine_factory, overwrite=True)
    reg.register(SERVICE_CLASS, TEMPEST_OPERATOR_CLASS, _tempest_factory, overwrite=True)
    reg.register(SERVICE_CLASS, PHASE_OPERATOR_CLASS, _phase_factory, overwrite=True)

    reg.register_operator_spec(CosineRuntimeNode.SPEC, overwrite=True)
    reg.register_operator_spec(TempestRuntimeNode.SPEC, overwrite=True)
    reg.register_operator_spec(PhaseRuntimeNode.SPEC, overwrite=True)
    return reg
