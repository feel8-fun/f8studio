from __future__ import annotations

from typing import Any

from f8pysdk.codec import parse_bool, parse_number_sequence
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    number_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ._signal_processing import (
    NodeComputationCache,
    SosFilterBank,
    clamp_order,
    clamp_positive,
    design_bandpass,
    format_output,
    sampling_hz_from_interval_ms,
)

OPERATOR_CLASS = "f8.bandpass_filter"


class BandpassFilterRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._sample_interval_ms = clamp_positive(
            self._initial_state.get("sampleIntervalMs"), default=1000.0 / 120.0, minimum=1e-6
        )
        self._low_cutoff = clamp_positive(self._initial_state.get("low_cutoff"), default=1.0, minimum=1e-6)
        self._high_cutoff = clamp_positive(self._initial_state.get("high_cutoff"), default=8.0, minimum=1e-6)
        self._order = clamp_order(self._initial_state.get("order"), default=2)
        self._reset_on_state_change = parse_bool(self._initial_state.get("reset_on_state_change"))
        if self._reset_on_state_change is None:
            self._reset_on_state_change = True
        self._sos = design_bandpass(
            sampling_hz=sampling_hz_from_interval_ms(self._sample_interval_ms, default_interval_ms=1000.0 / 120.0),
            low_cutoff=self._low_cutoff,
            high_cutoff=self._high_cutoff,
            order=self._order,
        )
        self._bank: SosFilterBank | None = None
        self._cache = NodeComputationCache()

    def _rebuild_filter(self, *, reset_state: bool) -> None:
        if self._high_cutoff <= self._low_cutoff:
            self._cache.mark_dirty()
            return
        self._sos = design_bandpass(
            sampling_hz=sampling_hz_from_interval_ms(self._sample_interval_ms, default_interval_ms=1000.0 / 120.0),
            low_cutoff=self._low_cutoff,
            high_cutoff=self._high_cutoff,
            order=self._order,
        )
        if reset_state or self._bank is None:
            self._bank = None
        elif self._cache.last_input is not None:
            self._bank = SosFilterBank.create(sos=self._sos, dimension=len(self._cache.last_input))
        self._cache.mark_dirty()

    def _ensure_bank(self, dimension: int) -> None:
        if self._bank is not None and len(self._bank.zi) == dimension:
            return
        self._bank = SosFilterBank.create(sos=self._sos, dimension=dimension)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "")
        if name == "sampleIntervalMs":
            self._sample_interval_ms = clamp_positive(value, default=self._sample_interval_ms, minimum=1e-6)
            self._rebuild_filter(reset_state=self._reset_on_state_change)
            return
        if name == "low_cutoff":
            self._low_cutoff = clamp_positive(value, default=self._low_cutoff, minimum=1e-6)
            self._rebuild_filter(reset_state=self._reset_on_state_change)
            return
        if name == "high_cutoff":
            self._high_cutoff = clamp_positive(value, default=self._high_cutoff, minimum=1e-6)
            self._rebuild_filter(reset_state=self._reset_on_state_change)
            return
        if name == "order":
            self._order = clamp_order(value, default=self._order)
            self._rebuild_filter(reset_state=self._reset_on_state_change)
            return
        if name == "reset_on_state_change":
            reset_value = parse_bool(value)
            if reset_value is not None:
                self._reset_on_state_change = reset_value

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port) != "value":
            return None

        sample = parse_number_sequence(await self.pull("value", ctx_id=ctx_id))
        if sample is None:
            return format_output(self._cache.last_output)
        if self._cache.should_reuse(sample, ctx_id):
            return format_output(self._cache.last_output)
        if self._high_cutoff <= self._low_cutoff:
            return format_output(self._cache.last_output)

        self._ensure_bank(len(sample))
        assert self._bank is not None
        output = self._bank.update(sample)
        self._cache.update(sample=sample, output=output, ctx_id=ctx_id)
        return format_output(output)


BandpassFilterRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.signal",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Bandpass Filter",
    description="Butterworth IIR band-pass filter for scalar or vector inputs.",
    tags=["signal", "filter", "bandpass", "butterworth"],
    dataInPorts=[F8DataPortSpec(name="value", description="Value to filter.", valueSchema=any_schema())],
    dataOutPorts=[F8DataPortSpec(name="value", description="Filtered output.", valueSchema=any_schema())],
    stateFields=[
        F8StateSpec(
            name="sampleIntervalMs",
            label="Sample Interval (ms)",
            description="Sampling interval in milliseconds.",
            valueSchema=number_schema(default=1000.0 / 120.0, minimum=0.001, maximum=50000.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="low_cutoff",
            label="Low Cutoff",
            description="Lower band edge in Hz.",
            valueSchema=number_schema(default=1.0, minimum=0.001, maximum=5000.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="high_cutoff",
            label="High Cutoff",
            description="Upper band edge in Hz.",
            valueSchema=number_schema(default=8.0, minimum=0.001, maximum=5000.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="order",
            label="Order",
            description="Butterworth filter order.",
            valueSchema=number_schema(default=2, minimum=1.0, maximum=12.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="reset_on_state_change",
            label="Reset On State Change",
            description="Reset filter history when parameters change.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(BandpassFilterRuntimeNode.SPEC, BandpassFilterRuntimeNode, overwrite=True)
    return registry
