from __future__ import annotations

from typing import Any

from f8pysdk.codec import parse_bool, parse_number_sequence
from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    number_schema,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ._signal_processing import (
    ExponentialTrendTracker,
    NodeComputationCache,
    clamp_alpha,
    format_output,
)

OPERATOR_CLASS = "f8.detrend"
MODE_CONSTANT = "CONSTANT"
MODE_LINEAR = "LINEAR"
MODE_CHOICES = (MODE_CONSTANT, MODE_LINEAR)


def _normalize_mode(value: Any) -> str:
    normalized = str(value or MODE_CONSTANT).strip().upper()
    if normalized in MODE_CHOICES:
        return normalized
    return MODE_CONSTANT


class DetrendRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._mode = _normalize_mode(self._initial_state.get("mode"))
        self._alpha = clamp_alpha(self._initial_state.get("alpha"), default=0.05)
        self._reset_on_state_change = parse_bool(self._initial_state.get("reset_on_state_change"))
        if self._reset_on_state_change is None:
            self._reset_on_state_change = True
        self._trackers: list[ExponentialTrendTracker] = []
        self._cache = NodeComputationCache()

    def _reset_trackers(self) -> None:
        self._trackers = []
        self._cache.mark_dirty()

    def _ensure_trackers(self, dimension: int) -> None:
        if len(self._trackers) == dimension:
            return
        self._trackers = [ExponentialTrendTracker(alpha=self._alpha) for _ in range(dimension)]
        self._cache.mark_dirty()

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "")
        if name == "mode":
            self._mode = _normalize_mode(value)
            if self._reset_on_state_change:
                self._reset_trackers()
            else:
                self._cache.mark_dirty()
            return
        if name == "alpha":
            self._alpha = clamp_alpha(value, default=0.05)
            if self._reset_on_state_change:
                self._reset_trackers()
            else:
                for tracker in self._trackers:
                    tracker.alpha = self._alpha
                self._cache.mark_dirty()
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

        self._ensure_trackers(len(sample))
        results: list[float] = []
        if self._mode == MODE_LINEAR:
            for index, value in enumerate(sample):
                results.append(self._trackers[index].update_linear(value, alpha=self._alpha))
        else:
            for index, value in enumerate(sample):
                results.append(self._trackers[index].update_constant(value, alpha=self._alpha))

        output = tuple(results)
        self._cache.update(sample=sample, output=output, ctx_id=ctx_id)
        return format_output(output)


DetrendRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.signal",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Detrend",
    description="Removes slow baseline or linear trend from scalar or vector inputs.",
    tags=["signal", "detrend", "filter"],
    dataInPorts=[F8DataPortSpec(name="value", description="Value to detrend.", valueSchema=any_schema())],
    dataOutPorts=[F8DataPortSpec(name="value", description="Detrended output.", valueSchema=any_schema())],
    stateFields=[
        F8StateSpec(
            name="mode",
            label="Mode",
            description="Detrend mode.",
            valueSchema=string_schema(default=MODE_CONSTANT, enum=list(MODE_CHOICES)),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="alpha",
            label="Alpha",
            description="Trend tracking smoothing factor.",
            valueSchema=number_schema(default=0.05, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.slider),
            showOnNode=True,
        ),
        F8StateSpec(
            name="reset_on_state_change",
            label="Reset On State Change",
            description="Reset tracker history when parameters change.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(DetrendRuntimeNode.SPEC, DetrendRuntimeNode, overwrite=True)
    return registry
