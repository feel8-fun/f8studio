from __future__ import annotations

import math
import time
from typing import Any, Iterable

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    number_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from .envelope import DoubleExponentialMovingAverage, ExponentialMovingAverage

OPERATOR_CLASS = "f8.smooth_filter"

FILTER_NONE = "NONE"
FILTER_EMA = "EMA"
FILTER_DEMA = "DEMA"
FILTER_ONE_EURO = "ONEEURO"
FILTER_CHOICES = (FILTER_NONE, FILTER_EMA, FILTER_DEMA, FILTER_ONE_EURO)


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _coerce_sequence(value: Any) -> tuple[float, ...] | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return ()
        out: list[float] = []
        for item in value:
            num = _coerce_number(item)
            if num is None:
                return None
            out.append(float(num))
        return tuple(out)
    num = _coerce_number(value)
    if num is None:
        return None
    return (float(num),)


def _format_output(result: Iterable[float] | None) -> Any:
    if result is None:
        return None
    values = list(result)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


class OneEuroFilter:
    """Adaptive One Euro filter for real-time signal smoothing."""

    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
        default_frequency: float = 120.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.default_frequency = max(1e-6, float(default_frequency))
        self._value_estimate: float | None = None
        self._derivative_estimate: float = 0.0
        self._last_time: float | None = None
        self._value_filter = ExponentialMovingAverage(alpha=1.0)
        self._derivative_filter = ExponentialMovingAverage(alpha=1.0)

    def reset(self) -> None:
        self._value_estimate = None
        self._derivative_estimate = 0.0
        self._last_time = None
        self._value_filter.reset()
        self._derivative_filter.reset()

    def update(self, value: float, timestamp: float | None = None) -> float:
        value = float(value)

        if timestamp is None:
            dt = 1.0 / self.default_frequency
        elif self._last_time is None:
            dt = max(1e-6, 1.0 / self.default_frequency)
        else:
            dt = max(1e-6, float(timestamp) - self._last_time)

        if self._last_time is None:
            self._value_estimate = value
            self._derivative_estimate = 0.0
            self._last_time = float(timestamp) if timestamp is not None else None
            return value

        self._last_time = float(timestamp) if timestamp is not None else (self._last_time + dt)

        prev = self._value_estimate if self._value_estimate is not None else value
        derivative = (value - prev) / dt
        d_alpha = self._compute_alpha(self.derivative_cutoff, dt)
        self._derivative_estimate = self._derivative_filter.update(derivative, alpha=d_alpha)

        cutoff = self.min_cutoff + self.beta * abs(self._derivative_estimate)
        v_alpha = self._compute_alpha(cutoff, dt)
        self._value_estimate = self._value_filter.update(value, alpha=v_alpha)
        return float(self._value_estimate)

    @staticmethod
    def _compute_alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class SmoothFilterRuntimeNode(OperatorNode):
    """
    Per-tick smoothing filter for scalar or vector inputs.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

        self._filter_type = self._normalize_filter(self._initial_state.get("filter_type") or FILTER_EMA)
        self._ema_alpha = self._coerce_alpha(self._initial_state.get("ema_alpha"), 0.4)
        self._dema_alpha = self._coerce_alpha(self._initial_state.get("dema_alpha"), 0.4)
        self._one_euro_min_cutoff = max(
            1e-6, float(_coerce_number(self._initial_state.get("one_euro_min_cutoff")) or 1.5)
        )
        self._one_euro_beta = max(0.0, float(_coerce_number(self._initial_state.get("one_euro_beta")) or 0.0))
        self._one_euro_derivative_cutoff = max(
            1e-6, float(_coerce_number(self._initial_state.get("one_euro_derivative_cutoff")) or 1.0)
        )
        self._one_euro_default_freq = max(
            1e-3, float(_coerce_number(self._initial_state.get("one_euro_default_freq")) or 90.0)
        )
        self._filters: list[Any] = []
        self._last_output: tuple[float, ...] | None = None
        self._last_input: tuple[float, ...] | None = None
        self._last_ctx_id: str | int | None = None
        self._dirty = True

        self._initialize_filters()

    @staticmethod
    def _normalize_filter(value: Any) -> str:
        normalized = str(value or FILTER_NONE).strip().upper()
        if normalized in FILTER_CHOICES:
            return normalized
        return FILTER_NONE

    @staticmethod
    def _coerce_alpha(value: Any, default: float) -> float:
        numeric = _coerce_number(value)
        if numeric is None:
            return float(default)
        return max(0.0, min(1.0, float(numeric)))

    def _initialize_filters(self) -> None:
        self._filter_type = self._normalize_filter(self._filter_type)
        self._filters = []
        self._dirty = True

    def _reset_bank(self) -> None:
        self._filters = []
        self._dirty = True

    def _ensure_filter_bank(self, dimension: int) -> None:
        if self._filter_type == FILTER_NONE:
            self._filters = []
            return
        if len(self._filters) == dimension:
            return
        self._filters = []
        if self._filter_type == FILTER_EMA:
            self._filters = [ExponentialMovingAverage(alpha=self._ema_alpha) for _ in range(dimension)]
        elif self._filter_type == FILTER_DEMA:
            self._filters = [DoubleExponentialMovingAverage(alpha=self._dema_alpha) for _ in range(dimension)]
        elif self._filter_type == FILTER_ONE_EURO:
            default_freq = max(1e-3, self._one_euro_default_freq)
            self._filters = [
                OneEuroFilter(
                    min_cutoff=self._one_euro_min_cutoff,
                    beta=self._one_euro_beta,
                    derivative_cutoff=self._one_euro_derivative_cutoff,
                    default_frequency=default_freq,
                )
                for _ in range(dimension)
            ]

    def _resolve_timestamp(self) -> float:
        return time.monotonic()

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "")
        if name == "filter_type":
            self._filter_type = self._normalize_filter(value)
            self._reset_bank()
            return
        if name == "ema_alpha":
            self._ema_alpha = self._coerce_alpha(value, 0.4)
            self._reset_bank()
            return
        if name == "dema_alpha":
            self._dema_alpha = self._coerce_alpha(value, 0.4)
            self._reset_bank()
            return
        if name == "one_euro_min_cutoff":
            numeric = _coerce_number(value)
            if numeric is not None:
                self._one_euro_min_cutoff = max(1e-6, float(numeric))
            self._reset_bank()
            return
        if name == "one_euro_beta":
            numeric = _coerce_number(value)
            if numeric is not None:
                self._one_euro_beta = max(0.0, float(numeric))
            self._reset_bank()
            return
        if name == "one_euro_derivative_cutoff":
            numeric = _coerce_number(value)
            if numeric is not None:
                self._one_euro_derivative_cutoff = max(1e-6, float(numeric))
            self._reset_bank()
            return
        if name == "one_euro_default_freq":
            numeric = _coerce_number(value)
            if numeric is not None:
                self._one_euro_default_freq = max(1e-3, float(numeric))
            self._reset_bank()

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port) != "value":
            return None

        sample = _coerce_sequence(await self.pull("value", ctx_id=ctx_id))
        if sample is None:
            return _format_output(self._last_output)

        if not self._dirty:
            if ctx_id is not None and ctx_id == self._last_ctx_id and sample == (self._last_input or ()):
                return _format_output(self._last_output)
            if ctx_id is None and sample == (self._last_input or ()):
                return _format_output(self._last_output)

        if self._filter_type == FILTER_NONE:
            self._last_output = tuple(sample)
            self._last_input = tuple(sample)
            self._last_ctx_id = ctx_id
            self._dirty = False
            return _format_output(self._last_output)

        self._ensure_filter_bank(len(sample))
        timestamp = self._resolve_timestamp()
        results: list[float] = []

        if self._filter_type in {FILTER_EMA, FILTER_DEMA}:
            alpha_value = self._ema_alpha if self._filter_type == FILTER_EMA else self._dema_alpha
            for index, value in enumerate(sample):
                filt = self._filters[index]
                results.append(float(filt.update(value, alpha=alpha_value)))
        elif self._filter_type == FILTER_ONE_EURO:
            for index, value in enumerate(sample):
                filt = self._filters[index]
                results.append(float(filt.update(value, timestamp)))
        else:
            results = list(sample)

        self._last_output = tuple(results)
        self._last_input = tuple(sample)
        self._last_ctx_id = ctx_id
        self._dirty = False
        return _format_output(self._last_output)


SmoothFilterRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Smooth Filter",
    description="Smooths scalar or vector inputs with EMA/DEMA/One Euro filtering.",
    tags=["filter", "smoothing", "one_euro", "signal"],
    dataInPorts=[F8DataPortSpec(name="value", description="Value to filter.", valueSchema=any_schema())],
    dataOutPorts=[F8DataPortSpec(name="value", description="Filtered output.", valueSchema=any_schema())],
    stateFields=[
        F8StateSpec(
            name="filter_type",
            label="Filter",
            description="Filter type.",
            valueSchema=string_schema(default="EMA", enum=list(FILTER_CHOICES)),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="ema_alpha",
            label="EMA Alpha",
            description="EMA smoothing factor (0..1).",
            valueSchema=number_schema(default=0.4, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="dema_alpha",
            label="DEMA Alpha",
            description="DEMA smoothing factor (0..1).",
            valueSchema=number_schema(default=0.4, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="one_euro_min_cutoff",
            label="One Euro Min Cutoff",
            description="Minimum cutoff frequency.",
            valueSchema=number_schema(default=1.5, minimum=0.01, maximum=10.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="one_euro_beta",
            label="One Euro Beta",
            description="Speed coefficient for dynamic cutoff.",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=5.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="one_euro_derivative_cutoff",
            label="One Euro Derivative Cutoff",
            description="Cutoff frequency for the derivative filter.",
            valueSchema=number_schema(default=1.0, minimum=0.01, maximum=10.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="one_euro_default_freq",
            label="One Euro Default Freq",
            description="Default sampling frequency (Hz).",
            valueSchema=number_schema(default=90.0, minimum=1.0, maximum=240.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return SmoothFilterRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(SmoothFilterRuntimeNode.SPEC, overwrite=True)
    return reg
