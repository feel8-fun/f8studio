from __future__ import annotations

import math
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    number_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.envelope"

_METHODS = ("EMA", "DEMA", "SMA")


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


def _normalize_method(value: Any, *, default: str = "EMA") -> str:
    method = str(value or "").strip().upper()
    if method in _METHODS:
        return method
    return default


class ExponentialMovingAverage:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = float(alpha)
        self.ema_pt: float | None = None

    def update(self, pt: float, *, alpha: float | None = None) -> float:
        active_alpha = self.alpha if alpha is None else float(alpha)
        value = float(pt)
        if self.ema_pt is None:
            self.ema_pt = value
        else:
            self.ema_pt = active_alpha * value + (1.0 - active_alpha) * self.ema_pt
        return float(self.ema_pt)

    def reset(self) -> None:
        self.ema_pt = None


class DoubleExponentialMovingAverage:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = float(alpha)
        self.ema_pt: float | None = None
        self.ema2_pt: float | None = None

    def update(self, pt: float, *, alpha: float | None = None) -> float:
        active_alpha = self.alpha if alpha is None else float(alpha)
        value = float(pt)
        if self.ema_pt is None:
            self.ema_pt = value
            self.ema2_pt = value
        else:
            self.ema_pt = active_alpha * value + (1.0 - active_alpha) * self.ema_pt
            self.ema2_pt = active_alpha * self.ema_pt + (1.0 - active_alpha) * self.ema2_pt
        return float(2.0 * self.ema_pt - self.ema2_pt)

    def reset(self) -> None:
        self.ema_pt = None
        self.ema2_pt = None


class SimpleMovingAverage:
    def __init__(self, window: int = 5) -> None:
        self.window = max(1, int(window))
        self._values: list[float] = []

    def update(self, value: float, *, alpha: float | None = None) -> float:
        self._values.append(float(value))
        if len(self._values) > self.window:
            self._values = self._values[-self.window :]
        return float(sum(self._values) / len(self._values))

    def reset(self) -> None:
        self._values = []

    def set_window(self, window: int) -> None:
        window = max(1, int(window))
        if window == self.window:
            return
        self.window = window
        if len(self._values) > self.window:
            self._values = self._values[-self.window :]


class EnvelopeTracker:
    """Adaptive upper and lower envelope tracker with configurable smoothing."""

    def __init__(
        self,
        *,
        method: str = "DEMA",
        rise_alpha: float = 0.3,
        fall_alpha: float = 0.03,
        min_span: float = 8.0,
        sma_window: int = 8,
    ) -> None:
        self.rise_alpha = float(rise_alpha)
        self.fall_alpha = float(fall_alpha)
        self.min_span = float(min_span)
        self.sma_window = max(1, int(sma_window))
        self.upper_filter = None
        self.lower_filter = None
        self.upper: float | None = None
        self.lower: float | None = None
        self.method = ""
        self.set_method(method)

    def set_method(self, method: str, *, sma_window: int | None = None) -> None:
        normalized = _normalize_method(method, default="DEMA")
        if normalized == self.method and self.upper_filter is not None:
            return
        if sma_window is not None:
            self.sma_window = max(1, int(sma_window))
        self.method = normalized
        self.upper_filter = self._create_filter()
        self.lower_filter = self._create_filter()
        self.reset()

    def set_parameters(
        self,
        *,
        rise_alpha: float | None = None,
        fall_alpha: float | None = None,
        min_span: float | None = None,
        sma_window: int | None = None,
        method: str | None = None,
    ) -> None:
        if rise_alpha is not None:
            self.rise_alpha = float(rise_alpha)
        if fall_alpha is not None:
            self.fall_alpha = float(fall_alpha)
        if min_span is not None:
            self.min_span = max(0.0, float(min_span))
        if method is not None or sma_window is not None:
            self.set_method(method or self.method, sma_window=sma_window)
        elif sma_window is not None:
            self.sma_window = max(1, int(sma_window))
        if self.method == "SMA" and sma_window is not None:
            self._configure_sma_filters()

    def reset(self) -> None:
        if self.upper_filter is not None:
            self.upper_filter.reset()
        if self.lower_filter is not None:
            self.lower_filter.reset()
        self.upper = None
        self.lower = None

    def update(self, value: float) -> float:
        if self.upper_filter is None or self.lower_filter is None:
            self.upper_filter = self._create_filter()
            self.lower_filter = self._create_filter()

        if self.upper is None or self.lower is None:
            self._reset_filters()
            self.upper = self._update_filter(self.upper_filter, value, self.rise_alpha)
            self.lower = self._update_filter(self.lower_filter, value, self.rise_alpha)
            return 0.5

        upper_alpha = self.rise_alpha if value >= self.upper else self.fall_alpha
        lower_alpha = self.rise_alpha if value <= self.lower else self.fall_alpha

        self.upper = self._update_filter(self.upper_filter, value, upper_alpha)
        self.lower = self._update_filter(self.lower_filter, value, lower_alpha)

        if self.upper - self.lower < self.min_span:
            midpoint = 0.5 * (self.upper + self.lower)
            half_span = self.min_span / 2.0
            self.upper = midpoint + half_span
            self.lower = midpoint - half_span

        span = self.upper - self.lower
        normalized = (value - self.lower) / span if span > 0 else 0.5
        return float(max(0.0, min(1.0, normalized)))

    def _reset_filters(self) -> None:
        if self.upper_filter is not None:
            self.upper_filter.reset()
        if self.lower_filter is not None:
            self.lower_filter.reset()
        self.upper = None
        self.lower = None

    def _create_filter(self):
        if self.method == "DEMA":
            return DoubleExponentialMovingAverage(alpha=self.rise_alpha)
        if self.method == "EMA":
            return ExponentialMovingAverage(alpha=self.rise_alpha)
        if self.method == "SMA":
            return SimpleMovingAverage(window=self.sma_window)
        raise ValueError(f"Unsupported method {self.method}")

    def _configure_sma_filters(self) -> None:
        if self.method != "SMA":
            return
        for filt in (self.upper_filter, self.lower_filter):
            if filt is None:
                continue
            try:
                filt.set_window(self.sma_window)
            except Exception:
                continue

    def _update_filter(self, filt, value: float, alpha: float) -> float:
        if self.method == "SMA":
            try:
                filt.set_window(self.sma_window)
            except Exception:
                pass
            return float(filt.update(value))
        return float(filt.update(value, alpha=alpha))


class EnvelopeRuntimeNode(OperatorNode):
    """
    Tracks upper/lower envelopes of a signal and outputs a normalized value (0..1).
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

        self._method = _normalize_method(self._initial_state.get("method"), default="EMA")
        self._rise_alpha = float(_coerce_number(self._initial_state.get("rise_alpha")) or 0.4)
        self._fall_alpha = float(_coerce_number(self._initial_state.get("fall_alpha")) or 0.05)
        self._min_span = max(0.0, float(_coerce_number(self._initial_state.get("min_span")) or 0.25))
        self._sma_window = self._coerce_window(self._initial_state.get("sma_window"), default=10)
        self._margin = float(_coerce_number(self._initial_state.get("margin")) or 0.0)

        self._tracker = EnvelopeTracker(
            method=self._method,
            rise_alpha=self._rise_alpha,
            fall_alpha=self._fall_alpha,
            min_span=self._min_span,
            sma_window=self._sma_window,
        )
        self._last_outputs: dict[str, float | None] = {"lower": None, "upper": None, "normalized": None}
        self._last_input_value: float | None = None
        self._last_ctx_id: str | int | None = None
        self._dirty = True

    @staticmethod
    def _coerce_window(value: Any, *, default: int) -> int:
        numeric = _coerce_number(value)
        if numeric is None:
            return int(default)
        return max(1, int(round(float(numeric))))

    def _reset_cache(self) -> None:
        self._last_outputs = {"lower": None, "upper": None, "normalized": None}
        self._last_input_value = None
        self._last_ctx_id = None
        self._dirty = True

    def _apply_state_values(self, values: dict[str, Any]) -> None:
        tracker_changed = False
        margin_changed = False

        if "method" in values:
            method = _normalize_method(values.get("method"), default=self._method)
            if method != self._method:
                self._method = method
                tracker_changed = True

        if "rise_alpha" in values:
            numeric = _coerce_number(values.get("rise_alpha"))
            if numeric is not None and numeric != self._rise_alpha:
                self._rise_alpha = float(numeric)
                tracker_changed = True

        if "fall_alpha" in values:
            numeric = _coerce_number(values.get("fall_alpha"))
            if numeric is not None and numeric != self._fall_alpha:
                self._fall_alpha = float(numeric)
                tracker_changed = True

        if "min_span" in values:
            numeric = _coerce_number(values.get("min_span"))
            if numeric is not None:
                numeric = max(0.0, float(numeric))
                if numeric != self._min_span:
                    self._min_span = numeric
                    tracker_changed = True

        if "sma_window" in values:
            window = self._coerce_window(values.get("sma_window"), default=self._sma_window)
            if window != self._sma_window:
                self._sma_window = window
                tracker_changed = True

        if "margin" in values:
            numeric = _coerce_number(values.get("margin"))
            if numeric is not None and float(numeric) != self._margin:
                self._margin = float(numeric)
                margin_changed = True

        if tracker_changed:
            self._tracker.set_parameters(
                method=self._method,
                rise_alpha=self._rise_alpha,
                fall_alpha=self._fall_alpha,
                min_span=self._min_span,
                sma_window=self._sma_window,
            )
            self._reset_cache()
        elif margin_changed:
            self._dirty = True

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "")
        if name in {"method", "rise_alpha", "fall_alpha", "min_span", "sma_window", "margin"}:
            self._apply_state_values({name: value})

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_s = str(port)
        if port_s not in self._last_outputs:
            return None

        raw_value = await self.pull("value", ctx_id=ctx_id)
        numeric = _coerce_number(raw_value)
        if numeric is None:
            return self._last_outputs.get(port_s)

        if not self._dirty and self._last_outputs.get("normalized") is not None:
            if ctx_id is not None and ctx_id == self._last_ctx_id:
                return self._last_outputs.get(port_s)
            if ctx_id is None and self._last_input_value == numeric:
                return self._last_outputs.get(port_s)

        self._tracker.update(float(numeric))
        lower = self._tracker.lower
        upper = self._tracker.upper
        if lower is None or upper is None:
            self._last_outputs = {"lower": None, "upper": None, "normalized": None}
            self._last_input_value = float(numeric)
            self._last_ctx_id = ctx_id
            self._dirty = False
            return self._last_outputs.get(port_s)

        margin = float(self._margin)
        if margin:
            lower -= margin
            upper += margin

        span = upper - lower
        if span <= 0:
            normalized = 0.5
        else:
            normalized = (float(numeric) - lower) / span
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0

        self._last_outputs = {
            "lower": float(lower),
            "upper": float(upper),
            "normalized": float(normalized),
        }
        self._last_input_value = float(numeric)
        self._last_ctx_id = ctx_id
        self._dirty = False
        return self._last_outputs.get(port_s)


EnvelopeRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Envelope",
    description="Tracks upper/lower envelopes and emits a normalized value (0..1).",
    tags=["signal", "envelope", "normalize", "transform"],
    dataInPorts=[
        F8DataPortSpec(name="value", description="Input value.", valueSchema=number_schema(), required=False),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="lower", description="Estimated lower envelope.", valueSchema=number_schema()),
        F8DataPortSpec(name="upper", description="Estimated upper envelope.", valueSchema=number_schema()),
        F8DataPortSpec(name="normalized", description="Normalized value (0..1).", valueSchema=number_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="method",
            label="Method",
            description="Envelope filter method.",
            valueSchema=string_schema(default="EMA", enum=["EMA", "DEMA", "SMA"]),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="rise_alpha",
            label="Rise Alpha",
            description="Smoothing factor when moving toward the envelope.",
            valueSchema=number_schema(default=0.4, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="fall_alpha",
            label="Fall Alpha",
            description="Smoothing factor when moving away from the envelope.",
            valueSchema=number_schema(default=0.05, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="min_span",
            label="Min Span",
            description="Minimum enforced envelope span.",
            valueSchema=number_schema(default=0.25, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="sma_window",
            label="SMA Window",
            description="Window size for SMA mode.",
            valueSchema=number_schema(default=10, minimum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="margin",
            label="Margin",
            description="Extra margin added to the lower/upper envelopes before normalization.",
            valueSchema=number_schema(default=0.0, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return EnvelopeRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(EnvelopeRuntimeNode.SPEC, overwrite=True)
    return reg
