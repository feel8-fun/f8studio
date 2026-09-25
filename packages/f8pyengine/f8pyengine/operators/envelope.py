from __future__ import annotations

import time
from typing import Any, TypeAlias

from f8pysdk.codec import coerce_bool
from f8pysdk.codec import parse_number
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    number_schema,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.envelope"

_METHODS = ("EMA", "DEMA", "SMA")
_EPS = 1e-9

def _normalize_method(value: Any, *, default: str = "EMA") -> str:
    method = str(value or "").strip().upper()
    if method in _METHODS:
        return method
    return default


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


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
            ema_pt = self.ema_pt
            ema2_pt = self.ema2_pt
            if ema_pt is None or ema2_pt is None:
                self.ema_pt = value
                self.ema2_pt = value
            else:
                self.ema_pt = active_alpha * value + (1.0 - active_alpha) * ema_pt
                self.ema2_pt = active_alpha * self.ema_pt + (1.0 - active_alpha) * ema2_pt
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


EnvelopeFilter: TypeAlias = ExponentialMovingAverage | DoubleExponentialMovingAverage | SimpleMovingAverage


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
        self.upper_filter: EnvelopeFilter | None = None
        self.lower_filter: EnvelopeFilter | None = None
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
        return _clamp01(normalized)

    def _reset_filters(self) -> None:
        if self.upper_filter is not None:
            self.upper_filter.reset()
        if self.lower_filter is not None:
            self.lower_filter.reset()
        self.upper = None
        self.lower = None

    def _create_filter(self) -> EnvelopeFilter:
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
            if not isinstance(filt, SimpleMovingAverage):
                continue
            filt.set_window(self.sma_window)

    def _update_filter(self, filt: EnvelopeFilter, value: float, alpha: float) -> float:
        if self.method == "SMA" and isinstance(filt, SimpleMovingAverage):
            filt.set_window(self.sma_window)
            return float(filt.update(value))
        return float(filt.update(value, alpha=alpha))


class EnvelopeRuntimeNode(OperatorNode):
    """Tracks upper/lower envelopes and emits normalized value."""

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

        self._method = _normalize_method(self._initial_state.get("method"), default="EMA")
        self._rise_alpha = float(parse_number(self._initial_state.get("rise_alpha")) or 0.4)
        self._fall_alpha = float(parse_number(self._initial_state.get("fall_alpha")) or 0.05)
        self._min_span = max(0.0, float(parse_number(self._initial_state.get("min_span")) or 0.25))
        self._sma_window = self._coerce_window(self._initial_state.get("sma_window"), default=10)
        self._margin = float(parse_number(self._initial_state.get("margin")) or 0.0)

        self._jump_enabled = coerce_bool(self._initial_state.get("jumpEnabled"), default=True)
        self._jump_span_mult = max(0.5, float(parse_number(self._initial_state.get("jumpSpanMult")) or 4.0))
        self._jump_consecutive_frames = self._coerce_window(
            self._initial_state.get("jumpConsecutiveFrames"), default=4
        )
        self._jump_reseed_frames = self._coerce_window(self._initial_state.get("jumpReseedFrames"), default=8)

        self._tracker = EnvelopeTracker(
            method=self._method,
            rise_alpha=self._rise_alpha,
            fall_alpha=self._fall_alpha,
            min_span=self._min_span,
            sma_window=self._sma_window,
        )

        self._far_count = 0
        self._far_reference_midpoint: float | None = None
        self._far_reference_span = 1.0
        self._in_reseed = False
        self._reseed_step = 0
        self._reseed_start_norm = 0.5
        self._jump_count = 0
        self._last_jump_ts_ms: int | None = None

        self._last_outputs: dict[str, float | None] = {
            "lower": None,
            "upper": None,
            "normalized": None,
        }
        self._last_input_value: float | None = None
        self._last_ctx_id: str | int | None = None
        self._dirty = True

    @staticmethod
    def _coerce_window(value: Any, *, default: int) -> int:
        numeric = parse_number(value)
        if numeric is None:
            return int(default)
        return max(1, int(round(float(numeric))))

    @staticmethod
    def _coerce_alpha(value: Any, *, default: float) -> float:
        numeric = parse_number(value)
        if numeric is None:
            return float(default)
        return _clamp01(float(numeric))

    def _reset_cache(self) -> None:
        self._last_outputs = {
            "lower": None,
            "upper": None,
            "normalized": None,
        }
        self._last_input_value = None
        self._last_ctx_id = None
        self._dirty = True

    def _reset_jump_state(self) -> None:
        self._far_count = 0
        self._far_reference_midpoint = None
        self._far_reference_span = 1.0
        self._in_reseed = False
        self._reseed_step = 0
        self._reseed_start_norm = 0.5

    def _apply_state_values(self, values: dict[str, Any]) -> None:
        tracker_changed = False
        margin_changed = False
        jump_changed = False

        if "method" in values:
            method = _normalize_method(values.get("method"), default=self._method)
            if method != self._method:
                self._method = method
                tracker_changed = True

        if "rise_alpha" in values:
            numeric = parse_number(values.get("rise_alpha"))
            if numeric is not None and numeric != self._rise_alpha:
                self._rise_alpha = float(numeric)
                tracker_changed = True

        if "fall_alpha" in values:
            numeric = parse_number(values.get("fall_alpha"))
            if numeric is not None and numeric != self._fall_alpha:
                self._fall_alpha = float(numeric)
                tracker_changed = True

        if "min_span" in values:
            numeric = parse_number(values.get("min_span"))
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
            numeric = parse_number(values.get("margin"))
            if numeric is not None and float(numeric) != self._margin:
                self._margin = float(numeric)
                margin_changed = True

        if "jumpEnabled" in values:
            jump_enabled = coerce_bool(values.get("jumpEnabled"), default=self._jump_enabled)
            if jump_enabled != self._jump_enabled:
                self._jump_enabled = jump_enabled
                jump_changed = True

        if "jumpSpanMult" in values:
            numeric = parse_number(values.get("jumpSpanMult"))
            if numeric is not None:
                numeric = max(0.5, float(numeric))
                if numeric != self._jump_span_mult:
                    self._jump_span_mult = numeric
                    jump_changed = True

        if "jumpConsecutiveFrames" in values:
            window = self._coerce_window(values.get("jumpConsecutiveFrames"), default=self._jump_consecutive_frames)
            if window != self._jump_consecutive_frames:
                self._jump_consecutive_frames = window
                jump_changed = True

        if "jumpReseedFrames" in values:
            window = self._coerce_window(values.get("jumpReseedFrames"), default=self._jump_reseed_frames)
            if window != self._jump_reseed_frames:
                self._jump_reseed_frames = window
                jump_changed = True

        if tracker_changed:
            self._tracker.set_parameters(
                method=self._method,
                rise_alpha=self._rise_alpha,
                fall_alpha=self._fall_alpha,
                min_span=self._min_span,
                sma_window=self._sma_window,
            )
            self._reset_jump_state()
            self._reset_cache()
        elif margin_changed or jump_changed:
            if jump_changed:
                self._reset_jump_state()
            self._dirty = True

    def _trigger_jump_reset(self) -> None:
        current_norm = self._last_outputs.get("normalized")
        self._reseed_start_norm = float(current_norm) if current_norm is not None else 0.5
        self._in_reseed = self._jump_reseed_frames > 0
        self._reseed_step = 0
        self._tracker.reset()
        self._far_count = 0
        self._far_reference_midpoint = None
        self._far_reference_span = 1.0
        self._jump_count += 1
        self._last_jump_ts_ms = int(time.time() * 1000)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "")
        if name in {
            "method",
            "rise_alpha",
            "fall_alpha",
            "min_span",
            "sma_window",
            "margin",
            "jumpEnabled",
            "jumpSpanMult",
            "jumpConsecutiveFrames",
            "jumpReseedFrames",
        }:
            self._apply_state_values({name: value})

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_s = str(port)
        if port_s not in self._last_outputs:
            return None

        raw_value = await self.pull("value", ctx_id=ctx_id)
        numeric = parse_number(raw_value)
        if numeric is None:
            return self._last_outputs.get(port_s)

        if not self._dirty and self._last_outputs.get("normalized") is not None:
            if ctx_id is not None and ctx_id == self._last_ctx_id:
                return self._last_outputs.get(port_s)
            if ctx_id is None and self._last_input_value == numeric:
                return self._last_outputs.get(port_s)

        input_value = float(numeric)
        if self._jump_enabled:
            lower_pre = self._tracker.lower
            upper_pre = self._tracker.upper
            if lower_pre is not None and upper_pre is not None:
                raw_span = upper_pre - lower_pre
                # Skip jump detection while envelope is still re-seeding and span is unstable.
                if raw_span < max(self._min_span * 0.5, _EPS):
                    self._far_count = 0
                    self._far_reference_midpoint = None
                    self._far_reference_span = 1.0
                else:
                    if self._far_count > 0 and self._far_reference_midpoint is not None:
                        midpoint = self._far_reference_midpoint
                        span = max(self._far_reference_span, _EPS)
                    else:
                        midpoint = 0.5 * (upper_pre + lower_pre)
                        span = max(raw_span, _EPS)
                    distance_in_span = abs(input_value - midpoint) / span
                    if distance_in_span >= self._jump_span_mult:
                        if self._far_count == 0:
                            self._far_reference_midpoint = midpoint
                            self._far_reference_span = span
                        self._far_count += 1
                    else:
                        self._far_count = 0
                        self._far_reference_midpoint = None
                        self._far_reference_span = 1.0
                    if self._far_count >= self._jump_consecutive_frames:
                        self._trigger_jump_reset()
            else:
                self._far_count = 0
                self._far_reference_midpoint = None
                self._far_reference_span = 1.0
        else:
            self._far_count = 0
            self._far_reference_midpoint = None
            self._far_reference_span = 1.0

        self._tracker.update(input_value)
        lower = self._tracker.lower
        upper = self._tracker.upper
        if lower is None or upper is None:
            self._last_outputs = {
                "lower": None,
                "upper": None,
                "normalized": None,
            }
            self._last_input_value = input_value
            self._last_ctx_id = ctx_id
            self._dirty = False
            return self._last_outputs.get(port_s)

        output_lower = float(lower)
        output_upper = float(upper)
        margin = float(self._margin)
        if margin:
            output_lower -= margin
            output_upper += margin

        span = output_upper - output_lower
        if span <= 0.0:
            normalized_raw = 0.5
        else:
            normalized_raw = _clamp01((input_value - output_lower) / span)

        normalized = normalized_raw
        if self._in_reseed and self._jump_reseed_frames > 0:
            blend = min(1.0, float(self._reseed_step + 1) / float(self._jump_reseed_frames))
            normalized = _clamp01((1.0 - blend) * self._reseed_start_norm + blend * normalized_raw)
            self._reseed_step += 1
            if self._reseed_step >= self._jump_reseed_frames:
                self._in_reseed = False

        self._last_outputs = {
            "lower": output_lower,
            "upper": output_upper,
            "normalized": normalized,
        }
        self._last_input_value = input_value
        self._last_ctx_id = ctx_id
        self._dirty = False
        return self._last_outputs.get(port_s)


EnvelopeRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.signal",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Envelope",
    description=(
        "Track a signal envelope and normalize it into a stable 0..1 range.\n"
        "\n"
        "Core\n"
        "- Input `value` is tracked with lower and upper envelope estimators.\n"
        "- Outputs are `lower`, `upper`, and `normalized`.\n"
        "- `normalized` maps the current input between the tracked envelopes, including optional margin and minimum span.\n"
        "\n"
        "Envelope Modes\n"
        "- `EMA`: simple exponential tracking\n"
        "- `DEMA`: double exponential tracking with faster response\n"
        "- `SMA`: moving average window smoothing\n"
        "\n"
        "Jump Handling\n"
        "- Optional jump detection can reseed the envelopes after large sustained changes.\n"
        "- Jump settings control the trigger threshold, consecutive frames, and reseed blend time.\n"
        "\n"
        "Examples\n"
        "- Normalize a noisy control signal into `normalized`\n"
        "- Track changing lower/upper motion bounds\n"
    ),
    tags=["signal", "envelope", "normalize", "transform"],
    dataInPorts=[
        F8DataPortSpec(name="value", description="Input value.", valueSchema=number_schema(), definitionProtected=False),
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
            description="Envelope tracking method: `EMA`, `DEMA`, or `SMA`.",
            valueSchema=string_schema(default="EMA", enum=["EMA", "DEMA", "SMA"]),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="rise_alpha",
            label="Rise Alpha",
            description="Smoothing factor when the estimator moves toward the current envelope edge.",
            valueSchema=number_schema(default=0.4, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="fall_alpha",
            label="Fall Alpha",
            description="Smoothing factor when the estimator relaxes away from the current envelope edge.",
            valueSchema=number_schema(default=0.05, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="min_span",
            label="Min Span",
            description="Minimum enforced distance between lower and upper envelopes before normalization.",
            valueSchema=number_schema(default=0.25, minimum=0.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="sma_window",
            label="SMA Window",
            description="Moving-average window size used when Method is `SMA`.",
            valueSchema=number_schema(default=10, minimum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="margin",
            label="Margin",
            description="Extra padding added outside the envelopes before computing `normalized`.",
            valueSchema=number_schema(default=0.0, minimum=0.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpEnabled",
            label="Jump Enabled",
            description="Enable consecutive-frame jump detection and reseed.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpSpanMult",
            label="Jump Span Mult",
            description="Distance threshold in envelope-span units for jump detection.",
            valueSchema=number_schema(default=4.0, minimum=0.5),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpConsecutiveFrames",
            label="Jump Frames",
            description="Consecutive far frames required before jump trigger.",
            valueSchema=number_schema(default=4, minimum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpReseedFrames",
            label="Reseed Frames",
            description="Blend length (frames) after jump reset.",
            valueSchema=number_schema(default=8, minimum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(EnvelopeRuntimeNode.SPEC, EnvelopeRuntimeNode, overwrite=True)
    return registry
