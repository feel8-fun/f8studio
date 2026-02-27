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
    boolean_schema,
    number_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.envelope"

_METHODS = ("EMA", "DEMA", "SMA")
_EPS = 1e-9
_CONFIDENCE_MISSING_DECAY = 0.95


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


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    numeric = _coerce_number(value)
    if numeric is not None:
        return bool(numeric != 0.0)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


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
        return _clamp01(normalized)

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


class PeriodicityConfidenceEstimator:
    """Estimate periodicity confidence from autocorrelation peak structure."""

    def __init__(
        self,
        *,
        window: int = 128,
        min_lag: int = 4,
        max_lag: int = 48,
        peak_prominence: float = 0.1,
        min_peaks: int = 1,
        smoothing_alpha: float = 0.25,
        noise_floor: float = 1e-4,
    ) -> None:
        self.window = max(8, int(window))
        self.min_lag = max(1, int(min_lag))
        self.max_lag = max(self.min_lag, int(max_lag))
        self.peak_prominence = max(0.0, float(peak_prominence))
        self.min_peaks = max(1, int(min_peaks))
        self.smoothing_alpha = _clamp01(float(smoothing_alpha))
        self.noise_floor = max(0.0, float(noise_floor))
        self.history: list[float] = []
        self.last_confidence = 0.0

    def set_parameters(
        self,
        *,
        window: int | None = None,
        min_lag: int | None = None,
        max_lag: int | None = None,
        peak_prominence: float | None = None,
        min_peaks: int | None = None,
        smoothing_alpha: float | None = None,
        noise_floor: float | None = None,
    ) -> None:
        if window is not None:
            self.window = max(8, int(window))
        if min_lag is not None:
            self.min_lag = max(1, int(min_lag))
        if max_lag is not None:
            self.max_lag = max(self.min_lag, int(max_lag))
        if peak_prominence is not None:
            self.peak_prominence = max(0.0, float(peak_prominence))
        if min_peaks is not None:
            self.min_peaks = max(1, int(min_peaks))
        if smoothing_alpha is not None:
            self.smoothing_alpha = _clamp01(float(smoothing_alpha))
        if noise_floor is not None:
            self.noise_floor = max(0.0, float(noise_floor))
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

    def reset(self, *, clear_confidence: bool) -> None:
        self.history = []
        if clear_confidence:
            self.last_confidence = 0.0

    def decay(self, factor: float) -> float:
        clipped_factor = _clamp01(float(factor))
        self.last_confidence = _clamp01(self.last_confidence * clipped_factor)
        return self.last_confidence

    def update(self, value: float) -> float:
        self.history.append(float(value))
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

        confidence_raw = self._compute_raw_confidence()
        alpha = self.smoothing_alpha
        self.last_confidence = _clamp01(alpha * confidence_raw + (1.0 - alpha) * self.last_confidence)
        return self.last_confidence

    def _compute_raw_confidence(self) -> float:
        sample_count = len(self.history)
        if sample_count < 6:
            return 0.0

        history_mean = sum(self.history) / float(sample_count)
        centered = [value - history_mean for value in self.history]
        energy = sum(value * value for value in centered)
        if energy <= self.noise_floor:
            return 0.0

        lag_min = max(1, min(self.min_lag, sample_count - 2))
        lag_max = max(lag_min, min(self.max_lag, sample_count - 2))
        if lag_max < lag_min:
            return 0.0

        correlations: list[float] = []
        for lag in range(lag_min, lag_max + 1):
            numerator = 0.0
            for idx in range(lag, sample_count):
                numerator += centered[idx] * centered[idx - lag]
            correlations.append(numerator / energy)

        if not correlations:
            return 0.0

        peaks = self._find_peaks(correlations)
        peak_score = max(peaks) if peaks else max(0.0, max(correlations))

        if self.min_peaks <= 0:
            density_penalty = 1.0
        else:
            density_penalty = min(1.0, float(len(peaks)) / float(self.min_peaks))

        return _clamp01(peak_score * density_penalty)

    def _find_peaks(self, correlations: list[float]) -> list[float]:
        peaks: list[float] = []
        if len(correlations) < 3:
            return peaks

        for idx in range(1, len(correlations) - 1):
            left_value = correlations[idx - 1]
            mid_value = correlations[idx]
            right_value = correlations[idx + 1]
            if mid_value < left_value or mid_value < right_value:
                continue
            local_prominence = mid_value - max(left_value, right_value)
            if mid_value > 0.0 and local_prominence >= self.peak_prominence:
                peaks.append(float(mid_value))
        return peaks


class EnvelopeRuntimeNode(OperatorNode):
    """Tracks upper/lower envelopes and emits normalized value plus confidence."""

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

        self._jump_enabled = self._coerce_bool(self._initial_state.get("jumpEnabled"), default=True)
        self._jump_span_mult = max(0.5, float(_coerce_number(self._initial_state.get("jumpSpanMult")) or 4.0))
        self._jump_consecutive_frames = self._coerce_window(
            self._initial_state.get("jumpConsecutiveFrames"), default=4
        )
        self._jump_reseed_frames = self._coerce_window(self._initial_state.get("jumpReseedFrames"), default=8)
        self._jump_reset_confidence = self._coerce_bool(
            self._initial_state.get("jumpResetConfidence"), default=True
        )

        self._confidence_enabled = self._coerce_bool(self._initial_state.get("confidenceEnabled"), default=True)
        self._confidence_window = max(8, self._coerce_window(self._initial_state.get("confidenceWindow"), default=128))
        self._confidence_min_lag = max(1, self._coerce_window(self._initial_state.get("confidenceMinLag"), default=4))
        self._confidence_max_lag = max(
            self._confidence_min_lag,
            self._coerce_window(self._initial_state.get("confidenceMaxLag"), default=48),
        )
        self._confidence_peak_prominence = max(
            0.0,
            float(_coerce_number(self._initial_state.get("confidencePeakProminence")) or 0.1),
        )
        self._confidence_min_peaks = max(
            1,
            self._coerce_window(self._initial_state.get("confidenceMinPeaks"), default=1),
        )
        self._confidence_smoothing_alpha = self._coerce_alpha(
            self._initial_state.get("confidenceSmoothingAlpha"), default=0.25
        )
        self._confidence_noise_floor = max(
            0.0,
            float(_coerce_number(self._initial_state.get("confidenceNoiseFloor")) or 1e-4),
        )
        self._confidence_reset_on_missing = self._coerce_bool(
            self._initial_state.get("confidenceResetOnMissing"), default=False
        )

        self._tracker = EnvelopeTracker(
            method=self._method,
            rise_alpha=self._rise_alpha,
            fall_alpha=self._fall_alpha,
            min_span=self._min_span,
            sma_window=self._sma_window,
        )
        self._confidence_estimator = PeriodicityConfidenceEstimator(
            window=self._confidence_window,
            min_lag=self._confidence_min_lag,
            max_lag=self._confidence_max_lag,
            peak_prominence=self._confidence_peak_prominence,
            min_peaks=self._confidence_min_peaks,
            smoothing_alpha=self._confidence_smoothing_alpha,
            noise_floor=self._confidence_noise_floor,
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
            "confidence": 0.0,
        }
        self._last_input_value: float | None = None
        self._last_ctx_id: str | int | None = None
        self._dirty = True

    @staticmethod
    def _coerce_window(value: Any, *, default: int) -> int:
        numeric = _coerce_number(value)
        if numeric is None:
            return int(default)
        return max(1, int(round(float(numeric))))

    @staticmethod
    def _coerce_alpha(value: Any, *, default: float) -> float:
        numeric = _coerce_number(value)
        if numeric is None:
            return float(default)
        return _clamp01(float(numeric))

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        coerced = _coerce_bool(value)
        if coerced is None:
            return bool(default)
        return bool(coerced)

    def _reset_cache(self) -> None:
        last_confidence = self._last_outputs.get("confidence")
        self._last_outputs = {
            "lower": None,
            "upper": None,
            "normalized": None,
            "confidence": float(last_confidence) if last_confidence is not None else 0.0,
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
        confidence_changed = False

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

        if "jumpEnabled" in values:
            jump_enabled = self._coerce_bool(values.get("jumpEnabled"), default=self._jump_enabled)
            if jump_enabled != self._jump_enabled:
                self._jump_enabled = jump_enabled
                jump_changed = True

        if "jumpSpanMult" in values:
            numeric = _coerce_number(values.get("jumpSpanMult"))
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

        if "jumpResetConfidence" in values:
            jump_reset_conf = self._coerce_bool(
                values.get("jumpResetConfidence"), default=self._jump_reset_confidence
            )
            if jump_reset_conf != self._jump_reset_confidence:
                self._jump_reset_confidence = jump_reset_conf
                jump_changed = True

        if "confidenceEnabled" in values:
            conf_enabled = self._coerce_bool(values.get("confidenceEnabled"), default=self._confidence_enabled)
            if conf_enabled != self._confidence_enabled:
                self._confidence_enabled = conf_enabled
                confidence_changed = True

        if "confidenceWindow" in values:
            window = max(8, self._coerce_window(values.get("confidenceWindow"), default=self._confidence_window))
            if window != self._confidence_window:
                self._confidence_window = window
                confidence_changed = True

        if "confidenceMinLag" in values:
            lag_min = max(1, self._coerce_window(values.get("confidenceMinLag"), default=self._confidence_min_lag))
            if lag_min != self._confidence_min_lag:
                self._confidence_min_lag = lag_min
                if self._confidence_max_lag < self._confidence_min_lag:
                    self._confidence_max_lag = self._confidence_min_lag
                confidence_changed = True

        if "confidenceMaxLag" in values:
            lag_max = self._coerce_window(values.get("confidenceMaxLag"), default=self._confidence_max_lag)
            lag_max = max(self._confidence_min_lag, lag_max)
            if lag_max != self._confidence_max_lag:
                self._confidence_max_lag = lag_max
                confidence_changed = True

        if "confidencePeakProminence" in values:
            numeric = _coerce_number(values.get("confidencePeakProminence"))
            if numeric is not None:
                numeric = max(0.0, float(numeric))
                if numeric != self._confidence_peak_prominence:
                    self._confidence_peak_prominence = numeric
                    confidence_changed = True

        if "confidenceMinPeaks" in values:
            min_peaks = max(1, self._coerce_window(values.get("confidenceMinPeaks"), default=self._confidence_min_peaks))
            if min_peaks != self._confidence_min_peaks:
                self._confidence_min_peaks = min_peaks
                confidence_changed = True

        if "confidenceSmoothingAlpha" in values:
            alpha = self._coerce_alpha(values.get("confidenceSmoothingAlpha"), default=self._confidence_smoothing_alpha)
            if alpha != self._confidence_smoothing_alpha:
                self._confidence_smoothing_alpha = alpha
                confidence_changed = True

        if "confidenceNoiseFloor" in values:
            numeric = _coerce_number(values.get("confidenceNoiseFloor"))
            if numeric is not None:
                numeric = max(0.0, float(numeric))
                if numeric != self._confidence_noise_floor:
                    self._confidence_noise_floor = numeric
                    confidence_changed = True

        if "confidenceResetOnMissing" in values:
            reset_on_missing = self._coerce_bool(
                values.get("confidenceResetOnMissing"), default=self._confidence_reset_on_missing
            )
            if reset_on_missing != self._confidence_reset_on_missing:
                self._confidence_reset_on_missing = reset_on_missing
                confidence_changed = True

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

        if confidence_changed:
            self._confidence_estimator.set_parameters(
                window=self._confidence_window,
                min_lag=self._confidence_min_lag,
                max_lag=self._confidence_max_lag,
                peak_prominence=self._confidence_peak_prominence,
                min_peaks=self._confidence_min_peaks,
                smoothing_alpha=self._confidence_smoothing_alpha,
                noise_floor=self._confidence_noise_floor,
            )
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
        if self._jump_reset_confidence:
            self._confidence_estimator.reset(clear_confidence=True)
            self._last_outputs["confidence"] = 0.0

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
            "jumpResetConfidence",
            "confidenceEnabled",
            "confidenceWindow",
            "confidenceMinLag",
            "confidenceMaxLag",
            "confidencePeakProminence",
            "confidenceMinPeaks",
            "confidenceSmoothingAlpha",
            "confidenceNoiseFloor",
            "confidenceResetOnMissing",
        }:
            self._apply_state_values({name: value})

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_s = str(port)
        if port_s not in self._last_outputs:
            return None

        raw_value = await self.pull("value", ctx_id=ctx_id)
        numeric = _coerce_number(raw_value)
        if numeric is None:
            if self._confidence_enabled and self._confidence_reset_on_missing:
                decayed = self._confidence_estimator.decay(_CONFIDENCE_MISSING_DECAY)
                self._last_outputs["confidence"] = decayed
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
                "confidence": self._last_outputs.get("confidence", 0.0),
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

        if self._confidence_enabled:
            confidence = self._confidence_estimator.update(input_value)
        else:
            confidence = 0.0

        self._last_outputs = {
            "lower": output_lower,
            "upper": output_upper,
            "normalized": normalized,
            "confidence": confidence,
        }
        self._last_input_value = input_value
        self._last_ctx_id = ctx_id
        self._dirty = False
        return self._last_outputs.get(port_s)


EnvelopeRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Envelope",
    description="Tracks upper/lower envelopes and emits normalized + confidence outputs.",
    tags=["signal", "envelope", "normalize", "confidence", "transform"],
    dataInPorts=[
        F8DataPortSpec(name="value", description="Input value.", valueSchema=number_schema(), required=False),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="lower", description="Estimated lower envelope.", valueSchema=number_schema()),
        F8DataPortSpec(name="upper", description="Estimated upper envelope.", valueSchema=number_schema()),
        F8DataPortSpec(name="normalized", description="Normalized value (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(
            name="confidence",
            description="Periodicity confidence estimate (0..1).",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=1.0),
        ),
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
            description="Extra margin added to envelopes before normalization.",
            valueSchema=number_schema(default=0.0, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpEnabled",
            label="Jump Enabled",
            description="Enable consecutive-frame jump detection and reseed.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpSpanMult",
            label="Jump Span Mult",
            description="Distance threshold in envelope-span units for jump detection.",
            valueSchema=number_schema(default=4.0, minimum=0.5),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpConsecutiveFrames",
            label="Jump Frames",
            description="Consecutive far frames required before jump trigger.",
            valueSchema=number_schema(default=4, minimum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpReseedFrames",
            label="Reseed Frames",
            description="Blend length (frames) after jump reset.",
            valueSchema=number_schema(default=8, minimum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpResetConfidence",
            label="Jump Reset Confidence",
            description="Reset confidence history when jump reset is triggered.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confidenceEnabled",
            label="Confidence Enabled",
            description="Enable periodicity confidence estimation.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confidenceWindow",
            label="Confidence Window",
            description="Sliding window size for autocorrelation confidence.",
            valueSchema=number_schema(default=128, minimum=8.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confidenceMinLag",
            label="Confidence Min Lag",
            description="Minimum lag used in autocorrelation scan.",
            valueSchema=number_schema(default=4, minimum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confidenceMaxLag",
            label="Confidence Max Lag",
            description="Maximum lag used in autocorrelation scan.",
            valueSchema=number_schema(default=48, minimum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confidencePeakProminence",
            label="Peak Prominence",
            description="Minimum local prominence for valid autocorrelation peaks.",
            valueSchema=number_schema(default=0.1, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confidenceMinPeaks",
            label="Min Peaks",
            description="Minimum valid autocorrelation peaks before full confidence.",
            valueSchema=number_schema(default=1, minimum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confidenceSmoothingAlpha",
            label="Conf Smooth Alpha",
            description="EMA smoothing factor for confidence output.",
            valueSchema=number_schema(default=0.25, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confidenceNoiseFloor",
            label="Noise Floor",
            description="Minimum energy needed before periodicity confidence can rise.",
            valueSchema=number_schema(default=1e-4, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confidenceResetOnMissing",
            label="Conf Reset On Missing",
            description="Decay confidence when input is missing.",
            valueSchema=boolean_schema(default=False),
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
