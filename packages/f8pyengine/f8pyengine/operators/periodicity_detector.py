from __future__ import annotations

import math
from typing import Any

import numpy as np
from f8pysdk.codec import coerce_bool, parse_number
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    number_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.periodicity_detector"
_MISSING_DECAY = 0.95


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


class _RmsWindow:
    def __init__(self, *, window: int) -> None:
        self.window = max(4, int(window))
        self.history: list[float] = []

    def set_window(self, window: int) -> None:
        self.window = max(4, int(window))
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]

    def reset(self) -> None:
        self.history = []

    def update(self, value: float) -> float:
        self.history.append(float(value))
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]
        if not self.history:
            return 0.0
        energy = sum(sample * sample for sample in self.history) / float(len(self.history))
        return math.sqrt(max(0.0, energy))


class ShortTimeAutocorrelationEstimator:
    def __init__(
        self,
        *,
        window: int = 150,
        min_lag: int = 10,
        max_lag: int = 150,
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
        self.last_peak_lag = 0

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
        self.last_peak_lag = min(self.last_peak_lag, self.max_lag)

    def reset(self, *, clear_confidence: bool) -> None:
        self.history = []
        self.last_peak_lag = 0
        if clear_confidence:
            self.last_confidence = 0.0

    def decay(self, factor: float) -> float:
        self.last_confidence = _clamp01(self.last_confidence * _clamp01(float(factor)))
        return self.last_confidence

    def update(self, value: float) -> float:
        self.history.append(float(value))
        if len(self.history) > self.window:
            self.history = self.history[-self.window :]
        confidence_raw, peak_lag = self._compute_raw_confidence()
        alpha = self.smoothing_alpha
        self.last_confidence = _clamp01(alpha * confidence_raw + (1.0 - alpha) * self.last_confidence)
        self.last_peak_lag = peak_lag
        return self.last_confidence

    def _compute_raw_confidence(self) -> tuple[float, int]:
        sample_count = len(self.history)
        if sample_count < 6:
            return 0.0, 0

        history = np.asarray(self.history, dtype=np.float64)
        centered = history - float(np.mean(history))
        energy = float(np.dot(centered, centered))
        if energy <= self.noise_floor:
            return 0.0, 0

        lag_min = max(1, min(self.min_lag, sample_count - 2))
        lag_max = max(lag_min, min(self.max_lag, sample_count - 2))
        if lag_max < lag_min:
            return 0.0, 0

        correlations: list[float] = []
        weighted_scores: list[float] = []
        for lag in range(lag_min, lag_max + 1):
            segment_a = centered[lag:]
            segment_b = centered[:-lag]
            numerator = float(np.dot(segment_a, segment_b))
            energy_a = float(np.dot(segment_a, segment_a))
            energy_b = float(np.dot(segment_b, segment_b))
            denom = math.sqrt(energy_a * energy_b)
            if denom <= self.noise_floor:
                correlation = 0.0
            else:
                # Normalize each lag by the overlapping segment energies, which is
                # equivalent to a Pearson-style lagged correlation on centered data.
                correlation = numerator / denom
            correlations.append(correlation)
            overlap_ratio = float(sample_count - lag) / float(sample_count)
            weighted_scores.append(correlation * overlap_ratio)

        if not correlations:
            return 0.0, 0

        peaks = self._find_peaks(correlations, weighted_scores=weighted_scores, lag_min=lag_min)
        if peaks:
            best_score, best_lag = max(peaks, key=lambda item: item[0])
            density_penalty = min(1.0, float(len(peaks)) / float(self.min_peaks))
        else:
            best_score = max(0.0, max(weighted_scores))
            best_index = weighted_scores.index(max(weighted_scores))
            best_lag = lag_min + best_index
            density_penalty = 1.0 if best_score > 0.0 else 0.0
        return _clamp01(best_score * density_penalty), int(best_lag)

    def _find_peaks(
        self,
        correlations: list[float],
        *,
        weighted_scores: list[float],
        lag_min: int,
    ) -> list[tuple[float, int]]:
        peaks: list[tuple[float, int]] = []
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
                peaks.append((float(weighted_scores[idx]), lag_min + idx))
        return peaks


class PeriodicityDetectorRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._window = self._coerce_window(self._initial_state.get("window"), default=150, minimum=8)
        self._min_lag = self._coerce_window(self._initial_state.get("min_lag"), default=10, minimum=1)
        self._max_lag = self._coerce_window(self._initial_state.get("max_lag"), default=150, minimum=self._min_lag)
        self._peak_prominence = max(0.0, float(parse_number(self._initial_state.get("peak_prominence")) or 0.1))
        self._min_peaks = self._coerce_window(self._initial_state.get("min_peaks"), default=1, minimum=1)
        self._smoothing_alpha = self._coerce_alpha(self._initial_state.get("smoothing_alpha"), default=0.25)
        self._noise_floor = max(0.0, float(parse_number(self._initial_state.get("noise_floor")) or 1e-4))
        self._threshold = self._coerce_alpha(self._initial_state.get("threshold"), default=0.6)
        self._rms_window = self._coerce_window(self._initial_state.get("rms_window"), default=64, minimum=4)
        self._reset_on_missing = coerce_bool(self._initial_state.get("reset_on_missing"), default=False)
        self._sample_interval_ms = self._coerce_sample_interval_ms(self._initial_state.get("sampleIntervalMs"), default=33.3333333333)

        self._estimator = ShortTimeAutocorrelationEstimator(
            window=self._window,
            min_lag=self._min_lag,
            max_lag=self._max_lag,
            peak_prominence=self._peak_prominence,
            min_peaks=self._min_peaks,
            smoothing_alpha=self._smoothing_alpha,
            noise_floor=self._noise_floor,
        )
        self._rms = _RmsWindow(window=self._rms_window)
        self._last_outputs: dict[str, float | bool] = {
            "confidence": 0.0,
            "rms": 0.0,
            "periodicEnergy": 0.0,
            "periodMs": 0.0,
            "period_hz": 0.0,
            "is_periodic": False,
        }
        self._last_input_value: float | None = None
        self._last_ctx_id: str | int | None = None
        self._dirty = True

    @staticmethod
    def _coerce_window(value: Any, *, default: int, minimum: int) -> int:
        numeric = parse_number(value)
        if numeric is None:
            return int(default)
        return max(minimum, int(round(float(numeric))))

    @staticmethod
    def _coerce_alpha(value: Any, *, default: float) -> float:
        numeric = parse_number(value)
        if numeric is None:
            return float(default)
        return _clamp01(float(numeric))

    def _apply_state(self) -> None:
        self._max_lag = max(self._min_lag, self._max_lag)
        self._estimator.set_parameters(
            window=self._window,
            min_lag=self._min_lag,
            max_lag=self._max_lag,
            peak_prominence=self._peak_prominence,
            min_peaks=self._min_peaks,
            smoothing_alpha=self._smoothing_alpha,
            noise_floor=self._noise_floor,
        )
        self._rms.set_window(self._rms_window)
        self._dirty = True

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "")
        if name == "window":
            self._window = self._coerce_window(value, default=self._window, minimum=8)
            self._apply_state()
            return
        if name == "min_lag":
            self._min_lag = self._coerce_window(value, default=self._min_lag, minimum=1)
            self._apply_state()
            return
        if name == "max_lag":
            self._max_lag = self._coerce_window(value, default=self._max_lag, minimum=self._min_lag)
            self._apply_state()
            return
        if name == "peak_prominence":
            numeric = parse_number(value)
            if numeric is not None:
                self._peak_prominence = max(0.0, float(numeric))
                self._apply_state()
            return
        if name == "min_peaks":
            self._min_peaks = self._coerce_window(value, default=self._min_peaks, minimum=1)
            self._apply_state()
            return
        if name == "smoothing_alpha":
            self._smoothing_alpha = self._coerce_alpha(value, default=self._smoothing_alpha)
            self._apply_state()
            return
        if name == "noise_floor":
            numeric = parse_number(value)
            if numeric is not None:
                self._noise_floor = max(0.0, float(numeric))
                self._apply_state()
            return
        if name == "threshold":
            self._threshold = self._coerce_alpha(value, default=self._threshold)
            self._dirty = True
            return
        if name == "rms_window":
            self._rms_window = self._coerce_window(value, default=self._rms_window, minimum=4)
            self._apply_state()
            return
        if name == "reset_on_missing":
            self._reset_on_missing = coerce_bool(value, default=self._reset_on_missing)
            return
        if name == "sampleIntervalMs":
            self._sample_interval_ms = self._coerce_sample_interval_ms(value, default=self._sample_interval_ms)
            self._dirty = True

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_name = str(port)
        if port_name not in self._last_outputs:
            return None

        numeric = parse_number(await self.pull("value", ctx_id=ctx_id))
        if numeric is None:
            if self._reset_on_missing:
                confidence = self._estimator.decay(_MISSING_DECAY)
                self._last_outputs["confidence"] = confidence
                self._last_outputs["periodicEnergy"] = float(self._last_outputs.get("rms", 0.0)) * confidence
                self._last_outputs["is_periodic"] = bool(confidence >= self._threshold)
                period_ms = float(self._last_outputs.get("periodMs", 0.0))
                self._last_outputs["period_hz"] = (1000.0 / period_ms) if period_ms > 0.0 else 0.0
            return self._last_outputs[port_name]

        if not self._dirty:
            if ctx_id is not None and ctx_id == self._last_ctx_id:
                return self._last_outputs[port_name]
            if ctx_id is None and self._last_input_value == numeric:
                return self._last_outputs[port_name]

        sample = float(numeric)
        confidence = self._estimator.update(sample)
        rms_value = self._rms.update(sample)
        periodic_energy = rms_value * confidence
        is_periodic = confidence >= self._threshold

        self._last_outputs["confidence"] = float(confidence)
        self._last_outputs["rms"] = float(rms_value)
        self._last_outputs["periodicEnergy"] = float(periodic_energy)
        period_ms = float(self._estimator.last_peak_lag) * self._sample_interval_ms if self._estimator.last_peak_lag > 0 else 0.0
        self._last_outputs["periodMs"] = period_ms
        self._last_outputs["period_hz"] = (1000.0 / period_ms) if period_ms > 0.0 else 0.0
        self._last_outputs["is_periodic"] = bool(is_periodic)
        self._last_input_value = sample
        self._last_ctx_id = ctx_id
        self._dirty = False
        return self._last_outputs[port_name]

    @staticmethod
    def _coerce_sample_interval_ms(value: Any, *, default: float) -> float:
        interval_ms = parse_number(value)
        if interval_ms is not None:
            return max(1e-6, float(interval_ms))
        return max(1e-6, float(default))


PeriodicityDetectorRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.signal",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Periodicity Detector",
    description="Detects whether a scalar signal is periodic using short-time autocorrelation peaks.",
    tags=["signal", "periodicity", "autocorrelation", "confidence", "rms"],
    dataInPorts=[F8DataPortSpec(name="value", description="Scalar signal input.", valueSchema=number_schema())],
    dataOutPorts=[
        F8DataPortSpec(name="confidence", description="Autocorrelation periodicity confidence (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="rms", description="Short-term RMS envelope.", valueSchema=number_schema()),
        F8DataPortSpec(name="periodicEnergy", description="RMS multiplied by periodicity confidence.", valueSchema=number_schema()),
        F8DataPortSpec(name="periodMs", description="Detected dominant period in milliseconds.", valueSchema=number_schema()),
        F8DataPortSpec(name="period_hz", description="Detected dominant frequency in Hz.", valueSchema=number_schema()),
        F8DataPortSpec(name="is_periodic", description="True when confidence exceeds threshold.", valueSchema=boolean_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="window",
            label="Window",
            description="Autocorrelation history window in samples.",
            valueSchema=number_schema(default=150, minimum=8.0, maximum=4096.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="min_lag",
            label="Min Lag",
            description="Minimum lag to scan for periodic peaks.",
            valueSchema=number_schema(default=10, minimum=1.0, maximum=4096.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="max_lag",
            label="Max Lag",
            description="Maximum lag to scan for periodic peaks.",
            valueSchema=number_schema(default=150, minimum=1.0, maximum=4096.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="peak_prominence",
            label="Peak Prominence",
            description="Minimum local prominence for a valid autocorrelation peak.",
            valueSchema=number_schema(default=0.1, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="min_peaks",
            label="Min Peaks",
            description="Minimum number of valid peaks before full confidence.",
            valueSchema=number_schema(default=1, minimum=1.0, maximum=16.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="smoothing_alpha",
            label="Smoothing Alpha",
            description="EMA smoothing factor applied to confidence.",
            valueSchema=number_schema(default=0.25, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="noise_floor",
            label="Noise Floor",
            description="Minimum centered energy before confidence can rise.",
            valueSchema=number_schema(default=1e-4, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="threshold",
            label="Threshold",
            description="Decision threshold for the boolean periodic output.",
            valueSchema=number_schema(default=0.6, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="rms_window",
            label="RMS Window",
            description="RMS window length in samples.",
            valueSchema=number_schema(default=64, minimum=4.0, maximum=4096.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="sampleIntervalMs",
            label="Sample Interval (ms)",
            description="Sampling interval in milliseconds used to convert detected period into frequency.",
            valueSchema=number_schema(default=33.3333333333, minimum=0.001, maximum=50000.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="reset_on_missing",
            label="Reset On Missing",
            description="Decay confidence when the input is missing.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(PeriodicityDetectorRuntimeNode.SPEC, PeriodicityDetectorRuntimeNode, overwrite=True)
    return registry
