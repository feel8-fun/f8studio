from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from f8pysdk.codec import parse_number

import numpy as np
from scipy.signal import butter, sosfilt

def format_output(result: Iterable[float] | None) -> Any:
    if result is None:
        return None
    values = list(result)
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return [float(value) for value in values]

def zero_output(dimension: int) -> tuple[float, ...]:
    if dimension <= 0:
        return ()
    return tuple(0.0 for _ in range(dimension))

@dataclass(slots=True)
class NodeComputationCache:
    last_input: tuple[float, ...] | None = None
    last_output: tuple[float, ...] | None = None
    last_ctx_id: str | int | None = None
    dirty: bool = True

    def should_reuse(self, sample: tuple[float, ...], ctx_id: str | int | None) -> bool:
        if self.dirty or self.last_output is None or self.last_input is None:
            return False
        if sample != self.last_input:
            return False
        if ctx_id is None:
            return self.last_ctx_id is None
        return ctx_id == self.last_ctx_id

    def update(self, *, sample: tuple[float, ...], output: tuple[float, ...], ctx_id: str | int | None) -> None:
        self.last_input = tuple(sample)
        self.last_output = tuple(output)
        self.last_ctx_id = ctx_id
        self.dirty = False

    def mark_dirty(self) -> None:
        self.dirty = True

class ExponentialTrendTracker:
    def __init__(self, *, alpha: float) -> None:
        self.alpha = float(alpha)
        self._level: float | None = None
        self._slope: float = 0.0

    def reset(self) -> None:
        self._level = None
        self._slope = 0.0

    def update_constant(self, value: float, *, alpha: float) -> float:
        if self._level is None:
            self._level = float(value)
        else:
            self._level = (1.0 - alpha) * self._level + alpha * float(value)
        return float(value - self._level)

    def update_linear(self, value: float, *, alpha: float) -> float:
        if self._level is None:
            self._level = float(value)
            self._slope = 0.0
            return 0.0

        predicted = self._level + self._slope
        residual = float(value) - predicted
        self._level = predicted + alpha * residual
        self._slope = self._slope + alpha * alpha * residual
        return float(value - self._level)

@dataclass(slots=True)
class SosFilterBank:
    sos: np.ndarray
    zi: list[np.ndarray]

    @classmethod
    def create(cls, *, sos: np.ndarray, dimension: int) -> "SosFilterBank":
        state_template = np.zeros((sos.shape[0], 2), dtype=np.float64)
        zi = [state_template.copy() for _ in range(dimension)]
        return cls(sos=sos, zi=zi)

    def update(self, sample: tuple[float, ...]) -> tuple[float, ...]:
        outputs: list[float] = []
        for index, value in enumerate(sample):
            y, updated_state = sosfilt(self.sos, np.asarray([value], dtype=np.float64), zi=self.zi[index])
            self.zi[index] = updated_state
            outputs.append(float(y[0]))
        return tuple(outputs)

def clamp_alpha(value: Any, *, default: float) -> float:
    numeric = parse_number(value)
    if numeric is None:
        return float(default)
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return float(numeric)

def clamp_positive(value: Any, *, default: float, minimum: float) -> float:
    numeric = parse_number(value)
    if numeric is None:
        return float(default)
    return max(minimum, float(numeric))

def clamp_order(value: Any, *, default: int) -> int:
    numeric = parse_number(value)
    if numeric is None:
        return int(default)
    return max(1, int(round(numeric)))

def sampling_hz_from_interval_ms(value: Any, *, default_interval_ms: float) -> float:
    interval_ms = parse_number(value)
    if interval_ms is None:
        interval_ms = float(default_interval_ms)
    interval_ms = max(1e-6, float(interval_ms))
    return 1000.0 / interval_ms

def design_lowpass(*, sampling_hz: float, cutoff: float, order: int) -> np.ndarray:
    nyquist = 0.5 * float(sampling_hz)
    normalized_cutoff = min(max(float(cutoff), 1e-6), nyquist - 1e-6)
    return cast(np.ndarray, butter(int(order), normalized_cutoff, btype="lowpass", fs=float(sampling_hz), output="sos"))

def design_highpass(*, sampling_hz: float, cutoff: float, order: int) -> np.ndarray:
    nyquist = 0.5 * float(sampling_hz)
    normalized_cutoff = min(max(float(cutoff), 1e-6), nyquist - 1e-6)
    return cast(np.ndarray, butter(int(order), normalized_cutoff, btype="highpass", fs=float(sampling_hz), output="sos"))

def design_bandpass(*, sampling_hz: float, low_cutoff: float, high_cutoff: float, order: int) -> np.ndarray:
    nyquist = 0.5 * float(sampling_hz)
    clipped_low = min(max(float(low_cutoff), 1e-6), nyquist - 2e-6)
    clipped_high = min(max(float(high_cutoff), clipped_low + 1e-6), nyquist - 1e-6)
    return cast(
        np.ndarray,
        butter(int(order), [clipped_low, clipped_high], btype="bandpass", fs=float(sampling_hz), output="sos"),
    )
