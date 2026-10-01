from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_PERIODIC_EPSILON = 1e-9


@dataclass
class LoopingLinearSampler:
    max_t: float
    times: np.ndarray
    values: np.ndarray
    last_segment_index: int | None = None

    @classmethod
    def from_points(cls, points: list[tuple[float, float]], *, max_t: float) -> "LoopingLinearSampler":
        deduped: dict[float, float] = {}
        for time_value, sample_value in points:
            if time_value < 0.0 or time_value > float(max_t):
                continue
            cycle_t = 0.0 if math.isclose(float(time_value), float(max_t), abs_tol=_PERIODIC_EPSILON) else float(time_value)
            deduped[cycle_t] = float(sample_value)
        ordered = sorted(deduped.items(), key=lambda item: item[0])
        if not ordered:
            times = np.zeros((0,), dtype=np.float64)
            values = np.zeros((0,), dtype=np.float64)
        else:
            times = np.asarray([item[0] for item in ordered], dtype=np.float64)
            values = np.asarray([item[1] for item in ordered], dtype=np.float64)
        return cls(max_t=float(max_t), times=times, values=values)

    def active_points(self) -> list[tuple[float, float]]:
        return list(zip(self.times.tolist(), self.values.tolist(), strict=True))

    def sample(self, t_value: float) -> float:
        if self.times.size == 0:
            return 0.0
        if self.times.size == 1:
            return float(self.values[0])

        wrapped_t = math.fmod(float(t_value), float(self.max_t))
        if wrapped_t < 0.0:
            wrapped_t += float(self.max_t)

        cached_index = self._match_cached_segment(wrapped_t)
        if cached_index is not None:
            return self._sample_on_segment(cached_index, wrapped_t)

        first_time = float(self.times[0])
        if wrapped_t < first_time:
            segment_index = int(self.times.size - 1)
        else:
            found_index = int(np.searchsorted(self.times, wrapped_t, side="right") - 1)
            segment_index = max(0, min(found_index, int(self.times.size - 1)))
        self.last_segment_index = segment_index
        return self._sample_on_segment(segment_index, wrapped_t)

    def _match_cached_segment(self, wrapped_t: float) -> int | None:
        cached_index = self.last_segment_index
        if cached_index is None:
            return None
        if not (0 <= cached_index < int(self.times.size)):
            self.last_segment_index = None
            return None
        first_time = float(self.times[0])
        start_time = float(self.times[cached_index])
        if cached_index + 1 < int(self.times.size):
            end_time = float(self.times[cached_index + 1])
            if start_time <= wrapped_t <= end_time:
                return cached_index
            return None
        if wrapped_t >= start_time or wrapped_t < first_time:
            return cached_index
        return None

    def _sample_on_segment(self, segment_index: int, wrapped_t: float) -> float:
        start_time = float(self.times[segment_index])
        start_value = float(self.values[segment_index])
        if segment_index + 1 < int(self.times.size):
            end_time = float(self.times[segment_index + 1])
            end_value = float(self.values[segment_index + 1])
            query_t = float(wrapped_t)
        else:
            end_time = float(self.times[0] + float(self.max_t))
            end_value = float(self.values[0])
            query_t = float(wrapped_t) if wrapped_t >= start_time else float(wrapped_t + float(self.max_t))
        duration = end_time - start_time
        if duration <= 0.0:
            return end_value
        alpha = (query_t - start_time) / duration
        return float(start_value + (end_value - start_value) * alpha)
