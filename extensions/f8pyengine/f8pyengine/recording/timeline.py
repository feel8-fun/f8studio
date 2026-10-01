from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .format import TIME_MODE_OFFSET_FROM_PLAY, TIME_MODE_RECORDED_EPOCH
from .models import DataSampleEvent, RecordingEvent, StateChangeEvent


@dataclass(frozen=True)
class TimelineEvent:
    event: RecordingEvent
    due_monotonic_s: float


@dataclass
class TimelineState:
    mode: Literal["recorded_epoch", "offset_from_play"]
    start_monotonic_s: float
    start_wall_ts_ms: int
    first_event_ts_ms: int


class TimelineCursor:
    def __init__(self, *, state: TimelineState) -> None:
        self._state = state

    def event_due_monotonic_s(self, event: DataSampleEvent | StateChangeEvent) -> float:
        if self._state.mode == TIME_MODE_RECORDED_EPOCH:
            event_ts_ms = _event_ts_ms(event)
            delta_ms = int(event_ts_ms) - int(self._state.start_wall_ts_ms)
            return float(self._state.start_monotonic_s) + (float(delta_ms) / 1000.0)
        delta_ms = int(event.relative_offset_ms)
        return float(self._state.start_monotonic_s) + (float(delta_ms) / 1000.0)

    def current_position_ms(self, *, now_monotonic_s: float) -> int:
        elapsed_ms = max(0.0, (float(now_monotonic_s) - float(self._state.start_monotonic_s)) * 1000.0)
        if self._state.mode == TIME_MODE_OFFSET_FROM_PLAY:
            return int(elapsed_ms)
        wall_ts_ms = int(self._state.start_wall_ts_ms + int(elapsed_ms))
        return max(0, int(wall_ts_ms - self._state.first_event_ts_ms))


def build_timeline_state(
    *,
    mode: Literal["recorded_epoch", "offset_from_play"],
    start_monotonic_s: float,
    start_wall_ts_ms: int,
    first_event_ts_ms: int,
) -> TimelineState:
    if mode not in (TIME_MODE_RECORDED_EPOCH, TIME_MODE_OFFSET_FROM_PLAY):
        raise ValueError(f"unsupported time mode: {mode}")
    return TimelineState(
        mode=mode,
        start_monotonic_s=float(start_monotonic_s),
        start_wall_ts_ms=int(start_wall_ts_ms),
        first_event_ts_ms=int(first_event_ts_ms),
    )


def _event_ts_ms(event: DataSampleEvent | StateChangeEvent) -> int:
    if isinstance(event, DataSampleEvent):
        return int(event.tick_ts_ms)
    return int(event.state_ts_ms)
