from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass

from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32

from f8media_protocol.models import OverlayResult


OverlayKey = tuple[str, str, str, int, int]


@dataclass(frozen=True)
class OverlayCounters:
    matched: int
    wait_timeouts: int
    expired: int
    rejected: int


class OverlayStore:
    def __init__(self, *, max_results: int = 256, ttl_s: float = 2.0) -> None:
        if max_results <= 0 or ttl_s <= 0:
            raise ValueError("overlay limits must be positive")
        self._max_results = max_results
        self._ttl_s = ttl_s
        self._results: OrderedDict[OverlayKey, tuple[float, OverlayResult]] = OrderedDict()
        self._condition = asyncio.Condition()
        self._matched = 0
        self._wait_timeouts = 0
        self._expired = 0
        self._rejected = 0

    @property
    def size(self) -> int:
        return len(self._results)

    @property
    def counters(self) -> OverlayCounters:
        return OverlayCounters(
            matched=self._matched,
            wait_timeouts=self._wait_timeouts,
            expired=self._expired,
            rejected=self._rejected,
        )

    async def publish(self, result: OverlayResult) -> None:
        self._validate(result)
        key = self._key(result)
        now = asyncio.get_running_loop().time()
        async with self._condition:
            self._prune(now)
            self._results[key] = (now + self._ttl_s, result)
            self._results.move_to_end(key)
            while len(self._results) > self._max_results:
                self._results.popitem(last=False)
                self._expired += 1
            self._condition.notify_all()

    async def match(
        self,
        *,
        source: str,
        stream_id: str,
        stream_epoch: str,
        frame_id: int,
        capture_timestamp_ms: int,
        wait_budget_s: float,
    ) -> OverlayResult | None:
        key = (source, stream_id, stream_epoch, frame_id, capture_timestamp_ms)
        deadline = asyncio.get_running_loop().time() + max(0.0, wait_budget_s)
        async with self._condition:
            while True:
                now = asyncio.get_running_loop().time()
                self._prune(now)
                stored = self._results.pop(key, None)
                if stored is not None:
                    self._matched += 1
                    return stored[1]
                remaining = deadline - now
                if remaining <= 0:
                    self._wait_timeouts += 1
                    return None
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError:
                    self._wait_timeouts += 1
                    return None

    def _prune(self, now: float) -> None:
        expired_keys = [key for key, (expires_at, _) in self._results.items() if expires_at <= now]
        for key in expired_keys:
            del self._results[key]
            self._expired += 1

    def _validate(self, result: OverlayResult) -> None:
        if not result.source.strip() or not result.stream_id.strip() or not result.stream_epoch.strip():
            self._rejected += 1
            raise ValueError("overlay source, streamId, and streamEpoch must be non-empty")
        if result.frame_id < 0 or result.capture_timestamp_ms < 0:
            self._rejected += 1
            raise ValueError("overlay frameId and captureTimestampMs must be non-negative")
        for detection in result.detections:
            values = (detection.x, detection.y, detection.width, detection.height)
            if not all(math_value == math_value and abs(math_value) != float("inf") for math_value in values):
                self._rejected += 1
                raise ValueError("overlay box values must be finite")
            if detection.width <= 0 or detection.height <= 0:
                self._rejected += 1
                raise ValueError("overlay box width and height must be positive")
            if detection.x < 0 or detection.y < 0 or detection.x + detection.width > 1 or detection.y + detection.height > 1:
                self._rejected += 1
                raise ValueError("overlay boxes must use normalized coordinates inside [0, 1]")
            if detection.score is not None and not 0 <= detection.score <= 1:
                self._rejected += 1
                raise ValueError("overlay score must be between zero and one")

    @staticmethod
    def _key(result: OverlayResult) -> OverlayKey:
        return (
            result.source,
            result.stream_id,
            result.stream_epoch,
            result.frame_id,
            result.capture_timestamp_ms,
        )


def compose_overlay_bgra(
    *,
    width: int,
    height: int,
    pitch: int,
    pixel_format: int,
    payload: bytes,
    result: OverlayResult,
) -> bytes:
    if pixel_format != VIDEO_FORMAT_BGRA32:
        raise ValueError("exact overlay composition currently requires BGRA32 video")
    if pitch < width * 4 or len(payload) < pitch * height:
        raise ValueError("overlay video payload does not match dimensions and pitch")
    output = bytearray(payload)
    for detection in result.detections:
        left = min(width - 1, max(0, round(detection.x * width)))
        top = min(height - 1, max(0, round(detection.y * height)))
        right = min(width - 1, max(left, round((detection.x + detection.width) * width) - 1))
        bottom = min(height - 1, max(top, round((detection.y + detection.height) * height) - 1))
        thickness = min(3, max(1, min(width, height) // 180))
        for offset in range(thickness):
            y_top = min(bottom, top + offset)
            y_bottom = max(top, bottom - offset)
            for x in range(left, right + 1):
                _set_green_pixel(output, pitch, x, y_top)
                _set_green_pixel(output, pitch, x, y_bottom)
            x_left = min(right, left + offset)
            x_right = max(left, right - offset)
            for y in range(top, bottom + 1):
                _set_green_pixel(output, pitch, x_left, y)
                _set_green_pixel(output, pitch, x_right, y)
    return bytes(output)


def _set_green_pixel(output: bytearray, pitch: int, x: int, y: int) -> None:
    index = y * pitch + x * 4
    output[index : index + 4] = b"\x4c\xdc\x78\xff"


__all__ = ["OverlayCounters", "OverlayStore", "compose_overlay_bgra"]
