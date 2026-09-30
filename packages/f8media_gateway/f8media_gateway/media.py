from __future__ import annotations

from .session_manager import SessionManager

import asyncio
import logging
import math
import struct
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Literal, Protocol
from uuid import uuid4

from aiortc import RTCPeerConnection, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
import numpy as np

from f8pysdk.video_transport import (
    VIDEO_FORMAT_BGRA32,
    VIDEO_FORMAT_FLOW2_F16,
    VIDEO_FORMAT_SCALAR1_F32,
    LatestVideoFrame,
    ZenohLatestVideoFrameTransport,
)
from f8pysdk.specs import F8JsonValue

from f8media_protocol.models import (
    MediaInputError,
    MediaFrameMapping,
    MediaSample,
    MediaSessionAnswer,
    MediaSessionOffer,
    OverlayResult,
)
from .overlay import OverlayStore, compose_overlay_bgra
from f8pysdk.binary_stream_transport import SharedStreamSession


logger = logging.getLogger(__name__)
VIDEO_TIME_BASE = Fraction(1, 90_000)


@dataclass(frozen=True)
class MediaQuality:
    name: Literal["main", "thumbnail"]
    max_width: int
    max_height: int
    max_fps: int


MEDIA_QUALITIES = {
    "thumbnail": MediaQuality(name="thumbnail", max_width=640, max_height=360, max_fps=30),
    "main": MediaQuality(name="main", max_width=1920, max_height=1080, max_fps=30),
}


@dataclass(frozen=True)
class RawVideoFrame:
    width: int
    height: int
    pitch: int
    format: int
    frame_id: int
    ts_ms: int
    payload: bytes
    stream_id: str = ""
    stream_epoch: str = ""


@dataclass(frozen=True)
class PreparedVideoFrame:
    width: int
    height: int
    source_frame_id: int
    source_ts_ms: int
    stream_id: str
    stream_epoch: str
    plane_payloads: tuple[bytes, ...]

    def instantiate(self) -> VideoFrame:
        frame = VideoFrame(width=self.width, height=self.height, format="yuv420p")
        if len(frame.planes) != len(self.plane_payloads):
            raise RuntimeError("cached YUV plane count does not match target frame")
        for index, payload in enumerate(self.plane_payloads):
            frame.planes[index].update(payload)
        return frame


class AsyncFrameProducer(Protocol):
    async def read(self) -> RawVideoFrame | None: ...

    async def close(self) -> None: ...


class SyntheticFrameProducer:
    def __init__(self, *, width: int = 640, height: int = 360, fps: int = 30) -> None:
        self._width = width
        self._height = height
        self._period_s = 1.0 / fps
        self._frame_id = 0
        self._closed = False

    async def read(self) -> RawVideoFrame | None:
        if self._closed:
            return None
        await asyncio.sleep(self._period_s)
        self._frame_id += 1
        captured_at_ms = int(time.time() * 1_000)
        phase = self._frame_id * 13 % 256
        first = bytes((phase, 40, 220, 255)) * (self._width // 2)
        second = bytes((255 - phase, 210, 35, 255)) * (self._width - self._width // 2)
        row = bytearray(first + second)
        marker_width = max(12, self._width // 24)
        marker_start = self._frame_id * 37 % (self._width - marker_width)
        marker_color = bytes((245, 245, 245, 255)) if self._frame_id % 2 else bytes((15, 15, 15, 255))
        row[marker_start * 4 : (marker_start + marker_width) * 4] = marker_color * marker_width
        barcode_block = max(4, min(16, self._width // 48))
        timestamp_modulo = captured_at_ms & 0xFFFF
        barcode_row = bytearray(row)
        for bit_index in range(16):
            value = 240 if timestamp_modulo & (1 << bit_index) else 16
            pixel = bytes((value, value, value, 255))
            x_start = bit_index * barcode_block
            offset = x_start * 4
            barcode_row[offset : offset + barcode_block * 4] = pixel * barcode_block
        barcode_height = min(barcode_block, self._height)
        payload = bytes(barcode_row) * barcode_height + bytes(row) * (self._height - barcode_height)
        return RawVideoFrame(
            width=self._width,
            height=self._height,
            pitch=self._width * 4,
            format=VIDEO_FORMAT_BGRA32,
            frame_id=self._frame_id,
            ts_ms=captured_at_ms,
            payload=payload,
        )

    async def close(self) -> None:
        self._closed = True


class ZenohFrameProducer:
    def __init__(self, source: str, session: SharedStreamSession | None = None) -> None:
        self._transport = (
            ZenohLatestVideoFrameTransport.open_subscriber(source)
            if session is None
            else ZenohLatestVideoFrameTransport(
                key_expr=source, raw_transport=session.subscribe(source, log_context="video", max_pending_samples=1)
            )
        )
        self._closed = False

    async def read(self) -> RawVideoFrame | None:
        if self._closed:
            return None
        frame = await asyncio.to_thread(self._transport.wait_latest, 250)
        if frame is None:
            return None
        return self._copy_frame(frame)

    @staticmethod
    def _copy_frame(frame: LatestVideoFrame) -> RawVideoFrame:
        try:
            return RawVideoFrame(
                width=frame.width,
                height=frame.height,
                pitch=frame.pitch,
                format=frame.fmt,
                frame_id=frame.frame_id,
                ts_ms=frame.ts_ms,
                payload=frame.payload_bytes(),
                stream_epoch=frame.stream_epoch,
            )
        finally:
            frame.release()

    async def close(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._transport.close)


def create_frame_producer(source: str, session: SharedStreamSession | None = None) -> AsyncFrameProducer:
    normalized = source.strip()
    if normalized == "synthetic://bars":
        return SyntheticFrameProducer()
    if normalized == "synthetic://bars-1080p":
        return SyntheticFrameProducer(width=1920, height=1080, fps=30)
    if not normalized.startswith("f8/") or len(normalized) > 512:
        raise MediaInputError("media source must be synthetic://bars, synthetic://bars-1080p, or an f8/ Zenoh key")
    return ZenohFrameProducer(normalized, session)


@dataclass
class LatestFrameHub:
    source: str
    producer: AsyncFrameProducer
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition, init=False)
    _latest: RawVideoFrame | None = field(default=None, init=False)
    _version: int = field(default=0, init=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False)
    _closed: bool = field(default=False, init=False)
    _prepared: dict[str, tuple[float, PreparedVideoFrame]] = field(default_factory=dict, init=False)
    _prepare_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _stream_epoch: str = field(default_factory=lambda: uuid4().hex, init=False)
    _last_frame_id: int | None = field(default=None, init=False)
    _history: deque[tuple[float, RawVideoFrame]] = field(default_factory=lambda: deque(maxlen=8), init=False)

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"media-source:{self.source}")

    async def _run(self) -> None:
        try:
            while not self._closed:
                frame = await self.producer.read()
                if frame is None:
                    continue
                if self._last_frame_id is not None and frame.frame_id <= self._last_frame_id:
                    self._stream_epoch = uuid4().hex
                    self._prepared.clear()
                    self._history.clear()
                self._last_frame_id = frame.frame_id
                frame = replace(
                    frame,
                    stream_id=frame.stream_id or self.source,
                    stream_epoch=frame.stream_epoch or self._stream_epoch,
                )
                async with self._condition:
                    self._latest = frame
                    self._version += 1
                    self._history.append((asyncio.get_running_loop().time(), frame))
                    self._prune_history()
                    self._condition.notify_all()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("media source failed source=%s", self.source)
        finally:
            async with self._condition:
                self._closed = True
                self._condition.notify_all()

    async def next_frame(self, after_version: int) -> tuple[int, RawVideoFrame]:
        async with self._condition:
            await self._condition.wait_for(lambda: self._version > after_version or self._closed)
            if self._latest is None or self._closed:
                raise MediaStreamError
            return self._version, self._latest

    def _prune_history(self) -> None:
        oldest = asyncio.get_running_loop().time() - 0.3
        while self._history and self._history[0][0] < oldest:
            self._history.popleft()

    @property
    def history_size(self) -> int:
        return len(self._history)

    async def prepare_frame(
        self,
        raw: RawVideoFrame,
        quality: MediaQuality,
        overlay: OverlayResult | None = None,
    ) -> PreparedVideoFrame:
        now = asyncio.get_running_loop().time()
        async with self._prepare_lock:
            cache_key = f"{quality.name}:{'overlay' if overlay is not None else 'plain'}"
            cached = self._prepared.get(cache_key)
            target_size = _fit_size(raw.width, raw.height, quality)
            if cached is not None and cached[0] > now:
                prepared = cached[1]
                same_dimensions = (prepared.width, prepared.height) == target_size
                same_source_frame = prepared.source_frame_id == raw.frame_id
                reusable_thumbnail = quality.name == "thumbnail" and overlay is None
                if same_dimensions and (reusable_thumbnail or same_source_frame):
                    return prepared
            if raw.format == VIDEO_FORMAT_BGRA32:
                prepared = _prepare_video_frame(raw, quality, overlay)
            else:
                prepared = await asyncio.to_thread(_prepare_video_frame, raw, quality, overlay)
            expires_at = asyncio.get_running_loop().time() + 1.0 / quality.max_fps
            self._prepared[cache_key] = (expires_at, prepared)
            return prepared

    async def close(self) -> None:
        self._closed = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.producer.close()
        async with self._condition:
            self._condition.notify_all()


def bgra_frame(raw: RawVideoFrame) -> VideoFrame:
    if raw.format != VIDEO_FORMAT_BGRA32:
        raise ValueError(f"WebRTC video requires BGRA32 input, received format {raw.format}")
    row_bytes = raw.width * 4
    if raw.width <= 0 or raw.height <= 0 or raw.pitch < row_bytes:
        raise ValueError("invalid BGRA frame dimensions or pitch")
    required = raw.pitch * raw.height
    if len(raw.payload) < required:
        raise ValueError("BGRA payload is smaller than pitch * height")

    frame = VideoFrame(width=raw.width, height=raw.height, format="bgra")
    plane = frame.planes[0]
    if raw.pitch == row_bytes and plane.line_size == row_bytes:
        plane.update(raw.payload[:required])
        return frame
    packed = bytearray(plane.buffer_size)
    for row_index in range(raw.height):
        source_start = row_index * raw.pitch
        target_start = row_index * plane.line_size
        packed[target_start : target_start + row_bytes] = raw.payload[source_start : source_start + row_bytes]
    plane.update(bytes(packed))
    return frame


def _flow_preview_frame(raw: RawVideoFrame, *, magnitude_scale: float = 20.0) -> VideoFrame:
    row_bytes = raw.width * 4
    if raw.width <= 0 or raw.height <= 0 or raw.pitch < row_bytes:
        raise ValueError("invalid FLOW2_F16 frame dimensions or pitch")
    if len(raw.payload) < raw.pitch * raw.height:
        raise ValueError("FLOW2_F16 payload is smaller than pitch * height")
    if magnitude_scale <= 0 or not math.isfinite(magnitude_scale):
        raise ValueError("magnitude_scale must be finite and positive")
    flow = np.ndarray((raw.height, raw.width, 2), dtype="<f2", buffer=raw.payload, strides=(raw.pitch, 4, 2)).astype(
        np.float64
    )
    finite = np.isfinite(flow).all(axis=2)
    flow[~finite] = 0
    dx, dy = flow[:, :, 0], flow[:, :, 1]
    hue = (np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * 6
    value = np.minimum(1, np.hypot(dx, dy) / magnitude_scale)
    # Piecewise-linear HSV channels avoid six full-size selection arrays.
    red = np.clip(np.abs(hue - 3) - 1, 0, 1) * value
    green = np.clip(2 - np.abs(hue - 2), 0, 1) * value
    blue = np.clip(2 - np.abs(hue - 4), 0, 1) * value
    output = np.empty((raw.height, raw.width, 4), dtype=np.uint8)
    output[:, :, :3] = np.rint(np.stack((blue, green, red), axis=2) * 255).astype(np.uint8)
    output[~finite, :3] = 0
    output[:, :, 3] = 255
    return bgra_frame(
        RawVideoFrame(
            width=raw.width,
            height=raw.height,
            pitch=row_bytes,
            format=VIDEO_FORMAT_BGRA32,
            frame_id=raw.frame_id,
            ts_ms=raw.ts_ms,
            payload=output.tobytes(),
        )
    )


def _scalar_preview_frame(raw: RawVideoFrame) -> VideoFrame:
    row_bytes = raw.width * 4
    if raw.width <= 0 or raw.height <= 0 or raw.pitch < row_bytes:
        raise ValueError("invalid SCALAR1_F32 frame dimensions or pitch")
    if len(raw.payload) < raw.pitch * raw.height:
        raise ValueError("SCALAR1_F32 payload is smaller than pitch * height")

    sample_stride = max(1, math.isqrt(max(1, raw.width * raw.height // 65_536)))
    values = np.ndarray((raw.height, raw.width), dtype="<f4", buffer=raw.payload, strides=(raw.pitch, 4)).astype(
        np.float64
    )
    sampled = values[::sample_stride, ::sample_stride]
    sampled = np.sort(sampled[np.isfinite(sampled)])
    if sampled.size:
        lower = sampled[round((sampled.size - 1) * 0.02)]
        upper = sampled[round((sampled.size - 1) * 0.98)]
        if upper <= lower:
            lower, upper = sampled[0], sampled[-1]
    else:
        lower, upper = 0.0, 0.0
    span = upper - lower
    finite = np.isfinite(values)
    values[~finite] = lower
    normalized = np.clip((values - lower) / span, 0, 1) if span > 0 else np.zeros_like(values)
    red = np.clip(1.6 * normalized - 0.2, 0, 1)
    green = np.clip(1.2 - np.abs(2 * normalized - 1) * 1.6, 0, 1)
    blue = np.clip(1.2 * (1 - normalized) - 0.1, 0, 1)
    output = np.empty((raw.height, raw.width, 4), dtype=np.uint8)
    output[:, :, :3] = np.rint(np.stack((blue, green, red), axis=2) * 255).astype(np.uint8)
    output[~finite, :3] = 0
    output[:, :, 3] = 255
    return bgra_frame(
        RawVideoFrame(
            width=raw.width,
            height=raw.height,
            pitch=row_bytes,
            format=VIDEO_FORMAT_BGRA32,
            frame_id=raw.frame_id,
            ts_ms=raw.ts_ms,
            payload=output.tobytes(),
        )
    )


def preview_frame(raw: RawVideoFrame) -> VideoFrame:
    if raw.format == VIDEO_FORMAT_BGRA32:
        return bgra_frame(raw)
    if raw.format == VIDEO_FORMAT_FLOW2_F16:
        return _flow_preview_frame(raw)
    if raw.format == VIDEO_FORMAT_SCALAR1_F32:
        return _scalar_preview_frame(raw)
    raise ValueError(f"unsupported video frame format {raw.format}")


def sample_raw_frame(source: str, raw: RawVideoFrame, *, x: int, y: int) -> MediaSample:
    if x < 0 or y < 0 or x >= raw.width or y >= raw.height:
        raise MediaInputError(f"sample coordinate ({x}, {y}) is outside {raw.width}x{raw.height} frame")
    if raw.format == VIDEO_FORMAT_BGRA32:
        bytes_per_pixel = 4
        format_name = "bgra32"
    elif raw.format == VIDEO_FORMAT_FLOW2_F16:
        bytes_per_pixel = 4
        format_name = "flow2_f16"
    elif raw.format == VIDEO_FORMAT_SCALAR1_F32:
        bytes_per_pixel = 4
        format_name = "scalar1_f32"
    else:
        raise ValueError(f"unsupported video frame format {raw.format}")
    row_bytes = raw.width * bytes_per_pixel
    if raw.pitch < row_bytes or len(raw.payload) < raw.pitch * raw.height:
        raise ValueError("video frame payload does not match dimensions and pitch")
    offset = y * raw.pitch + x * bytes_per_pixel
    finite = True
    value: F8JsonValue
    if raw.format == VIDEO_FORMAT_BGRA32:
        blue, green, red, alpha = struct.unpack_from("<4B", raw.payload, offset)
        value = {"red": red, "green": green, "blue": blue, "alpha": alpha}
    elif raw.format == VIDEO_FORMAT_FLOW2_F16:
        dx, dy = struct.unpack_from("<ee", raw.payload, offset)
        finite = math.isfinite(dx) and math.isfinite(dy)
        value = {"dx": dx if math.isfinite(dx) else None, "dy": dy if math.isfinite(dy) else None}
    else:
        scalar = struct.unpack_from("<f", raw.payload, offset)[0]
        finite = math.isfinite(scalar)
        value = scalar if finite else None
    return MediaSample(
        source=source,
        format=format_name,
        frame_id=raw.frame_id,
        ts_ms=raw.ts_ms,
        width=raw.width,
        height=raw.height,
        x=x,
        y=y,
        finite=finite,
        value=value,
        stream_id=raw.stream_id,
        stream_epoch=raw.stream_epoch,
    )


def _fit_size(width: int, height: int, quality: MediaQuality) -> tuple[int, int]:
    scale = min(1.0, quality.max_width / width, quality.max_height / height)
    target_width = max(2, int(width * scale) // 2 * 2)
    target_height = max(2, int(height * scale) // 2 * 2)
    return target_width, target_height


def _prepare_video_frame(
    raw: RawVideoFrame,
    quality: MediaQuality,
    overlay: OverlayResult | None = None,
) -> PreparedVideoFrame:
    render_raw = raw
    if overlay is not None:
        if (
            overlay.stream_id != raw.stream_id
            or overlay.stream_epoch != raw.stream_epoch
            or overlay.frame_id != raw.frame_id
            or overlay.capture_timestamp_ms != raw.ts_ms
        ):
            raise ValueError("overlay identity does not match video frame")
        render_raw = replace(
            raw,
            payload=compose_overlay_bgra(
                width=raw.width,
                height=raw.height,
                pitch=raw.pitch,
                pixel_format=raw.format,
                payload=raw.payload,
                result=overlay,
            ),
        )
    frame = preview_frame(render_raw)
    width, height = _fit_size(raw.width, raw.height, quality)
    prepared_frame = frame.reformat(width=width, height=height, format="yuv420p")
    return PreparedVideoFrame(
        width=width,
        height=height,
        source_frame_id=raw.frame_id,
        source_ts_ms=raw.ts_ms,
        stream_id=raw.stream_id,
        stream_epoch=raw.stream_epoch,
        plane_payloads=tuple(bytes(plane) for plane in prepared_frame.planes),
    )


class LatestFrameVideoTrack(VideoStreamTrack):
    def __init__(
        self,
        hub: LatestFrameHub,
        quality: MediaQuality,
        *,
        session_id: str = "",
        overlay_store: OverlayStore | None = None,
        overlay_wait_budget_s: float = 0.05,
    ) -> None:
        super().__init__()
        self._hub = hub
        self._quality = quality
        self._version = 0
        self._first_source_ts_ms: int | None = None
        self._last_pts = -1
        self._next_frame_at = 0.0
        self._session_id = session_id
        self._overlay_store = overlay_store
        self._overlay_wait_budget_s = overlay_wait_budget_s
        self._mappings: deque[MediaFrameMapping] = deque(maxlen=180)

    async def recv(self) -> VideoFrame:
        delay = self._next_frame_at - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)
        self._version, raw = await self._hub.next_frame(self._version)
        self._next_frame_at = asyncio.get_running_loop().time() + 1.0 / self._quality.max_fps

        overlay: OverlayResult | None = None
        if self._overlay_store is not None:
            overlay = await self._overlay_store.match(
                source=self._hub.source,
                stream_id=raw.stream_id,
                stream_epoch=raw.stream_epoch,
                frame_id=raw.frame_id,
                capture_timestamp_ms=raw.ts_ms,
                wait_budget_s=self._overlay_wait_budget_s,
            )
        prepared = await self._hub.prepare_frame(raw, self._quality, overlay)
        frame = prepared.instantiate()

        if self._first_source_ts_ms is None:
            self._first_source_ts_ms = prepared.source_ts_ms
        source_pts = max(0, prepared.source_ts_ms - self._first_source_ts_ms) * 90
        self._last_pts = max(self._last_pts + 1, source_pts)
        frame.pts = self._last_pts
        frame.time_base = VIDEO_TIME_BASE
        self._mappings.append(
            MediaFrameMapping(
                session_id=self._session_id,
                media_timestamp=self._last_pts,
                source=self._hub.source,
                stream_id=prepared.stream_id,
                stream_epoch=prepared.stream_epoch,
                frame_id=prepared.source_frame_id,
                capture_timestamp_ms=prepared.source_ts_ms,
                sent_timestamp_ms=int(time.time() * 1_000),
            )
        )
        return frame

    def mapping(self, media_timestamp: int) -> MediaFrameMapping | None:
        for mapping in reversed(self._mappings):
            if mapping.media_timestamp == media_timestamp:
                return mapping
        return None


@dataclass
class MediaSession:
    session_id: str
    hub: LatestFrameHub
    source: str
    quality: MediaQuality
    peer: RTCPeerConnection
    track: LatestFrameVideoTrack
    overlay: bool


class MediaSessionManager(SessionManager[LatestFrameHub, MediaSession]):
    def __init__(
        self,
        *,
        producer_factory: Callable[[str], AsyncFrameProducer] = create_frame_producer,
        disconnected_grace_s: float = 10.0,
    ) -> None:
        self._producer_factory = producer_factory
        super().__init__(hub_factory=self._make_hub, disconnected_grace_s=disconnected_grace_s, kind="video")
        self.overlays = OverlayStore()

    async def create(self, offer: MediaSessionOffer) -> MediaSessionAnswer:
        source = offer.source.strip()
        quality = MEDIA_QUALITIES.get(offer.quality)
        if quality is None:
            raise MediaInputError("media quality must be thumbnail or main")

        def build(hub: LatestFrameHub, session_id: str, peer: RTCPeerConnection) -> MediaSession:
            track = LatestFrameVideoTrack(
                hub, quality, session_id=session_id, overlay_store=self.overlays if offer.overlay else None
            )
            return MediaSession(
                hub=hub,
                session_id=session_id,
                source=source,
                peer=peer,
                track=track,
                quality=quality,
                overlay=offer.overlay,
            )

        session, local = await self._create_session(source=source, sdp=offer.sdp, offer_type=offer.type, build=build)
        return MediaSessionAnswer(
            session_id=session.session_id,
            source=source,
            quality=quality.name,
            sdp=local.sdp,
            type="answer",
            max_width=quality.max_width,
            max_height=quality.max_height,
            max_fps=quality.max_fps,
            overlay=offer.overlay,
        )

    async def publish_overlay(self, result: OverlayResult) -> None:
        await self.overlays.publish(result)

    async def frame_mapping(self, session_id: str, media_timestamp: int) -> MediaFrameMapping | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.track.mapping(media_timestamp)

    async def sample(self, source: str, *, x: int, y: int, timeout_s: float = 2.0) -> MediaSample:
        normalized_source = source.strip()
        hub = await self._acquire_hub(normalized_source)
        try:
            _, raw = await asyncio.wait_for(hub.next_frame(0), timeout=timeout_s)
            return sample_raw_frame(normalized_source, raw, x=x, y=y)
        finally:
            await self._release_hub(hub)

    def _make_hub(self, source: str) -> LatestFrameHub:
        return LatestFrameHub(source=source, producer=self._producer_factory(source))


__all__ = [
    "LatestFrameHub",
    "LatestFrameVideoTrack",
    "MEDIA_QUALITIES",
    "MediaSessionManager",
    "PreparedVideoFrame",
    "RawVideoFrame",
    "SyntheticFrameProducer",
    "bgra_frame",
    "preview_frame",
    "sample_raw_frame",
]
