from __future__ import annotations

import msgspec

from f8pysdk.specs import F8JsonValue


MEDIA_API_VERSION = "f8media-api/1"


class MediaGatewayHealth(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    status: str
    service: str
    version: str
    protocol_version: str
    gateway_epoch: str
    process_id: int


class MediaSessionOffer(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    source: str
    quality: str
    sdp: str
    type: str = "offer"
    overlay: bool = False


class MediaSessionAnswer(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    session_id: str
    source: str
    quality: str
    sdp: str
    type: str
    max_width: int
    max_height: int
    max_fps: int
    overlay: bool = False


class MediaSample(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    source: str
    format: str
    frame_id: int
    ts_ms: int
    width: int
    height: int
    x: int
    y: int
    finite: bool
    value: F8JsonValue
    stream_id: str = ""
    stream_epoch: str = ""


class AudioSessionOffer(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    source: str
    sdp: str
    type: str = "offer"


class AudioSessionAnswer(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    session_id: str
    source: str
    sdp: str
    type: str
    sample_rate: int
    channels: int
    transport_policy: str


class OverlayDetection(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    x: float
    y: float
    width: float
    height: float
    label: str = ""
    score: float | None = None


class OverlayResult(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    source: str
    stream_id: str
    stream_epoch: str
    frame_id: int
    capture_timestamp_ms: int
    detections: tuple[OverlayDetection, ...]


class MediaFrameMapping(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    session_id: str
    media_timestamp: int
    source: str
    stream_id: str
    stream_epoch: str
    frame_id: int
    capture_timestamp_ms: int
    sent_timestamp_ms: int


class MediaMetrics(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    video_sessions: int
    video_sources: int
    audio_sessions: int
    audio_sources: int
    overlay_matched: int
    overlay_wait_timeouts: int
    overlay_expired: int
    overlay_rejected: int
    audio_dropped_chunks: int


class MediaErrorResponse(msgspec.Struct, frozen=True, kw_only=True):
    detail: F8JsonValue


__all__ = [
    "AudioSessionAnswer",
    "AudioSessionOffer",
    "MEDIA_API_VERSION",
    "MediaErrorResponse",
    "MediaFrameMapping",
    "MediaGatewayHealth",
    "MediaMetrics",
    "MediaSample",
    "MediaSessionAnswer",
    "MediaSessionOffer",
    "OverlayDetection",
    "OverlayResult",
]
