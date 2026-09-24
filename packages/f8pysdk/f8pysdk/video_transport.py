from __future__ import annotations

import struct
import time
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol

from .binary_stream_transport import ZenohLatestBinaryStreamTransport


ZENOH_VIDEO_FRAME_MAGIC = 0xF85A1001
ZENOH_VIDEO_FRAME_SCHEMA_VERSION = 2
VIDEO_FORMAT_BGRA32 = 1
VIDEO_FORMAT_FLOW2_F16 = 2
VIDEO_FORMAT_SCALAR1_F32 = 3

_ZENOH_VIDEO_FRAME_HEADER_STRUCT = struct.Struct("<8IQqQQ")


@dataclass
class LatestVideoFrame:
    width: int
    height: int
    pitch: int
    fmt: int
    frame_id: int
    ts_ms: int
    payload: memoryview
    stream_epoch: str = "00000000000000000000000000000000"
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def frame_bytes(self) -> int:
        return int(self.pitch) * int(self.height)

    def payload_bytes(self) -> bytes:
        if self._released:
            raise RuntimeError("video frame payload has been released")
        return bytes(self.payload)

    def release(self) -> None:
        if self._released:
            return
        self.payload.release()
        self._released = True

    def __enter__(self) -> "LatestVideoFrame":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


class LatestVideoFrameTransport(Protocol):
    def close(self) -> None: ...

    def set_min_sample_interval_ms(self, min_sample_interval_ms: int) -> None: ...

    def publish_frame(
        self,
        *,
        width: int,
        height: int,
        pitch: int,
        payload: bytes | bytearray | memoryview,
        fmt: int,
        ts_ms: int | None = None,
    ) -> None: ...

    def poll_latest(self) -> LatestVideoFrame | None: ...

    def wait_latest(self, timeout_ms: int) -> LatestVideoFrame | None: ...


def encode_zenoh_video_frame(
    *,
    width: int,
    height: int,
    pitch: int,
    payload: bytes | bytearray | memoryview,
    fmt: int,
    frame_id: int,
    ts_ms: int,
    stream_epoch: str = "00000000000000000000000000000000",
) -> bytes:
    width_i = int(width)
    height_i = int(height)
    pitch_i = int(pitch)
    fmt_i = int(fmt)
    frame_id_i = int(frame_id)
    ts_ms_i = int(ts_ms)
    try:
        epoch_int = UUID(hex=stream_epoch).int
    except ValueError as exc:
        raise ValueError("stream_epoch must be a 32-digit UUID hex value") from exc
    epoch_high = epoch_int >> 64
    epoch_low = epoch_int & 0xFFFF_FFFF_FFFF_FFFF
    if width_i <= 0 or height_i <= 0 or pitch_i <= 0:
        raise ValueError("width, height, and pitch must be positive")
    if fmt_i <= 0:
        raise ValueError("fmt must be positive")
    if frame_id_i <= 0:
        raise ValueError("frame_id must be positive")
    payload_view = memoryview(payload).cast("B")
    try:
        frame_bytes = pitch_i * height_i
        if len(payload_view) < frame_bytes:
            raise ValueError("payload is smaller than pitch * height")
        header = _ZENOH_VIDEO_FRAME_HEADER_STRUCT.pack(
            ZENOH_VIDEO_FRAME_MAGIC,
            ZENOH_VIDEO_FRAME_SCHEMA_VERSION,
            _ZENOH_VIDEO_FRAME_HEADER_STRUCT.size,
            width_i,
            height_i,
            pitch_i,
            fmt_i,
            frame_bytes,
            frame_id_i,
            ts_ms_i,
            epoch_high,
            epoch_low,
        )
        return header + bytes(payload_view[:frame_bytes])
    finally:
        payload_view.release()


def decode_zenoh_video_frame(raw: bytes | bytearray | memoryview) -> LatestVideoFrame | None:
    raw_view = memoryview(raw).cast("B")
    try:
        if len(raw_view) < _ZENOH_VIDEO_FRAME_HEADER_STRUCT.size:
            return None
        fields = _ZENOH_VIDEO_FRAME_HEADER_STRUCT.unpack_from(raw_view, 0)
        magic = int(fields[0])
        version = int(fields[1])
        header_bytes = int(fields[2])
        width = int(fields[3])
        height = int(fields[4])
        pitch = int(fields[5])
        fmt = int(fields[6])
        payload_bytes = int(fields[7])
        frame_id = int(fields[8])
        ts_ms = int(fields[9])
        epoch_high = int(fields[10])
        epoch_low = int(fields[11])
        if magic != ZENOH_VIDEO_FRAME_MAGIC or version != ZENOH_VIDEO_FRAME_SCHEMA_VERSION:
            return None
        if header_bytes < _ZENOH_VIDEO_FRAME_HEADER_STRUCT.size:
            return None
        if width <= 0 or height <= 0 or pitch <= 0 or fmt <= 0 or frame_id <= 0:
            return None
        if payload_bytes != pitch * height:
            return None
        if header_bytes + payload_bytes > len(raw_view):
            return None
        payload = raw_view[header_bytes : header_bytes + payload_bytes]
        return LatestVideoFrame(
            width=width,
            height=height,
            pitch=pitch,
            fmt=fmt,
            frame_id=frame_id,
            ts_ms=ts_ms,
            stream_epoch=f"{(epoch_high << 64) | epoch_low:032x}",
            payload=payload,
        )
    except (BufferError, TypeError, ValueError, struct.error):
        return None


class ZenohLatestVideoFrameTransport:
    def __init__(
        self,
        *,
        key_expr: str,
        session: Any | None = None,
        subscriber: Any | None = None,
        publisher: Any | None = None,
        raw_transport: ZenohLatestBinaryStreamTransport | None = None,
        min_sample_interval_ms: int = 0,
    ) -> None:
        key = str(key_expr or "").strip()
        if not key:
            raise ValueError("key_expr must be non-empty")
        self.key_expr = key
        if raw_transport is None:
            if session is None:
                raise ValueError("session is required when raw_transport is not provided")
            raw_transport = ZenohLatestBinaryStreamTransport(
                key_expr=key,
                session=session,
                subscriber=subscriber,
                publisher=publisher,
                log_context="video",
                min_sample_interval_ms=int(min_sample_interval_ms),
            )
        self._raw = raw_transport
        self._frame_id = 0
        self._stream_epoch = uuid4().hex

    @classmethod
    def open_publisher(
        cls,
        key_expr: str,
        *,
        config_path: str | None = None,
        connect: tuple[str, ...] = (),
        listen: tuple[str, ...] = (),
        shm_pool_bytes: int = 256 * 1024 * 1024,
    ) -> "ZenohLatestVideoFrameTransport":
        raw = ZenohLatestBinaryStreamTransport.open_publisher(
            key_expr,
            config_path=config_path,
            connect=connect,
            listen=listen,
            shm_pool_bytes=shm_pool_bytes,
            log_context="video",
        )
        return cls(key_expr=key_expr, raw_transport=raw)

    @classmethod
    def open_subscriber(
        cls,
        key_expr: str,
        *,
        config_path: str | None = None,
        connect: tuple[str, ...] = (),
        listen: tuple[str, ...] = (),
        shm_pool_bytes: int = 256 * 1024 * 1024,
        min_sample_interval_ms: int = 0,
    ) -> "ZenohLatestVideoFrameTransport":
        raw = ZenohLatestBinaryStreamTransport.open_subscriber(
            key_expr,
            config_path=config_path,
            connect=connect,
            listen=listen,
            shm_pool_bytes=shm_pool_bytes,
            log_context="video",
            min_sample_interval_ms=int(min_sample_interval_ms),
        )
        return cls(key_expr=key_expr, raw_transport=raw)

    @classmethod
    def open_pubsub(
        cls,
        key_expr: str,
        *,
        config_path: str | None = None,
        connect: tuple[str, ...] = (),
        listen: tuple[str, ...] = (),
        shm_pool_bytes: int = 256 * 1024 * 1024,
        min_sample_interval_ms: int = 0,
    ) -> "ZenohLatestVideoFrameTransport":
        raw = ZenohLatestBinaryStreamTransport.open_pubsub(
            key_expr,
            config_path=config_path,
            connect=connect,
            listen=listen,
            shm_pool_bytes=shm_pool_bytes,
            log_context="video",
            min_sample_interval_ms=int(min_sample_interval_ms),
        )
        return cls(key_expr=key_expr, raw_transport=raw)

    def close(self) -> None:
        self._raw.close()

    def set_min_sample_interval_ms(self, min_sample_interval_ms: int) -> None:
        self._raw.set_min_sample_interval_ms(int(min_sample_interval_ms))

    def publish_frame(
        self,
        *,
        width: int,
        height: int,
        pitch: int,
        payload: bytes | bytearray | memoryview,
        fmt: int,
        ts_ms: int | None = None,
    ) -> None:
        self._frame_id += 1
        raw = encode_zenoh_video_frame(
            width=width,
            height=height,
            pitch=pitch,
            payload=payload,
            fmt=fmt,
            frame_id=self._frame_id,
            ts_ms=int(ts_ms) if ts_ms is not None else int(time.time() * 1000),
            stream_epoch=self._stream_epoch,
        )
        self._raw.publish_raw(raw)

    def poll_latest(self) -> LatestVideoFrame | None:
        raw = self._raw.poll_latest_raw()
        if raw is None:
            return None
        return decode_zenoh_video_frame(raw)

    def wait_latest(self, timeout_ms: int) -> LatestVideoFrame | None:
        raw = self._raw.wait_latest_raw(timeout_ms)
        if raw is None:
            return None
        return decode_zenoh_video_frame(raw)

    def _on_sample(self, sample: Any) -> None:
        self._raw._on_sample(sample)


__all__ = [
    "LatestVideoFrame",
    "LatestVideoFrameTransport",
    "VIDEO_FORMAT_BGRA32",
    "VIDEO_FORMAT_FLOW2_F16",
    "VIDEO_FORMAT_SCALAR1_F32",
    "ZENOH_VIDEO_FRAME_MAGIC",
    "ZENOH_VIDEO_FRAME_SCHEMA_VERSION",
    "ZenohLatestVideoFrameTransport",
    "decode_zenoh_video_frame",
    "encode_zenoh_video_frame",
]
