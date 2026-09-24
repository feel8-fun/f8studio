from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol

from .binary_stream_transport import ZenohLatestBinaryStreamTransport


SAMPLE_FORMAT_F32LE = 1
ZENOH_AUDIO_CHUNK_MAGIC = 0xF85A2001
ZENOH_AUDIO_CHUNK_SCHEMA_VERSION = 1

_ZENOH_AUDIO_CHUNK_HEADER_STRUCT = struct.Struct("<9IQQq")


@dataclass
class LatestAudioChunk:
    sample_rate: int
    channels: int
    fmt: int
    frames: int
    bytes_per_frame: int
    seq: int
    frame_index: int
    ts_ms: int
    payload: memoryview
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def payload_bytes(self) -> int:
        return int(self.frames) * int(self.bytes_per_frame)

    def payload_copy(self) -> bytes:
        if self._released:
            raise RuntimeError("audio chunk payload has been released")
        return bytes(self.payload)

    def release(self) -> None:
        if self._released:
            return
        self.payload.release()
        self._released = True

    def __enter__(self) -> "LatestAudioChunk":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


class LatestAudioChunkTransport(Protocol):
    def close(self) -> None: ...

    def publish_chunk(
        self,
        *,
        sample_rate: int,
        channels: int,
        payload: bytes | bytearray | memoryview,
        frames: int,
        fmt: int = SAMPLE_FORMAT_F32LE,
        ts_ms: int | None = None,
    ) -> None: ...

    def poll_latest(self) -> LatestAudioChunk | None: ...

    def wait_latest(self, timeout_ms: int) -> LatestAudioChunk | None: ...


def encode_zenoh_audio_chunk(
    *,
    sample_rate: int,
    channels: int,
    payload: bytes | bytearray | memoryview,
    frames: int,
    fmt: int,
    seq: int,
    frame_index: int,
    ts_ms: int,
) -> bytes:
    sample_rate_i = int(sample_rate)
    channels_i = int(channels)
    fmt_i = int(fmt)
    frames_i = int(frames)
    seq_i = int(seq)
    frame_index_i = int(frame_index)
    ts_ms_i = int(ts_ms)
    if sample_rate_i <= 0 or channels_i <= 0 or frames_i <= 0:
        raise ValueError("sample_rate, channels, and frames must be positive")
    if fmt_i <= 0:
        raise ValueError("fmt must be positive")
    if seq_i <= 0:
        raise ValueError("seq must be positive")
    bytes_per_frame = 4 * channels_i if fmt_i == int(SAMPLE_FORMAT_F32LE) else 0
    if bytes_per_frame <= 0:
        raise ValueError("only f32le audio chunks are supported by schema v1")
    payload_view = memoryview(payload).cast("B")
    try:
        payload_bytes = frames_i * bytes_per_frame
        if len(payload_view) < payload_bytes:
            raise ValueError("payload is smaller than frames * bytes_per_frame")
        header = _ZENOH_AUDIO_CHUNK_HEADER_STRUCT.pack(
            ZENOH_AUDIO_CHUNK_MAGIC,
            ZENOH_AUDIO_CHUNK_SCHEMA_VERSION,
            _ZENOH_AUDIO_CHUNK_HEADER_STRUCT.size,
            sample_rate_i,
            channels_i,
            fmt_i,
            frames_i,
            bytes_per_frame,
            payload_bytes,
            seq_i,
            frame_index_i,
            ts_ms_i,
        )
        return header + bytes(payload_view[:payload_bytes])
    finally:
        payload_view.release()


def decode_zenoh_audio_chunk(raw: bytes | bytearray | memoryview) -> LatestAudioChunk | None:
    raw_view = memoryview(raw).cast("B")
    try:
        if len(raw_view) < _ZENOH_AUDIO_CHUNK_HEADER_STRUCT.size:
            return None
        fields = _ZENOH_AUDIO_CHUNK_HEADER_STRUCT.unpack_from(raw_view, 0)
        magic = int(fields[0])
        version = int(fields[1])
        header_bytes = int(fields[2])
        sample_rate = int(fields[3])
        channels = int(fields[4])
        fmt = int(fields[5])
        frames = int(fields[6])
        bytes_per_frame = int(fields[7])
        payload_bytes = int(fields[8])
        seq = int(fields[9])
        frame_index = int(fields[10])
        ts_ms = int(fields[11])
        if magic != ZENOH_AUDIO_CHUNK_MAGIC or version != ZENOH_AUDIO_CHUNK_SCHEMA_VERSION:
            return None
        if header_bytes < _ZENOH_AUDIO_CHUNK_HEADER_STRUCT.size:
            return None
        if sample_rate <= 0 or channels <= 0 or fmt <= 0 or frames <= 0 or bytes_per_frame <= 0 or seq <= 0:
            return None
        if payload_bytes != frames * bytes_per_frame:
            return None
        if header_bytes + payload_bytes > len(raw_view):
            return None
        payload = raw_view[header_bytes : header_bytes + payload_bytes]
        return LatestAudioChunk(
            sample_rate=sample_rate,
            channels=channels,
            fmt=fmt,
            frames=frames,
            bytes_per_frame=bytes_per_frame,
            seq=seq,
            frame_index=frame_index,
            ts_ms=ts_ms,
            payload=payload,
        )
    except (BufferError, TypeError, ValueError, struct.error):
        return None


class ZenohLatestAudioChunkTransport:
    def __init__(
        self,
        *,
        key_expr: str,
        session: Any | None = None,
        subscriber: Any | None = None,
        publisher: Any | None = None,
        raw_transport: ZenohLatestBinaryStreamTransport | None = None,
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
                log_context="audio",
            )
        self._raw = raw_transport
        self._seq = 0
        self._frame_index = 0

    @classmethod
    def open_publisher(
        cls,
        key_expr: str,
        *,
        config_path: str | None = None,
        connect: tuple[str, ...] = (),
        listen: tuple[str, ...] = (),
        shm_pool_bytes: int = 256 * 1024 * 1024,
    ) -> "ZenohLatestAudioChunkTransport":
        raw = ZenohLatestBinaryStreamTransport.open_publisher(
            key_expr,
            config_path=config_path,
            connect=connect,
            listen=listen,
            shm_pool_bytes=shm_pool_bytes,
            log_context="audio",
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
    ) -> "ZenohLatestAudioChunkTransport":
        raw = ZenohLatestBinaryStreamTransport.open_subscriber(
            key_expr,
            config_path=config_path,
            connect=connect,
            listen=listen,
            shm_pool_bytes=shm_pool_bytes,
            log_context="audio",
            max_pending_samples=16,
        )
        return cls(key_expr=key_expr, raw_transport=raw)

    def close(self) -> None:
        self._raw.close()

    def publish_chunk(
        self,
        *,
        sample_rate: int,
        channels: int,
        payload: bytes | bytearray | memoryview,
        frames: int,
        fmt: int = SAMPLE_FORMAT_F32LE,
        ts_ms: int | None = None,
    ) -> None:
        frames_i = int(frames)
        self._seq += 1
        self._frame_index += max(0, frames_i)
        raw = encode_zenoh_audio_chunk(
            sample_rate=int(sample_rate),
            channels=int(channels),
            payload=payload,
            frames=frames_i,
            fmt=int(fmt),
            seq=int(self._seq),
            frame_index=int(self._frame_index),
            ts_ms=int(ts_ms) if ts_ms is not None else int(time.time() * 1000),
        )
        self._raw.publish_raw(raw)

    def poll_latest(self) -> LatestAudioChunk | None:
        raw = self._raw.poll_latest_raw()
        if raw is None:
            return None
        return decode_zenoh_audio_chunk(raw)

    def wait_latest(self, timeout_ms: int) -> LatestAudioChunk | None:
        raw = self._raw.wait_latest_raw(timeout_ms)
        if raw is None:
            return None
        return decode_zenoh_audio_chunk(raw)

    def _on_sample(self, sample: Any) -> None:
        self._raw._on_sample(sample)


__all__ = [
    "LatestAudioChunk",
    "LatestAudioChunkTransport",
    "SAMPLE_FORMAT_F32LE",
    "ZENOH_AUDIO_CHUNK_MAGIC",
    "ZENOH_AUDIO_CHUNK_SCHEMA_VERSION",
    "ZenohLatestAudioChunkTransport",
    "decode_zenoh_audio_chunk",
    "encode_zenoh_audio_chunk",
]
