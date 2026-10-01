from __future__ import annotations

import os
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from f8pysdk.bus import ServiceBus, ServiceBusConfig
from f8pysdk.video_transport import LatestVideoFrame, ZenohLatestVideoFrameTransport

def _env_tuple(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    items: list[str] = []
    for part in str(raw or "").split(","):
        item = part.strip()
        if item:
            items.append(item)
    return tuple(items)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return int(default)
    try:
        value = int(raw.strip())
    except ValueError:
        return int(default)
    return max(0, value)


@dataclass(frozen=True)
class VideoFrameSourceConfig:
    config_path: str | None = None
    connect: tuple[str, ...] = ()
    listen: tuple[str, ...] = ()
    shm_pool_bytes: int = 256 * 1024 * 1024

    @classmethod
    def from_bus(cls, bus: Any | None) -> "VideoFrameSourceConfig":
        if isinstance(bus, ServiceBus):
            cfg: ServiceBusConfig = bus.config
            return cls(
                config_path=cfg.zenoh_config_path,
                connect=cfg.zenoh_connect,
                listen=cfg.zenoh_listen,
                shm_pool_bytes=cfg.zenoh_shm_pool_bytes,
            )
        config_path_raw = os.environ.get("F8_ZENOH_CONFIG", "").strip()
        return cls(
            config_path=config_path_raw or None,
            connect=_env_tuple("F8_ZENOH_CONNECT"),
            listen=_env_tuple("F8_ZENOH_LISTEN"),
            shm_pool_bytes=_env_int("F8_ZENOH_SHM_POOL_BYTES", 256 * 1024 * 1024),
        )


@dataclass
class VideoFramePacket:
    width: int
    height: int
    pitch: int
    fmt: int
    frame_id: int
    ts_ms: int
    stream_epoch: str
    payload: memoryview
    _released: bool = False

    @property
    def frame_bytes(self) -> int:
        return int(self.pitch) * int(self.height)

    def release(self) -> None:
        if self._released:
            return
        self.payload.release()
        self._released = True

    def __enter__(self) -> "VideoFramePacket":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def video_source_metadata() -> dict[str, str]:
    return {"payloadKind": "video_frame"}


class LatestVideoFrameSource:
    def __init__(self, *, config: VideoFrameSourceConfig) -> None:
        self._config = config
        self._zenoh_reader: ZenohLatestVideoFrameTransport | None = None
        self._zenoh_open_key = ""
        self._last_signature: tuple[str, str, int, int] | None = None

    def close(self) -> None:
        self._close_zenoh()
        self._last_signature = None

    def reset(self) -> None:
        self.close()

    def read_latest(
        self,
        *,
        stream_key: str,
        timeout_ms: int,
        dedupe: bool = True,
    ) -> VideoFramePacket | None:
        return self._read_zenoh_latest(stream_key=str(stream_key).strip(), timeout_ms=timeout_ms, dedupe=dedupe)

    def _read_zenoh_latest(self, *, stream_key: str, timeout_ms: int, dedupe: bool) -> VideoFramePacket | None:
        if not stream_key:
            return None
        reader = self._ensure_zenoh_reader(stream_key)
        frame = reader.wait_latest(max(0, int(timeout_ms)))
        if frame is None:
            return None
        return self._packet_from_zenoh(stream_key=stream_key, frame=frame, dedupe=dedupe)

    def _packet_from_zenoh(
        self, *, stream_key: str, frame: LatestVideoFrame, dedupe: bool
    ) -> VideoFramePacket | None:
        signature = (str(stream_key), frame.stream_epoch, int(frame.frame_id), int(frame.ts_ms))
        if dedupe and signature == self._last_signature:
            frame.release()
            return None
        if dedupe:
            self._last_signature = signature
        return VideoFramePacket(
            width=int(frame.width),
            height=int(frame.height),
            pitch=int(frame.pitch),
            fmt=int(frame.fmt),
            frame_id=int(frame.frame_id),
            ts_ms=int(frame.ts_ms),
            stream_epoch=frame.stream_epoch,
            payload=frame.payload,
        )

    def _ensure_zenoh_reader(self, key_expr: str) -> ZenohLatestVideoFrameTransport:
        key = str(key_expr or "").strip()
        if self._zenoh_reader is not None and self._zenoh_open_key == key:
            return self._zenoh_reader
        self._close_zenoh()
        reader = ZenohLatestVideoFrameTransport.open_subscriber(
            key,
            config_path=self._config.config_path,
            connect=self._config.connect,
            listen=self._config.listen,
            shm_pool_bytes=self._config.shm_pool_bytes,
        )
        self._zenoh_reader = reader
        self._zenoh_open_key = key
        self._last_signature = None
        return reader

    def _close_zenoh(self) -> None:
        reader = self._zenoh_reader
        self._zenoh_reader = None
        self._zenoh_open_key = ""
        if reader is not None:
            reader.close()
