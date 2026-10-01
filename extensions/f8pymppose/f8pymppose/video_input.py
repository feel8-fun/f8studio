from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32
from f8pysdk.video_transport import (
    LatestVideoFrameTransport,
    ZenohLatestVideoFrameTransport,
)


@dataclass(frozen=True)
class FrameContext:
    frame_id: int
    ts_ms: int
    width: int
    height: int
    pitch: int
    payload: bytes


class LatestVideoInput:
    def __init__(self) -> None:
        self._reader: LatestVideoFrameTransport | None = None
        self._open_key = ""

    @property
    def open_name(self) -> str:
        return self._open_key

    @property
    def is_open(self) -> bool:
        return self._reader is not None

    def is_open_for(self, *, stream_key: str) -> bool:
        return self._reader is not None and self._open_key == str(stream_key or "").strip()

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        self._reader = None
        self._open_key = ""

    def open(
        self,
        *,
        stream_key: str,
        config_path: str | None,
        connect: tuple[str, ...],
        listen: tuple[str, ...],
        shm_pool_bytes: int,
    ) -> None:
        self.close()
        reader = ZenohLatestVideoFrameTransport.open_subscriber(
            str(stream_key or "").strip(),
            config_path=config_path,
            connect=connect,
            listen=listen,
            shm_pool_bytes=shm_pool_bytes,
        )
        self._reader = reader
        self._open_key = str(stream_key or "").strip()

    def read_frame(self) -> FrameContext | None:
        assert self._reader is not None
        frame = self._reader.wait_latest(10)
        if frame is None:
            return None
        try:
            if int(frame.fmt) != int(VIDEO_FORMAT_BGRA32):
                return None
            width = int(frame.width)
            height = int(frame.height)
            pitch = int(frame.pitch)
            if width <= 0 or height <= 0 or pitch <= 0:
                return None

            frame_bytes = pitch * height
            if len(frame.payload) < frame_bytes:
                return None

            return FrameContext(
                frame_id=int(frame.frame_id),
                ts_ms=int(frame.ts_ms),
                width=width,
                height=height,
                pitch=pitch,
                payload=bytes(frame.payload[:frame_bytes]),
            )
        finally:
            frame.release()


def frame_rgb_from_context(frame: FrameContext, *, np_module: Any) -> Any:
    buf = np_module.frombuffer(frame.payload, dtype=np_module.uint8)
    rows = buf.reshape((frame.height, frame.pitch))
    bgra = rows[:, : frame.width * 4].reshape((frame.height, frame.width, 4))
    frame_bgr = bgra[:, :, 0:3]
    return np_module.ascontiguousarray(frame_bgr[:, :, ::-1])
