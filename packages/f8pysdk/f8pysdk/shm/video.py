from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .core import open_shared_memory_create, open_shared_memory_readonly
from .naming import frame_event_name, video_shm_name
from .win_event import Win32Event


VIDEO_SHM_MAGIC = 0xF8A11A01
VIDEO_SHM_VERSION = 1
VIDEO_FORMAT_BGRA32 = 1

_VIDEO_HEADER_STRUCT = struct.Struct("<7I4xQq2I")


@dataclass(frozen=True)
class VideoShmHeader:
    magic: int
    version: int
    slot_count: int
    width: int
    height: int
    pitch: int
    fmt: int
    frame_id: int
    ts_ms: int
    active_slot: int
    payload_capacity: int

    @property
    def header_bytes(self) -> int:
        return _VIDEO_HEADER_STRUCT.size

    @property
    def frame_bytes(self) -> int:
        return int(self.pitch) * int(self.height)

    @property
    def slot_offset_bytes(self) -> int:
        return self.header_bytes + int(self.active_slot) * int(self.payload_capacity)


def default_video_shm_name(service_id: str) -> str:
    return video_shm_name(service_id)


def read_video_header(buf: memoryview) -> Optional[VideoShmHeader]:
    if len(buf) < _VIDEO_HEADER_STRUCT.size:
        return None
    try:
        fields = _VIDEO_HEADER_STRUCT.unpack_from(buf, 0)
    except Exception:
        return None
    return VideoShmHeader(
        magic=fields[0],
        version=fields[1],
        slot_count=fields[2],
        width=fields[3],
        height=fields[4],
        pitch=fields[5],
        fmt=fields[6],
        frame_id=fields[7],
        ts_ms=fields[8],
        active_slot=fields[9],
        payload_capacity=fields[10],
    )


class VideoShmReader:
    def __init__(self, shm_name: str):
        self.shm_name = shm_name
        self._shm = None
        self._event: Optional[Win32Event] = None

    def open(self, use_event: bool = True) -> None:
        self._shm = open_shared_memory_readonly(self.shm_name)
        if use_event and os.name == "nt":
            self._event = Win32Event.open(frame_event_name(self.shm_name))

    def close(self) -> None:
        if self._event:
            self._event.close()
            self._event = None
        if self._shm:
            self._shm.close()
            self._shm = None

    @property
    def has_event(self) -> bool:
        return self._event is not None

    @property
    def buf(self) -> memoryview:
        if not self._shm:
            raise RuntimeError("VideoShmReader is not open")
        return self._shm.buf

    def wait_new_frame(self, timeout_ms: int = 10) -> bool:
        if self._event:
            return self._event.wait(timeout_ms)
        time.sleep(max(1, timeout_ms) / 1000.0)
        return False

    def read_header(self) -> Optional[VideoShmHeader]:
        return read_video_header(self.buf)

    def read_latest_bgra(self) -> Tuple[Optional[VideoShmHeader], Optional[memoryview]]:
        buf = self.buf
        h0 = read_video_header(buf)
        if not h0 or h0.magic != VIDEO_SHM_MAGIC or h0.version != VIDEO_SHM_VERSION:
            return None, None
        if h0.fmt != VIDEO_FORMAT_BGRA32 or h0.width <= 0 or h0.height <= 0 or h0.pitch <= 0:
            return None, None
        if h0.frame_bytes > h0.payload_capacity:
            return None, None
        if h0.slot_offset_bytes + h0.frame_bytes > len(buf):
            return None, None
        h1 = read_video_header(buf)
        if not h1 or h1.frame_id != h0.frame_id or h1.active_slot != h0.active_slot:
            return None, None
        return h0, buf[h0.slot_offset_bytes : h0.slot_offset_bytes + h0.frame_bytes]


class VideoShmWriter:
    def __init__(self, shm_name: str, size: int, slot_count: int = 2):
        self.shm_name = shm_name
        self.size = int(size)
        self.slot_count = int(max(1, slot_count))
        self._shm: Optional[SharedMemory] = None
        self._event: Optional[Win32Event] = None
        self._active_slot = 0
        self._frame_id = 0
        self._payload_capacity = 0

    def open(self) -> None:
        self._shm = open_shared_memory_create(self.shm_name, self.size)
        if os.name == "nt":
            self._event = Win32Event.create(self.shm_name + "_evt", manual_reset=True, initial_state=False)
        self._init_header()

    def close(self, unlink: bool = False) -> None:
        if self._event:
            self._event.close()
            self._event = None
        if self._shm:
            self._shm.close()
            if unlink:
                try:
                    self._shm.unlink()
                except Exception:
                    pass
            self._shm = None

    @property
    def buf(self) -> memoryview:
        if not self._shm:
            raise RuntimeError("VideoShmWriter is not open")
        return self._shm.buf

    def _init_header(self) -> None:
        buf = self.buf
        header_bytes = _VIDEO_HEADER_STRUCT.size
        usable = max(0, len(buf) - header_bytes)
        self._payload_capacity = usable // self.slot_count
        _VIDEO_HEADER_STRUCT.pack_into(
            buf,
            0,
            VIDEO_SHM_MAGIC,
            VIDEO_SHM_VERSION,
            self.slot_count,
            0,
            0,
            0,
            VIDEO_FORMAT_BGRA32,
            0,
            0,
            0,
            self._payload_capacity,
        )

    def write_frame_bgra(self, width: int, height: int, pitch: int, payload: bytes) -> None:
        buf = self.buf
        if width <= 0 or height <= 0 or pitch <= 0:
            return
        frame_bytes = int(pitch) * int(height)
        if len(payload) < frame_bytes:
            return
        if frame_bytes > self._payload_capacity:
            return

        self._active_slot = (self._active_slot + 1) % self.slot_count
        header_bytes = _VIDEO_HEADER_STRUCT.size
        slot_off = header_bytes + self._active_slot * self._payload_capacity
        buf[slot_off : slot_off + frame_bytes] = payload[:frame_bytes]

        self._frame_id += 1
        ts_ms = int(time.time() * 1000)
        _VIDEO_HEADER_STRUCT.pack_into(
            buf,
            0,
            VIDEO_SHM_MAGIC,
            VIDEO_SHM_VERSION,
            self.slot_count,
            int(width),
            int(height),
            int(pitch),
            VIDEO_FORMAT_BGRA32,
            int(self._frame_id),
            int(ts_ms),
            int(self._active_slot),
            int(self._payload_capacity),
        )

        if self._event:
            self._event.pulse()
