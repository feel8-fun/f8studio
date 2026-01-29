from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .core import open_shared_memory_create, open_shared_memory_readonly
from .naming import audio_shm_name, frame_event_name
from .win_event import Win32Event


AUDIO_SHM_MAGIC = 0xF8A11A02
AUDIO_SHM_VERSION = 1

SAMPLE_FORMAT_F32LE = 1
SAMPLE_FORMAT_S16LE = 2

_AUDIO_HEADER_STRUCT = struct.Struct("<IIIHHIIIIQQq")
_CHUNK_HEADER_STRUCT = struct.Struct("<QqII")


@dataclass(frozen=True)
class AudioShmHeader:
    magic: int
    version: int
    sample_rate: int
    channels: int
    fmt: int
    frames_per_chunk: int
    chunk_count: int
    bytes_per_frame: int
    payload_bytes_per_chunk: int
    write_seq: int
    write_frame_index: int
    ts_ms: int

    @property
    def header_bytes(self) -> int:
        return _AUDIO_HEADER_STRUCT.size

    @property
    def chunk_header_bytes(self) -> int:
        return _CHUNK_HEADER_STRUCT.size

    @property
    def chunk_stride_bytes(self) -> int:
        return self.chunk_header_bytes + int(self.payload_bytes_per_chunk)


@dataclass(frozen=True)
class AudioShmChunkHeader:
    seq: int
    ts_ms: int
    frames: int


def default_audio_shm_name(service_id: str) -> str:
    return audio_shm_name(service_id)


def read_audio_header(buf: memoryview) -> Optional[AudioShmHeader]:
    if len(buf) < _AUDIO_HEADER_STRUCT.size:
        return None
    try:
        fields = _AUDIO_HEADER_STRUCT.unpack_from(buf, 0)
    except Exception:
        return None
    return AudioShmHeader(
        magic=fields[0],
        version=fields[1],
        sample_rate=fields[2],
        channels=fields[3],
        fmt=fields[4],
        frames_per_chunk=fields[5],
        chunk_count=fields[6],
        bytes_per_frame=fields[7],
        payload_bytes_per_chunk=fields[8],
        write_seq=fields[9],
        write_frame_index=fields[10],
        ts_ms=fields[11],
    )


def read_chunk_header(buf: memoryview, offset: int) -> Optional[AudioShmChunkHeader]:
    if offset < 0 or offset + _CHUNK_HEADER_STRUCT.size > len(buf):
        return None
    try:
        fields = _CHUNK_HEADER_STRUCT.unpack_from(buf, offset)
    except Exception:
        return None
    return AudioShmChunkHeader(seq=fields[0], ts_ms=fields[1], frames=fields[2])


class AudioShmReader:
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
    def buf(self) -> memoryview:
        if not self._shm:
            raise RuntimeError("AudioShmReader is not open")
        return self._shm.buf

    def wait_new_chunk(self, timeout_ms: int = 10) -> bool:
        if self._event:
            return self._event.wait(timeout_ms)
        time.sleep(max(1, timeout_ms) / 1000.0)
        return False

    def read_header(self) -> Optional[AudioShmHeader]:
        return read_audio_header(self.buf)

    def read_chunk_f32(self, seq: int) -> Tuple[Optional[AudioShmHeader], Optional[AudioShmChunkHeader], Optional[memoryview]]:
        buf = self.buf
        hdr = read_audio_header(buf)
        if not hdr or hdr.magic != AUDIO_SHM_MAGIC or hdr.version != AUDIO_SHM_VERSION:
            return None, None, None
        if hdr.fmt != SAMPLE_FORMAT_F32LE:
            return None, None, None
        if hdr.frames_per_chunk <= 0 or hdr.chunk_count <= 0 or hdr.payload_bytes_per_chunk <= 0:
            return None, None, None

        idx = int(seq) % int(hdr.chunk_count)
        base = hdr.header_bytes + idx * hdr.chunk_stride_bytes
        ch = read_chunk_header(buf, base)
        if not ch or ch.seq != int(seq):
            return hdr, None, None
        frames = int(min(ch.frames, hdr.frames_per_chunk))
        payload_off = base + hdr.chunk_header_bytes
        payload_len = frames * int(hdr.bytes_per_frame)
        if payload_off + payload_len > len(buf):
            return hdr, None, None
        return hdr, ch, buf[payload_off : payload_off + payload_len]


class AudioShmWriter:
    def __init__(
        self,
        shm_name: str,
        size: int,
        sample_rate: int = 48000,
        channels: int = 2,
        frames_per_chunk: int = 480,
        chunk_count: int = 200,
        fmt: int = SAMPLE_FORMAT_F32LE,
    ):
        self.shm_name = shm_name
        self.size = int(size)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frames_per_chunk = int(frames_per_chunk)
        self.chunk_count = int(chunk_count)
        self.fmt = int(fmt)
        self._shm: Optional[SharedMemory] = None
        self._event: Optional[Win32Event] = None
        self._write_seq = 0
        self._write_frame_index = 0

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
            raise RuntimeError("AudioShmWriter is not open")
        return self._shm.buf

    def _init_header(self) -> None:
        bytes_per_sample = 4 if self.fmt == SAMPLE_FORMAT_F32LE else 2
        bytes_per_frame = bytes_per_sample * self.channels
        payload_bytes_per_chunk = bytes_per_frame * self.frames_per_chunk
        _AUDIO_HEADER_STRUCT.pack_into(
            self.buf,
            0,
            AUDIO_SHM_MAGIC,
            AUDIO_SHM_VERSION,
            self.sample_rate,
            self.channels,
            self.fmt,
            self.frames_per_chunk,
            self.chunk_count,
            bytes_per_frame,
            payload_bytes_per_chunk,
            0,
            0,
            0,
        )

    def write_chunk_f32(self, interleaved_f32_bytes: bytes, frames: int) -> None:
        if self.fmt != SAMPLE_FORMAT_F32LE:
            return
        if frames <= 0:
            return
        hdr = read_audio_header(self.buf)
        if not hdr:
            return

        frames = int(min(frames, hdr.frames_per_chunk))
        bytes_needed = frames * int(hdr.bytes_per_frame)
        if len(interleaved_f32_bytes) < bytes_needed:
            return

        self._write_seq += 1
        seq = self._write_seq
        idx = seq % int(hdr.chunk_count)
        base = hdr.header_bytes + idx * hdr.chunk_stride_bytes
        payload_off = base + hdr.chunk_header_bytes
        self.buf[payload_off : payload_off + bytes_needed] = interleaved_f32_bytes[:bytes_needed]

        ts_ms = int(time.time() * 1000)
        _CHUNK_HEADER_STRUCT.pack_into(self.buf, base, int(seq), int(ts_ms), int(frames), 0)
        self._write_frame_index += frames
        _AUDIO_HEADER_STRUCT.pack_into(
            self.buf,
            0,
            AUDIO_SHM_MAGIC,
            AUDIO_SHM_VERSION,
            hdr.sample_rate,
            hdr.channels,
            hdr.fmt,
            hdr.frames_per_chunk,
            hdr.chunk_count,
            hdr.bytes_per_frame,
            hdr.payload_bytes_per_chunk,
            int(seq),
            int(self._write_frame_index),
            int(ts_ms),
        )
        if self._event:
            self._event.pulse()
