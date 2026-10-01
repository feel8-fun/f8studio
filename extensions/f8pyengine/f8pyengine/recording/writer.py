from __future__ import annotations

from pathlib import Path
import os
import struct
from typing import Any

import msgspec

from .format import FRAME_HEADER_SIZE
from .models import (
    DataSampleEvent,
    HeaderEvent,
    RecordingEvent,
    RecordingHeader,
    StateChangeEvent,
    event_to_dict,
)


_ENCODER = msgspec.msgpack.Encoder()


class RecordingWriter:
    def __init__(self, path: str, *, header: RecordingHeader, append: bool) -> None:
        self._path = Path(path)
        self._header = header
        self._append = bool(append)
        self._fh: Any | None = None

    @property
    def header(self) -> RecordingHeader:
        return self._header

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if self._append else "wb"
        self._fh = self._path.open(mode)
        if self._append and self._path.stat().st_size > 0:
            return
        self.write_header()

    def close(self) -> None:
        fh = self._fh
        self._fh = None
        if fh is not None:
            fh.close()

    def write_header(self) -> None:
        self.write_event(HeaderEvent(type="header", header=self._header))

    def write_data_sample(self, *, tick_ts_ms: int, relative_offset_ms: int, data: dict[str, Any]) -> None:
        self.write_event(
            DataSampleEvent(
                type="data_sample",
                tick_ts_ms=int(tick_ts_ms),
                relative_offset_ms=int(relative_offset_ms),
                data=dict(data),
            )
        )

    def write_state_change(self, *, state_ts_ms: int, relative_offset_ms: int, field: str, value: Any) -> None:
        self.write_event(
            StateChangeEvent(
                type="state_change",
                state_ts_ms=int(state_ts_ms),
                relative_offset_ms=int(relative_offset_ms),
                field=str(field),
                value=value,
            )
        )

    def write_event(self, event: RecordingEvent) -> None:
        if self._fh is None:
            raise RuntimeError("recording writer is not open")
        payload = _ENCODER.encode(event_to_dict(event))
        frame_size = len(payload)
        self._fh.write(struct.pack(">I", frame_size))
        self._fh.write(payload)
        self._fh.flush()
        os.fsync(self._fh.fileno())


def frame_size_from_prefix(prefix: bytes) -> int:
    if len(prefix) != FRAME_HEADER_SIZE:
        raise ValueError("invalid frame prefix length")
    return int(struct.unpack(">I", prefix)[0])
