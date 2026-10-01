from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterator

import msgspec

from .format import FORMAT_VERSION, FRAME_HEADER_SIZE
from .models import DataSampleEvent, HeaderEvent, RecordingEvent, RecordingHeader, StateChangeEvent, parse_event


_DECODER = msgspec.msgpack.Decoder(type=dict[str, object])


@dataclass(frozen=True)
class RecordingInfo:
    header: RecordingHeader
    duration_ms: int
    event_count: int


class RecordingReader:
    def __init__(self, path: str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def read_info(self) -> RecordingInfo:
        header: RecordingHeader | None = None
        event_count = 0
        max_offset_ms = 0
        for event in self.iter_events():
            event_count += 1
            if isinstance(event, HeaderEvent):
                header = event.header
                continue
            if isinstance(event, DataSampleEvent):
                max_offset_ms = max(max_offset_ms, int(event.relative_offset_ms))
                continue
            if isinstance(event, StateChangeEvent):
                max_offset_ms = max(max_offset_ms, int(event.relative_offset_ms))
        if header is None:
            raise ValueError("recording file is missing header")
        if int(header.format_version) != FORMAT_VERSION:
            raise ValueError(f"unsupported recording format version: {header.format_version}")
        return RecordingInfo(header=header, duration_ms=max_offset_ms, event_count=event_count)

    def read_header(self) -> RecordingHeader:
        for event in self.iter_events():
            if isinstance(event, HeaderEvent):
                if int(event.header.format_version) != FORMAT_VERSION:
                    raise ValueError(f"unsupported recording format version: {event.header.format_version}")
                return event.header
            raise ValueError("recording file does not start with header")
        raise ValueError("recording file is empty")

    def iter_events(self) -> Iterator[RecordingEvent]:
        if not self._path.exists():
            raise FileNotFoundError(str(self._path))
        with self._path.open("rb") as fh:
            while True:
                prefix = fh.read(FRAME_HEADER_SIZE)
                if not prefix:
                    return
                if len(prefix) != FRAME_HEADER_SIZE:
                    raise ValueError("truncated frame prefix")
                size = int(struct.unpack(">I", prefix)[0])
                if size <= 0:
                    raise ValueError("invalid frame size")
                payload = fh.read(size)
                if len(payload) != size:
                    raise ValueError("truncated frame payload")
                raw = _DECODER.decode(payload)
                yield parse_event(raw)
