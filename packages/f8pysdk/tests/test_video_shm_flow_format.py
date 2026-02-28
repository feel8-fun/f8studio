from __future__ import annotations

import uuid

from f8pysdk.shm.video import (
    VIDEO_FORMAT_FLOW2_F16,
    VideoShmReader,
    VideoShmWriter,
)


def test_video_shm_flow2f16_roundtrip() -> None:
    shm_name = f"test.shm.flow.{uuid.uuid4().hex}"
    writer = VideoShmWriter(shm_name=shm_name, size=1024 * 1024, slot_count=2)
    reader = VideoShmReader(shm_name=shm_name)
    try:
        writer.open()
        reader.open(use_event=False)

        width = 8
        height = 4
        pitch = width * 4
        frame_bytes = pitch * height
        payload = bytes((i % 251 for i in range(frame_bytes)))
        writer.write_frame(width=width, height=height, pitch=pitch, payload=payload, fmt=VIDEO_FORMAT_FLOW2_F16)

        header, frame = reader.read_latest_frame()
        assert header is not None
        assert frame is not None
        assert int(header.fmt) == VIDEO_FORMAT_FLOW2_F16
        assert int(header.width) == width
        assert int(header.height) == height
        assert int(header.pitch) == pitch
        assert bytes(frame) == payload
        frame.release()
        frame = None

        bgra_header, bgra_frame = reader.read_latest_bgra()
        assert bgra_header is None
        assert bgra_frame is None
        if bgra_frame is not None:
            bgra_frame.release()
    finally:
        reader.close()
        writer.close(unlink=True)
