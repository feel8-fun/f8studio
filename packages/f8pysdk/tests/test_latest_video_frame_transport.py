from __future__ import annotations

import time
import uuid

import pytest

from f8pysdk.binary_stream_transport import _declare_zenoh_latest_publisher
from f8pysdk.video_transport import (
    VIDEO_FORMAT_BGRA32,
    ZenohLatestVideoFrameTransport,
    decode_zenoh_video_frame,
    encode_zenoh_video_frame,
)


def test_zenoh_video_frame_codec_roundtrip() -> None:
    width = 3
    height = 2
    pitch = width * 4
    payload = bytes(range(pitch * height))

    raw = encode_zenoh_video_frame(
        width=width,
        height=height,
        pitch=pitch,
        payload=payload,
        fmt=VIDEO_FORMAT_BGRA32,
        frame_id=7,
        ts_ms=1234,
        stream_epoch="0123456789abcdef0123456789abcdef",
    )
    frame = decode_zenoh_video_frame(raw)
    assert frame is not None
    try:
        assert frame.width == width
        assert frame.height == height
        assert frame.pitch == pitch
        assert frame.fmt == VIDEO_FORMAT_BGRA32
        assert frame.frame_id == 7
        assert frame.ts_ms == 1234
        assert frame.stream_epoch == "0123456789abcdef0123456789abcdef"
        assert frame.payload_bytes() == payload
    finally:
        frame.release()


class _PublisherSession:
    def __init__(self) -> None:
        self.key_expr = ""
        self.options: dict[str, object] = {}

    def declare_publisher(self, key_expr: str, **options: object) -> object:
        self.key_expr = str(key_expr)
        self.options = dict(options)
        return object()


def test_zenoh_video_declared_publisher_uses_latest_frame_qos() -> None:
    zenoh = pytest.importorskip("zenoh")
    session = _PublisherSession()

    publisher = _declare_zenoh_latest_publisher(session, "f8/test/video/qos")

    assert publisher is not None
    assert session.key_expr == "f8/test/video/qos"
    assert session.options["encoding"] == zenoh.Encoding.APPLICATION_OCTET_STREAM
    assert session.options["congestion_control"] == zenoh.CongestionControl.DROP
    assert session.options["priority"] == zenoh.Priority.REAL_TIME
    assert session.options["express"] is True
    assert session.options["reliability"] == zenoh.Reliability.BEST_EFFORT


def test_zenoh_latest_video_frame_transport_skips_to_latest() -> None:
    pytest.importorskip("zenoh")
    key = f"f8/test/video/{uuid.uuid4().hex}"
    subscriber = ZenohLatestVideoFrameTransport.open_subscriber(key)
    publisher = ZenohLatestVideoFrameTransport.open_publisher(key)
    try:
        width = 2
        height = 2
        pitch = width * 4
        deadline = time.monotonic() + 2.0
        latest_frame = None
        latest_payload = b""
        next_publish_at = 0.0
        seq = 0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if seq < 5 and now >= next_publish_at:
                latest_payload = bytes(((seq * 17 + i) % 256 for i in range(pitch * height)))
                publisher.publish_frame(
                    width=width,
                    height=height,
                    pitch=pitch,
                    payload=latest_payload,
                    fmt=VIDEO_FORMAT_BGRA32,
                )
                seq += 1
                next_publish_at = now + 0.02
            frame = subscriber.wait_latest(timeout_ms=50)
            if frame is None:
                continue
            if frame.frame_id == 5:
                latest_frame = frame
                break
            frame.release()
        assert latest_frame is not None
        try:
            assert latest_frame.payload_bytes() == latest_payload
        finally:
            latest_frame.release()

        assert subscriber.poll_latest() is None
    finally:
        subscriber.close()
        publisher.close()
