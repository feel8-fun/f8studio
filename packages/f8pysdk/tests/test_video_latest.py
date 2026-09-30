import asyncio
import logging

import pytest

from f8pysdk.video_latest import VideoLatestPacketCodec, VideoLatestSubscription, VideoLatestSubscriptions
from f8pysdk.video_transport import LatestVideoFrame, VIDEO_FORMAT_BGRA32, VIDEO_FORMAT_FLOW2_F16


@pytest.mark.parametrize("fmt,kind,channels", [(VIDEO_FORMAT_BGRA32, "bgra32", 4), (VIDEO_FORMAT_FLOW2_F16, "flow2_f16", 2)])
def test_padded_and_truncated_frames(fmt: int, kind: str, channels: int) -> None:
    codec = VideoLatestPacketCodec()
    header = {"width": 1, "height": 2, "pitch": 8, "fmt": fmt}
    decoded = codec.decode_payload(header=header, raw=bytes(16), decode_mode="auto")
    assert decoded is not None
    assert decoded["kind"] == kind
    assert decoded["shape"] == [2, 1, channels]
    truncated = codec.decode_payload(header=header, raw=bytes(12), decode_mode="auto")
    assert truncated is not None and truncated["data"] is None
    assert codec.decode_payload(header=header, raw=bytes(16), decode_mode="none") is None


class Reader:
    def __init__(self) -> None:
        self.closed = False
        self.frame = LatestVideoFrame(width=1, height=1, pitch=4, fmt=VIDEO_FORMAT_BGRA32,
                                      frame_id=1, ts_ms=1, payload=memoryview(b"1234"))
        self.delivered = False

    def wait_latest(self, timeout_ms: int) -> LatestVideoFrame | None:
        if self.delivered:
            return None
        self.delivered = True
        return self.frame

    def close(self) -> None:
        self.closed = True


class Subscriptions(VideoLatestSubscriptions):
    def __init__(self) -> None:
        self.enabled = False
        self.readers: list[Reader] = []
        super().__init__(node_id="test", log_context="test", read_enabled=self.can_read)

    def can_read(self) -> bool:
        return self.enabled

    def _open_reader(self, sub: VideoLatestSubscription) -> Reader:
        reader = Reader()
        self.readers.append(reader)
        return reader


def test_pause_replace_release_and_shutdown() -> None:
    async def scenario() -> None:
        subscriptions = Subscriptions()
        subscriptions.subscribe("video", stream_key="test/video", decode="none")
        await asyncio.sleep(0)
        assert subscriptions.readers == []
        subscriptions.enabled = True

        async def packet_ready() -> None:
            while subscriptions.get_packet("video") is None:
                await asyncio.sleep(0.001)

        try:
            await asyncio.wait_for(packet_ready(), timeout=1)
            packet = subscriptions.get_packet("video")
            assert packet is not None and packet["raw"] == b"1234"
            packet["header"]["frameId"] = 99
            assert subscriptions.get_packet("video")["header"]["frameId"] == 1
            with pytest.raises(RuntimeError, match="released"):
                subscriptions.readers[0].frame.payload_bytes()
            subscriptions.subscribe("video", stream_key="test/replacement", decode="none")
            assert subscriptions.readers[0].closed
            await asyncio.wait_for(packet_ready(), timeout=1)
        finally:
            await subscriptions.shutdown_async()
        assert all(reader.closed for reader in subscriptions.readers)
        assert subscriptions.list_status() == []

    asyncio.run(scenario())


def test_repeated_transport_errors_are_counted_and_log_traceback_once(caplog: pytest.LogCaptureFixture) -> None:
    subscriptions = Subscriptions()
    sub = VideoLatestSubscription(key="video", stream_key="test/video", decode_mode="none")
    with caplog.at_level(logging.ERROR):
        try:
            raise OSError("transport unavailable")
        except OSError as exc:
            subscriptions._log_sub_error(sub, "open", exc)
            subscriptions._log_sub_error(sub, "open", exc)
    assert sub.error_count == 2
    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is not None
