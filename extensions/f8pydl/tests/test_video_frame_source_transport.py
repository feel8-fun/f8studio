import os
import sys


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (PKG_PYDL,):
    if path not in sys.path:
        sys.path.insert(0, path)


from f8pydl.video_frame_source import LatestVideoFrameSource, VideoFrameSourceConfig, video_source_metadata  # noqa: E402
from f8pysdk.video_transport import LatestVideoFrame  # noqa: E402


def test_video_source_metadata_reports_typed_frame_stream() -> None:
    assert video_source_metadata() == {"payloadKind": "video_frame"}


def _latest_frame(*, stream_epoch: str, frame_id: int = 7, ts_ms: int = 1000) -> LatestVideoFrame:
    return LatestVideoFrame(
        width=2,
        height=1,
        pitch=8,
        fmt=0,
        frame_id=frame_id,
        ts_ms=ts_ms,
        payload=memoryview(bytearray(8)),
        stream_epoch=stream_epoch,
    )


def test_packet_carries_stream_epoch_from_transport() -> None:
    source = LatestVideoFrameSource(config=VideoFrameSourceConfig())
    packet = source._packet_from_zenoh(stream_key="k", frame=_latest_frame(stream_epoch="a" * 32), dedupe=True)
    assert packet is not None
    assert packet.stream_epoch == "a" * 32


def test_dedupe_does_not_drop_same_frame_id_from_restarted_stream() -> None:
    source = LatestVideoFrameSource(config=VideoFrameSourceConfig())
    first = source._packet_from_zenoh(stream_key="k", frame=_latest_frame(stream_epoch="a" * 32), dedupe=True)
    repeat = source._packet_from_zenoh(stream_key="k", frame=_latest_frame(stream_epoch="a" * 32), dedupe=True)
    restarted = source._packet_from_zenoh(stream_key="k", frame=_latest_frame(stream_epoch="b" * 32), dedupe=True)
    assert first is not None
    assert repeat is None
    assert restarted is not None
