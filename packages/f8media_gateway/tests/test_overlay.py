import asyncio

from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32
from f8media_protocol.models import OverlayDetection, OverlayResult
from f8media_gateway.overlay import OverlayStore, compose_overlay_bgra


def overlay(*, epoch: str = "epoch-1", frame_id: int = 7) -> OverlayResult:
    return OverlayResult(
        source="f8/source/video",
        stream_id="camera-1",
        stream_epoch=epoch,
        frame_id=frame_id,
        capture_timestamp_ms=1234,
        detections=(OverlayDetection(x=0.25, y=0.25, width=0.5, height=0.5, label="person", score=0.9),),
    )


def test_overlay_composition_draws_only_the_exact_box() -> None:
    source = bytes((10, 20, 30, 255)) * 8 * 8
    composed = compose_overlay_bgra(
        width=8,
        height=8,
        pitch=32,
        pixel_format=VIDEO_FORMAT_BGRA32,
        payload=source,
        result=overlay(),
    )
    assert composed != source
    assert composed[2 * 32 + 2 * 4 : 2 * 32 + 2 * 4 + 4] == b"\x4c\xdc\x78\xff"
    assert composed[:4] == source[:4]


def test_overlay_store_never_cross_matches_frame_or_stream_epoch() -> None:
    async def scenario() -> None:
        store = OverlayStore(max_results=4, ttl_s=1.0)
        await store.publish(overlay(epoch="old", frame_id=1))
        wrong_epoch = await store.match(
            source="f8/source/video",
            stream_id="camera-1",
            stream_epoch="new",
            frame_id=1,
            capture_timestamp_ms=1234,
            wait_budget_s=0,
        )
        wrong_frame = await store.match(
            source="f8/source/video",
            stream_id="camera-1",
            stream_epoch="old",
            frame_id=2,
            capture_timestamp_ms=1234,
            wait_budget_s=0,
        )
        exact = await store.match(
            source="f8/source/video",
            stream_id="camera-1",
            stream_epoch="old",
            frame_id=1,
            capture_timestamp_ms=1234,
            wait_budget_s=0,
        )
        assert wrong_epoch is None
        assert wrong_frame is None
        assert exact is not None
        assert store.counters.matched == 1
        assert store.counters.wait_timeouts == 2

    asyncio.run(scenario())


def test_overlay_store_expires_and_stays_bounded() -> None:
    async def scenario() -> None:
        store = OverlayStore(max_results=2, ttl_s=0.01)
        for frame_id in range(5):
            await store.publish(overlay(frame_id=frame_id))
        assert store.size == 2
        assert store.counters.expired == 3
        await asyncio.sleep(0.02)
        assert await store.match(
            source="missing", stream_id="missing", stream_epoch="missing", frame_id=0,
            capture_timestamp_ms=0, wait_budget_s=0
        ) is None
        assert store.size == 0
        assert store.counters.expired == 5

    asyncio.run(scenario())
