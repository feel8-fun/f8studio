import asyncio
import math
import struct

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import VideoFrame

from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32, VIDEO_FORMAT_FLOW2_F16, VIDEO_FORMAT_SCALAR1_F32
from f8media_gateway.media import (
    LatestFrameHub,
    LatestFrameVideoTrack,
    MEDIA_QUALITIES,
    MediaSessionManager,
    RawVideoFrame,
    SyntheticFrameProducer,
    bgra_frame,
    preview_frame,
    sample_raw_frame,
)
from f8media_protocol.models import MediaSessionOffer
from f8media_protocol.models import OverlayDetection, OverlayResult
from f8media_gateway.overlay import OverlayStore


async def wait_for_connected(peer: RTCPeerConnection) -> None:
    deadline = asyncio.get_running_loop().time() + 5.0
    while peer.connectionState != "connected":
        if peer.connectionState in {"closed", "failed"}:
            raise RuntimeError(f"peer entered {peer.connectionState} before connecting")
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("peer did not connect within 5 seconds")
        await asyncio.sleep(0.01)


def test_bgra_frame_copies_non_contiguous_rows() -> None:
    # Two 2-pixel rows with four bytes of source padding after each row.
    first_row = bytes((1, 2, 3, 255, 4, 5, 6, 255))
    second_row = bytes((7, 8, 9, 255, 10, 11, 12, 255))
    raw = RawVideoFrame(
        width=2,
        height=2,
        pitch=12,
        format=VIDEO_FORMAT_BGRA32,
        frame_id=1,
        ts_ms=100,
        payload=first_row + b"pad!" + second_row + b"pad!",
    )

    frame = bgra_frame(raw)
    plane = frame.planes[0]
    packed = bytes(plane)

    assert frame.width == 2
    assert frame.height == 2
    assert packed[:8] == first_row
    assert packed[plane.line_size : plane.line_size + 8] == second_row


def test_exact_media_samples_respect_format_stride_and_non_finite_values() -> None:
    scalar_payload = struct.pack("<ff", 1.25, math.nan) + b"padding!"
    scalar = RawVideoFrame(
        width=2, height=1, pitch=16, format=VIDEO_FORMAT_SCALAR1_F32,
        frame_id=8, ts_ms=900, payload=scalar_payload,
    )
    first = sample_raw_frame("scalar", scalar, x=0, y=0)
    invalid = sample_raw_frame("scalar", scalar, x=1, y=0)
    assert first.format == "scalar1_f32"
    assert first.value == 1.25
    assert first.finite is True
    assert invalid.value is None
    assert invalid.finite is False

    flow_payload = b"rowpadding!!" + struct.pack("<ee", -2.5, 3.0) + b"padding!"
    flow = RawVideoFrame(
        width=1, height=2, pitch=12, format=VIDEO_FORMAT_FLOW2_F16,
        frame_id=9, ts_ms=901, payload=flow_payload,
    )
    vector = sample_raw_frame("flow", flow, x=0, y=1)
    assert vector.format == "flow2_f16"
    assert vector.value == {"dx": -2.5, "dy": 3.0}

    try:
        sample_raw_frame("flow", flow, x=1, y=0)
    except ValueError as exc:
        assert "outside 1x2" in str(exc)
    else:
        raise AssertionError("out-of-range sample coordinate was accepted")


def test_flow_and_scalar_frames_convert_to_video_previews() -> None:
    flow_row = struct.pack("<eeee", 1.0, 0.0, 0.0, 2.0) + b"pad!"
    flow = RawVideoFrame(
        width=2,
        height=1,
        pitch=12,
        format=VIDEO_FORMAT_FLOW2_F16,
        frame_id=1,
        ts_ms=1,
        payload=flow_row,
    )
    flow_preview = preview_frame(flow)
    assert flow_preview.width == 2
    assert flow_preview.height == 1
    assert any(bytes(flow_preview.planes[0])[:8])

    scalar_row = struct.pack("<fff", 0.0, 1.0, math.nan) + b"pad!"
    scalar = RawVideoFrame(
        width=3,
        height=1,
        pitch=16,
        format=VIDEO_FORMAT_SCALAR1_F32,
        frame_id=2,
        ts_ms=2,
        payload=scalar_row,
    )
    scalar_preview = preview_frame(scalar)
    assert scalar_preview.width == 3
    assert scalar_preview.height == 1
    preview_bytes = bytes(scalar_preview.planes[0])
    assert preview_bytes[3] == 255
    assert preview_bytes[11] == 255


def test_synthetic_webrtc_session_delivers_decoded_video() -> None:
    async def scenario() -> None:
        manager = MediaSessionManager()
        client = RTCPeerConnection()
        remote_track: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()

        @client.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if not remote_track.done():
                remote_track.set_result(track)

        client.addTransceiver("video", direction="recvonly")
        offer = await client.createOffer()
        await client.setLocalDescription(offer)
        local = client.localDescription
        answer = await manager.create(
            MediaSessionOffer(
                source="synthetic://bars",
                quality="thumbnail",
                sdp=local.sdp,
                type=local.type,
            )
        )
        await client.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))

        track = await asyncio.wait_for(remote_track, timeout=5.0)
        decoded = await asyncio.wait_for(track.recv(), timeout=5.0)
        assert isinstance(decoded, VideoFrame)
        assert decoded.width == 640
        assert decoded.height == 360
        assert answer.max_fps == 10

        assert await manager.close_session(answer.session_id) is True
        assert await manager.close_session(answer.session_id) is False
        await client.close()
        await manager.close()

    asyncio.run(scenario())


def test_resolution_change_preserves_monotonic_timestamps() -> None:
    class ChangingProducer:
        def __init__(self) -> None:
            self.index = 0

        async def read(self) -> RawVideoFrame | None:
            await asyncio.sleep(0.02)
            dimensions = ((320, 180), (640, 360), (640, 360))
            timestamps = (1_000, 1_033, 100)
            frame_index = min(self.index, 2)
            width, height = dimensions[frame_index]
            self.index += 1
            return RawVideoFrame(
                width=width,
                height=height,
                pitch=width * 4,
                format=VIDEO_FORMAT_BGRA32,
                frame_id=self.index,
                ts_ms=timestamps[frame_index],
                payload=bytes((20, 80, 160, 255)) * width * height,
            )

        async def close(self) -> None:
            return None

    async def scenario() -> None:
        hub = LatestFrameHub(source="changing", producer=ChangingProducer())
        hub.start()
        track = LatestFrameVideoTrack(hub, MEDIA_QUALITIES["main"])
        first = await asyncio.wait_for(track.recv(), timeout=1.0)
        second = await asyncio.wait_for(track.recv(), timeout=1.0)
        third = await asyncio.wait_for(track.recv(), timeout=1.0)
        assert (first.width, first.height) == (320, 180)
        assert (second.width, second.height) == (640, 360)
        assert first.pts is not None
        assert second.pts is not None
        assert second.pts > first.pts
        assert third.pts is not None
        assert third.pts > second.pts
        track.stop()
        await hub.close()

    asyncio.run(scenario())


def test_video_track_composes_only_exact_overlay_identity() -> None:
    class OneFrameProducer:
        def __init__(self) -> None:
            self.sent = False

        async def read(self) -> RawVideoFrame | None:
            if self.sent:
                await asyncio.sleep(1.0)
                return None
            self.sent = True
            return RawVideoFrame(
                width=16,
                height=16,
                pitch=64,
                format=VIDEO_FORMAT_BGRA32,
                frame_id=5,
                ts_ms=700,
                payload=bytes((10, 20, 30, 255)) * 16 * 16,
                stream_id="camera",
                stream_epoch="epoch",
            )

        async def close(self) -> None:
            return None

    async def scenario() -> None:
        store = OverlayStore()
        await store.publish(
            OverlayResult(
                source="source",
                stream_id="camera",
                stream_epoch="epoch",
                frame_id=5,
                capture_timestamp_ms=700,
                detections=(OverlayDetection(x=0.25, y=0.25, width=0.5, height=0.5),),
            )
        )
        hub = LatestFrameHub(source="source", producer=OneFrameProducer())
        hub.start()
        track = LatestFrameVideoTrack(
            hub,
            MEDIA_QUALITIES["main"],
            session_id="session",
            overlay_store=store,
        )
        rendered = await asyncio.wait_for(track.recv(), timeout=1.0)
        bgra = rendered.reformat(format="bgra")
        plane = bytes(bgra.planes[0])
        pixel = plane[4 * bgra.planes[0].line_size + 4 * 4 : 4 * bgra.planes[0].line_size + 4 * 4 + 4]
        blue, green, red, _ = pixel
        assert green > red
        assert green > blue
        assert store.counters.matched == 1
        mapping = track.mapping(rendered.pts or 0)
        assert mapping is not None
        assert mapping.frame_id == 5
        assert mapping.stream_epoch == "epoch"
        track.stop()
        await hub.close()

    asyncio.run(scenario())


def test_sessions_share_same_source_and_isolate_different_sources() -> None:
    created_sources: list[str] = []

    def producer_factory(source: str) -> SyntheticFrameProducer:
        created_sources.append(source)
        return SyntheticFrameProducer(width=320, height=180, fps=30)

    async def connect(
        manager: MediaSessionManager,
        source: str,
    ) -> tuple[RTCPeerConnection, MediaStreamTrack, str]:
        client = RTCPeerConnection()
        remote_track: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()

        @client.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if not remote_track.done():
                remote_track.set_result(track)

        client.addTransceiver("video", direction="recvonly")
        offer = await client.createOffer()
        await client.setLocalDescription(offer)
        local = client.localDescription
        answer = await manager.create(
            MediaSessionOffer(source=source, quality="thumbnail", sdp=local.sdp, type=local.type)
        )
        await client.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))
        return client, await asyncio.wait_for(remote_track, timeout=5.0), answer.session_id

    async def scenario() -> None:
        manager = MediaSessionManager(producer_factory=producer_factory)
        first_peer, first_track, first_id = await connect(manager, "source-a")
        second_peer, second_track, second_id = await connect(manager, "source-a")
        third_peer, third_track, third_id = await connect(manager, "source-b")
        assert manager.session_count == 3
        assert manager.source_count == 2
        assert created_sources == ["source-a", "source-b"]

        decoded = await asyncio.gather(
            asyncio.wait_for(first_track.recv(), timeout=5.0),
            asyncio.wait_for(second_track.recv(), timeout=5.0),
            asyncio.wait_for(third_track.recv(), timeout=5.0),
        )
        assert all(isinstance(frame, VideoFrame) for frame in decoded)
        assert await manager.close_session(first_id) is True
        assert manager.source_count == 2
        assert await manager.close_session(second_id) is True
        assert manager.source_count == 1
        assert await manager.close_session(third_id) is True
        assert manager.session_count == 0
        assert manager.source_count == 0
        await asyncio.gather(first_peer.close(), second_peer.close(), third_peer.close())
        await manager.close()

    asyncio.run(scenario())


def test_raw_frame_history_is_bounded() -> None:
    async def scenario() -> None:
        hub = LatestFrameHub(
            source="history",
            producer=SyntheticFrameProducer(width=32, height=18, fps=500),
        )
        hub.start()
        await asyncio.sleep(0.1)
        assert 1 <= hub.history_size <= 8
        await hub.close()

    asyncio.run(scenario())


def test_media_session_can_reopen_without_leaking_sources() -> None:
    async def scenario() -> None:
        manager = MediaSessionManager()
        for _ in range(5):
            client = RTCPeerConnection()
            client.addTransceiver("video", direction="recvonly")
            offer = await client.createOffer()
            await client.setLocalDescription(offer)
            local = client.localDescription
            answer = await manager.create(
                MediaSessionOffer(
                    source="synthetic://bars", quality="thumbnail", sdp=local.sdp, type=local.type
                )
            )
            await client.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))
            await wait_for_connected(client)
            assert await manager.close_session(answer.session_id) is True
            await client.close()
            assert manager.session_count == 0
            assert manager.source_count == 0
        await manager.close()

    asyncio.run(scenario())
