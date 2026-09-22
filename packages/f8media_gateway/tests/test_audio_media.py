import asyncio
import math
import struct

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame

from f8media_gateway.audio_media import (
    AUDIO_SAMPLE_RATE,
    AudioSessionManager,
    RawAudioChunk,
    SyntheticToneProducer,
    audio_frame,
)
from f8media_protocol.models import AudioSessionOffer


def test_f32_audio_is_explicitly_converted_to_s16() -> None:
    chunk = RawAudioChunk(
        sample_rate=AUDIO_SAMPLE_RATE,
        channels=1,
        frames=4,
        sequence=1,
        frame_index=4,
        timestamp_ms=100,
        payload=struct.pack("<ffff", -1.0, -0.5, math.nan, 1.5),
    )
    frame = audio_frame(chunk)
    assert frame.format.name == "s16"
    assert frame.layout.name == "mono"
    assert frame.sample_rate == AUDIO_SAMPLE_RATE
    assert struct.unpack("<hhhh", bytes(frame.planes[0])[:8]) == (-32767, -16384, 0, 32767)


def test_synthetic_audio_webrtc_delivers_non_silent_frames_and_releases_source() -> None:
    async def scenario() -> None:
        manager = AudioSessionManager()
        client = RTCPeerConnection()
        remote_track: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()

        @client.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if not remote_track.done():
                remote_track.set_result(track)

        client.addTransceiver("audio", direction="recvonly")
        offer = await client.createOffer()
        await client.setLocalDescription(offer)
        local = client.localDescription
        answer = await manager.create(
            AudioSessionOffer(source="synthetic://tone", sdp=local.sdp, type=local.type)
        )
        await client.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))
        track = await asyncio.wait_for(remote_track, timeout=5.0)
        decoded = await asyncio.wait_for(track.recv(), timeout=5.0)
        assert isinstance(decoded, AudioFrame)
        assert decoded.sample_rate == AUDIO_SAMPLE_RATE
        assert any(bytes(decoded.planes[0]))
        assert answer.transport_policy.startswith("latest-chunk")
        assert manager.session_count == 1
        assert manager.source_count == 1

        assert await manager.close_session(answer.session_id) is True
        await client.close()
        assert manager.session_count == 0
        assert manager.source_count == 0
        await manager.close()

    asyncio.run(scenario())


def test_audio_sessions_share_one_source_hub() -> None:
    created_sources: list[str] = []

    def producer_factory(source: str) -> SyntheticToneProducer:
        created_sources.append(source)
        return SyntheticToneProducer()

    async def offer(manager: AudioSessionManager) -> tuple[RTCPeerConnection, str]:
        client = RTCPeerConnection()
        client.addTransceiver("audio", direction="recvonly")
        created = await client.createOffer()
        await client.setLocalDescription(created)
        local = client.localDescription
        answer = await manager.create(
            AudioSessionOffer(source="shared", sdp=local.sdp, type=local.type)
        )
        await client.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))
        return client, answer.session_id

    async def scenario() -> None:
        manager = AudioSessionManager(producer_factory=producer_factory)
        first, first_id = await offer(manager)
        second, second_id = await offer(manager)
        assert created_sources == ["shared"]
        assert manager.source_count == 1
        await manager.close_session(first_id)
        assert manager.source_count == 1
        await manager.close_session(second_id)
        assert manager.source_count == 0
        await asyncio.gather(first.close(), second.close())
        await manager.close()

    asyncio.run(scenario())
