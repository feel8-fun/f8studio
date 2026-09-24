from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import uuid4

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import VideoFrame

from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32, ZenohLatestVideoFrameTransport
from f8media_gateway.media import MediaSessionManager
from f8media_protocol.models import MediaFrameMapping, MediaSessionOffer


class VideoReceiver(Protocol):
    async def recv(self) -> object: ...


@dataclass
class ConnectedSession:
    peer: RTCPeerConnection
    receiver: VideoReceiver
    session_id: str


async def connect(manager: MediaSessionManager, source: str) -> ConnectedSession:
    peer = RTCPeerConnection()
    remote_track: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()

    @peer.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if not remote_track.done():
            remote_track.set_result(track)

    peer.addTransceiver("video", direction="recvonly")
    offer = await peer.createOffer()
    await peer.setLocalDescription(offer)
    local = peer.localDescription
    answer = await manager.create(
        MediaSessionOffer(source=source, quality="thumbnail", sdp=local.sdp, type=local.type)
    )
    await peer.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))
    return ConnectedSession(
        peer=peer,
        receiver=cast(VideoReceiver, await asyncio.wait_for(remote_track, timeout=5.0)),
        session_id=answer.session_id,
    )


async def publish_until_stopped(
    publisher: ZenohLatestVideoFrameTransport,
    stop: asyncio.Event,
    *,
    color: tuple[int, int, int, int],
) -> None:
    width, height = 320, 180
    payload = bytes(color) * width * height
    while not stop.is_set():
        publisher.publish_frame(
            width=width,
            height=height,
            pitch=width * 4,
            payload=payload,
            fmt=VIDEO_FORMAT_BGRA32,
        )
        await asyncio.sleep(1.0 / 30.0)


async def receive_mapping(
    manager: MediaSessionManager,
    session: ConnectedSession,
    *,
    different_epoch: str | None = None,
    timeout_s: float = 5.0,
) -> MediaFrameMapping:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        decoded = await asyncio.wait_for(session.receiver.recv(), timeout=timeout_s)
        if not isinstance(decoded, VideoFrame) or decoded.pts is None:
            continue
        mapping = await manager.frame_mapping(session.session_id, decoded.pts)
        if mapping is not None and (different_epoch is None or mapping.stream_epoch != different_epoch):
            return mapping
    raise TimeoutError("stream did not resume with a new producer epoch within five seconds")


async def run_probe() -> None:
    key = f"f8/test/web-studio/restart/{uuid4().hex}"
    manager = MediaSessionManager()
    first_publisher = ZenohLatestVideoFrameTransport.open_publisher(key)
    first_stop = asyncio.Event()
    first_task = asyncio.create_task(
        publish_until_stopped(first_publisher, first_stop, color=(20, 40, 220, 255)),
        name="restart-probe-publisher-1",
    )
    session: ConnectedSession | None = None
    second_publisher: ZenohLatestVideoFrameTransport | None = None
    second_stop = asyncio.Event()
    second_task: asyncio.Task[None] | None = None
    try:
        session = await connect(manager, key)
        first = await receive_mapping(manager, session)
        first_stop.set()
        await first_task
        first_publisher.close()

        await asyncio.sleep(0.5)
        restarted_at = time.monotonic()
        second_publisher = ZenohLatestVideoFrameTransport.open_publisher(key)
        second_task = asyncio.create_task(
            publish_until_stopped(second_publisher, second_stop, color=(220, 40, 20, 255)),
            name="restart-probe-publisher-2",
        )
        second = await receive_mapping(manager, session, different_epoch=first.stream_epoch)
        recovered_in = time.monotonic() - restarted_at
        if recovered_in > 5.0:
            raise RuntimeError(f"stream recovery took {recovered_in:.3f}s")
        if manager.session_count != 1 or manager.source_count != 1:
            raise RuntimeError("stream restart duplicated the media session or source")
        print(
            "Stream restart probe passed: "
            f"recovered_in={recovered_in:.3f}s first_epoch={first.stream_epoch} "
            f"second_epoch={second.stream_epoch} sessions=1 sources=1"
        )
    finally:
        first_stop.set()
        if not first_task.done():
            await first_task
        if second_task is not None:
            second_stop.set()
            await second_task
        if second_publisher is not None:
            second_publisher.close()
        if session is not None:
            await manager.close_session(session.session_id)
            await session.peer.close()
        await manager.close()
        if manager.session_count != 0 or manager.source_count != 0:
            raise RuntimeError("stream restart probe leaked media resources")


def main() -> int:
    asyncio.run(run_probe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
