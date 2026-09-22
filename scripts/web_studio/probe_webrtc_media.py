from __future__ import annotations

import argparse
import asyncio
import zlib
from collections.abc import Sequence
from typing import Protocol, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import msgspec
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import VideoFrame

from f8media_protocol.models import MediaSessionAnswer, MediaSessionOffer


class VideoReceiver(Protocol):
    async def recv(self) -> object: ...


def _request(base_url: str, method: str, path: str, *, body: object | None = None) -> bytes:
    encoded = None if body is None else msgspec.json.encode(body)
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=encoded,
        headers={} if encoded is None else {"content-type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=20.0) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc


async def run_probe(base_url: str, quality: str, source: str) -> None:
    peer = RTCPeerConnection()
    remote_track: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()
    session_id: str | None = None

    @peer.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if not remote_track.done():
            remote_track.set_result(track)

    try:
        peer.addTransceiver("video", direction="recvonly")
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        local = peer.localDescription
        raw_answer = await asyncio.to_thread(
            _request,
            base_url,
            "POST",
            "/api/media/sessions",
            body=MediaSessionOffer(
                source=source, quality=quality,
                sdp=local.sdp, type=local.type,
            ),
        )
        answer = msgspec.json.decode(raw_answer, type=MediaSessionAnswer)
        session_id = answer.session_id
        await peer.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))

        track = await asyncio.wait_for(remote_track, timeout=10.0)
        receiver = cast(VideoReceiver, track)
        frames: list[VideoFrame] = []
        for _ in range(3):
            decoded = await asyncio.wait_for(receiver.recv(), timeout=10.0)
            if not isinstance(decoded, VideoFrame):
                raise TypeError(f"expected VideoFrame, received {type(decoded).__name__}")
            frames.append(decoded)
        checksums = [zlib.crc32(bytes(frame.reformat(format="rgb24").planes[0])) for frame in frames]
        timestamps: list[int] = []
        for frame in frames:
            if frame.pts is None:
                raise RuntimeError("decoded frame does not have an RTP timestamp")
            timestamps.append(frame.pts)
        if len(set(checksums)) < 2:
            raise RuntimeError(f"decoded frames did not change: checksums={checksums}")
        if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
            raise RuntimeError(f"decoded frame timestamps are not monotonic: {timestamps}")
        print(
            "WebRTC media probe passed: "
            f"session={session_id} quality={answer.quality} "
            f"decoded={frames[0].width}x{frames[0].height} checksums={checksums}"
        )
    finally:
        await peer.close()
        if session_id is not None:
            await asyncio.to_thread(_request, base_url, "DELETE", f"/api/media/sessions/{session_id}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Negotiate WebRTC through the real HTTP API and decode changing frames.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8210")
    parser.add_argument("--quality", choices=("thumbnail", "main"), default="thumbnail")
    parser.add_argument("--source", default="synthetic://bars")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    asyncio.run(run_probe(cast(str, args.base_url), cast(str, args.quality), cast(str, args.source)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
