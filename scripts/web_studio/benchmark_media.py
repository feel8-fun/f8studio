from __future__ import annotations

import argparse
import asyncio
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import VideoFrame

from f8media_gateway.media import MediaSessionManager
from f8media_protocol.models import MediaSessionOffer


class VideoReceiver(Protocol):
    async def recv(self) -> object: ...


@dataclass
class PeerSession:
    peer: RTCPeerConnection
    receiver: VideoReceiver
    session_id: str
    quality: str


def _rss_bytes() -> int:
    status = Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            return int(parts[1]) * 1024
    raise RuntimeError("VmRSS is unavailable in /proc status")


def _require_resource_counts(manager: MediaSessionManager, *, sessions: int, sources: int) -> None:
    actual = (manager.session_count, manager.source_count)
    expected = (sessions, sources)
    if actual != expected:
        raise RuntimeError(f"media resource counts are {actual}, expected {expected}")


async def _connect(manager: MediaSessionManager, quality: str) -> PeerSession:
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
        MediaSessionOffer(
            source="synthetic://bars-1080p",
            quality=quality,
            sdp=local.sdp,
            type=local.type,
        )
    )
    await peer.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))
    track = await asyncio.wait_for(remote_track, timeout=10.0)
    deadline = asyncio.get_running_loop().time() + 10.0
    while peer.connectionState != "connected":
        if peer.connectionState in {"closed", "failed"}:
            raise RuntimeError(f"peer entered {peer.connectionState} before connecting")
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("peer did not connect within 10 seconds")
        await asyncio.sleep(0.01)
    return PeerSession(
        peer=peer,
        receiver=cast(VideoReceiver, track),
        session_id=answer.session_id,
        quality=quality,
    )


async def _count_frames(session: PeerSession, duration_s: float) -> tuple[int, int, int]:
    deadline = asyncio.get_running_loop().time() + duration_s
    count = 0
    width = 0
    height = 0
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return count, width, height
        try:
            decoded = await asyncio.wait_for(session.receiver.recv(), timeout=min(remaining, 2.0))
        except TimeoutError:
            continue
        if not isinstance(decoded, VideoFrame):
            raise TypeError(f"expected VideoFrame, received {type(decoded).__name__}")
        count += 1
        width = decoded.width
        height = decoded.height


async def _close_sessions(manager: MediaSessionManager, sessions: list[PeerSession]) -> None:
    for session in sessions:
        await manager.close_session(session.session_id)
    await asyncio.gather(*(session.peer.close() for session in sessions))


async def _reopen_cycles(manager: MediaSessionManager, cycles: int) -> None:
    for _ in range(cycles):
        session = await _connect(manager, "thumbnail")
        await manager.close_session(session.session_id)
        await session.peer.close()
        _require_resource_counts(manager, sessions=0, sources=0)


async def _warm_media_stack(manager: MediaSessionManager, thumbnail_count: int) -> None:
    sessions = [await _connect(manager, "main")]
    for _ in range(thumbnail_count):
        sessions.append(await _connect(manager, "thumbnail"))
    try:
        for session in sessions:
            decoded = await asyncio.wait_for(session.receiver.recv(), timeout=10.0)
            if not isinstance(decoded, VideoFrame):
                raise TypeError(f"expected VideoFrame, received {type(decoded).__name__}")
    finally:
        await _close_sessions(manager, sessions)
    gc.collect()
    await asyncio.sleep(0.5)


async def run_benchmark(duration_s: float, thumbnail_count: int, reopen_cycles: int) -> None:
    manager = MediaSessionManager()
    sessions: list[PeerSession] = []
    try:
        await _warm_media_stack(manager, thumbnail_count)
        _require_resource_counts(manager, sessions=0, sources=0)
        baseline_rss = _rss_bytes()

        sessions.append(await _connect(manager, "main"))
        for _ in range(thumbnail_count):
            sessions.append(await _connect(manager, "thumbnail"))
        _require_resource_counts(manager, sessions=thumbnail_count + 1, sources=1)

        await asyncio.sleep(2.0)
        cpu_started = time.process_time()
        wall_started = time.monotonic()
        frame_results = await asyncio.gather(*(_count_frames(session, duration_s) for session in sessions))
        wall_elapsed = time.monotonic() - wall_started
        cpu_elapsed = time.process_time() - cpu_started
        loaded_rss = _rss_bytes()
        await _close_sessions(manager, sessions)
        sessions.clear()
        gc.collect()
        await asyncio.sleep(0.5)
        post_load_rss = _rss_bytes()

        await _reopen_cycles(manager, reopen_cycles)
        gc.collect()
        await asyncio.sleep(0.5)
        final_rss = _rss_bytes()
        _require_resource_counts(manager, sessions=0, sources=0)

        main_count, main_width, main_height = frame_results[0]
        thumbnail_results = frame_results[1:]
        main_fps = main_count / wall_elapsed
        thumbnail_fps = [count / wall_elapsed for count, _, _ in thumbnail_results]
        cold_to_warm_growth = post_load_rss - baseline_rss
        reopen_rss_growth = final_rss - post_load_rss
        rss_budget = max(50 * 1024 * 1024, post_load_rss // 10)
        print(
            "Media benchmark complete\n"
            f"  duration={wall_elapsed:.2f}s cpu={cpu_elapsed / wall_elapsed * 100:.1f}%\n"
            f"  main={main_width}x{main_height} frames={main_count} fps={main_fps:.2f}\n"
            f"  thumbnails={thumbnail_count} fps={[round(value, 2) for value in thumbnail_fps]}\n"
            f"  rss_baseline={baseline_rss / 1024 / 1024:.1f}MiB "
            f"rss_loaded={loaded_rss / 1024 / 1024:.1f}MiB "
            f"rss_post_load={post_load_rss / 1024 / 1024:.1f}MiB "
            f"rss_final={final_rss / 1024 / 1024:.1f}MiB\n"
            f"  cold_to_warm_growth={cold_to_warm_growth / 1024 / 1024:.1f}MiB\n"
            f"  reopen_cycles={reopen_cycles} rss_growth={reopen_rss_growth / 1024 / 1024:.1f}MiB "
            f"rss_budget={rss_budget / 1024 / 1024:.1f}MiB"
        )
        if reopen_rss_growth > rss_budget:
            raise RuntimeError("RSS growth exceeded the close/reopen budget")
    finally:
        if sessions:
            await _close_sessions(manager, sessions)
        await manager.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the shared WebRTC media source and resource cleanup.")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--thumbnails", type=int, default=4)
    parser.add_argument("--reopen-cycles", type=int, default=50)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.thumbnails < 0 or args.reopen_cycles < 0:
        parser.error("--thumbnails and --reopen-cycles must be non-negative")
    return args


def main() -> int:
    args = _parse_args()
    asyncio.run(
        run_benchmark(
            duration_s=cast(float, args.duration),
            thumbnail_count=cast(int, args.thumbnails),
            reopen_cycles=cast(int, args.reopen_cycles),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
