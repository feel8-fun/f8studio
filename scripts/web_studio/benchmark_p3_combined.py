from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame, VideoFrame

from f8media_protocol.models import AudioSessionOffer, MediaSessionOffer
from f8media_gateway.service import InProcessMediaGateway
from f8studio_server import create_app
from f8studio_server.application import StudioApplication


class MediaReceiver(Protocol):
    async def recv(self) -> object: ...


@dataclass
class VideoPeer:
    peer: RTCPeerConnection
    receiver: MediaReceiver
    session_id: str
    quality: str


@dataclass
class AudioPeer:
    peer: RTCPeerConnection
    receiver: MediaReceiver
    session_id: str


@dataclass(frozen=True)
class CombinedBenchmarkResult:
    duration_s: float
    with_audio: bool
    cpu_percent: float
    main_fps: float
    main_width: int
    main_height: int
    thumbnail_fps: tuple[float, ...]
    audio_frames: int
    presentation_events: int
    presentation_queue_max: int
    control_requests: int
    control_p95_ms: float
    rss_start_mib: float
    rss_loaded_mib: float
    rss_end_mib: float
    rss_samples_mib: tuple[float, ...]
    rss_growth_last_5m_mib: float
    task_count_loaded: int
    task_count_end: int
    task_count_max: int
    video_sessions_end: int
    video_sources_end: int
    audio_sessions_end: int
    audio_sources_end: int
    errors: tuple[str, ...]


def rss_bytes() -> int:
    status = Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable in /proc status")


def percentile_95(values: list[float]) -> float:
    if not values:
        raise RuntimeError("no control latency samples were recorded")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, (len(ordered) * 95 + 99) // 100 - 1))]


async def connect_video(studio: StudioApplication, quality: str) -> VideoPeer:
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
    answer = await studio.media_gateway.create_video_session(
        MediaSessionOffer(
            source="synthetic://bars-1080p",
            quality=quality,
            sdp=local.sdp,
            type=local.type,
        )
    )
    await peer.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))
    return VideoPeer(
        peer=peer,
        receiver=cast(MediaReceiver, await asyncio.wait_for(remote_track, timeout=10.0)),
        session_id=answer.session_id,
        quality=quality,
    )


async def connect_audio(studio: StudioApplication) -> AudioPeer:
    peer = RTCPeerConnection()
    remote_track: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()

    @peer.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if not remote_track.done():
            remote_track.set_result(track)

    peer.addTransceiver("audio", direction="recvonly")
    offer = await peer.createOffer()
    await peer.setLocalDescription(offer)
    local = peer.localDescription
    answer = await studio.media_gateway.create_audio_session(
        AudioSessionOffer(source="synthetic://tone", sdp=local.sdp, type=local.type)
    )
    await peer.setRemoteDescription(RTCSessionDescription(sdp=answer.sdp, type=answer.type))
    return AudioPeer(
        peer=peer,
        receiver=cast(MediaReceiver, await asyncio.wait_for(remote_track, timeout=10.0)),
        session_id=answer.session_id,
    )


async def count_video(peer: VideoPeer, stop: asyncio.Event) -> tuple[int, int, int]:
    frames = 0
    width = 0
    height = 0
    while not stop.is_set():
        try:
            decoded = await asyncio.wait_for(peer.receiver.recv(), timeout=2.0)
        except TimeoutError:
            continue
        if not isinstance(decoded, VideoFrame):
            raise TypeError(f"expected VideoFrame, received {type(decoded).__name__}")
        frames += 1
        width, height = decoded.width, decoded.height
    return frames, width, height


async def count_audio(peer: AudioPeer, stop: asyncio.Event) -> int:
    frames = 0
    while not stop.is_set():
        try:
            decoded = await asyncio.wait_for(peer.receiver.recv(), timeout=2.0)
        except TimeoutError:
            continue
        if not isinstance(decoded, AudioFrame):
            raise TypeError(f"expected AudioFrame, received {type(decoded).__name__}")
        frames += 1
    return frames


async def drive_presentation(
    studio: StudioApplication,
    stop: asyncio.Event,
    counters: dict[str, int],
) -> None:
    frame_id = 0
    while not stop.is_set():
        frame_id += 1
        studio.presentation.emit(
            "benchmark-three",
            "viz.three_d.set",
            {
                "tsMs": int(time.time() * 1_000),
                "worldUp": "+y",
                "people": [
                    {
                        "name": "benchmark",
                        "bbox": None,
                        "skeletonProtocol": "benchmark/1",
                        "skeletonEdges": [[0, 1]],
                        "nodes": [
                            {"index": 0, "name": "root", "pos": [0, 0, 0], "rot": None},
                            {"index": 1, "name": "tip", "pos": [0, 1 + frame_id % 10 / 20, 0], "rot": None},
                        ],
                    }
                ],
            },
        )
        counters["published"] = counters.get("published", 0) + 1
        await asyncio.sleep(0.1)


async def consume_presentation(
    studio: StudioApplication,
    stop: asyncio.Event,
    counters: dict[str, int],
) -> None:
    stream = await studio.events.open_stream(client_epoch=None, after_sequence=None)
    try:
        while not stop.is_set():
            counters["queue_max"] = max(counters.get("queue_max", 0), stream.queue.qsize())
            try:
                event = await asyncio.wait_for(stream.queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            if event.type == "presentation.command":
                counters["consumed"] = counters.get("consumed", 0) + 1
    finally:
        await studio.events.close_stream(stream.subscription_id)


async def drive_control_api(
    client: httpx.AsyncClient,
    stop: asyncio.Event,
    latencies_ms: list[float],
    errors: list[str],
) -> None:
    iteration = 0
    while not stop.is_set():
        iteration += 1
        started = time.perf_counter()
        try:
            if iteration % 4 == 0:
                response = await client.put(
                    "/api/projects/p3-benchmark",
                    json={"name": "P3 combined benchmark", "description": f"control-{iteration % 2}"},
                )
            else:
                path = ("/api/health", "/api/catalog", "/api/runtime/monitors")[iteration % 3]
                response = await client.get(path)
            response.raise_for_status()
            latencies_ms.append((time.perf_counter() - started) * 1_000)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            errors.append(f"control:{type(exc).__name__}:{exc}")
        await asyncio.sleep(0.5)


async def sample_resources(
    stop: asyncio.Event,
    rss_samples: list[float],
    task_samples: list[int],
) -> None:
    while True:
        rss_samples.append(rss_bytes() / 1024 / 1024)
        task_samples.append(len(asyncio.all_tasks()))
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=10.0)
        except TimeoutError:
            continue


async def close_peers(studio: StudioApplication, videos: list[VideoPeer], audio: AudioPeer | None) -> None:
    for peer in videos:
        await studio.media_gateway.close_video_session(peer.session_id)
    await asyncio.gather(*(peer.peer.close() for peer in videos))
    if audio is not None:
        await studio.media_gateway.close_audio_session(audio.session_id)
        await audio.peer.close()


async def run_benchmark(duration_s: float, *, with_audio: bool = False) -> CombinedBenchmarkResult:
    errors: list[str] = []
    latencies_ms: list[float] = []
    presentation: dict[str, int] = {}
    rss_samples: list[float] = []
    task_samples: list[int] = []
    stop = asyncio.Event()
    videos: list[VideoPeer] = []
    audio: AudioPeer | None = None
    rss_start = rss_bytes()
    with tempfile.TemporaryDirectory(prefix="f8studio-p3-bench-") as temp_dir:
        gateway = InProcessMediaGateway()
        studio = StudioApplication(data_dir=Path(temp_dir), media_gateway=gateway)
        app = create_app(web_dist=Path(temp_dir), application=studio)
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                created = await client.post(
                    "/api/projects",
                    json={"projectId": "p3-benchmark", "name": "P3 combined benchmark"},
                )
                created.raise_for_status()
                videos.append(await connect_video(studio, "main"))
                for _ in range(4):
                    videos.append(await connect_video(studio, "thumbnail"))
                if with_audio:
                    audio = await connect_audio(studio)
                rss_loaded = rss_bytes()
                tasks_loaded = len(asyncio.all_tasks())
                video_tasks = [asyncio.create_task(count_video(peer, stop)) for peer in videos]
                audio_task = asyncio.create_task(count_audio(audio, stop)) if audio is not None else None
                presentation_producer = asyncio.create_task(drive_presentation(studio, stop, presentation))
                presentation_consumer = asyncio.create_task(consume_presentation(studio, stop, presentation))
                control_task = asyncio.create_task(drive_control_api(client, stop, latencies_ms, errors))
                resource_task = asyncio.create_task(sample_resources(stop, rss_samples, task_samples))
                cpu_started = time.process_time()
                wall_started = time.monotonic()
                await asyncio.sleep(duration_s)
                measure_ended = time.monotonic()
                cpu_ended = time.process_time()
                stop.set()
                results = await asyncio.gather(*video_tasks, return_exceptions=True)
                side_tasks: list[asyncio.Task[object]] = [
                    cast(asyncio.Task[object], presentation_producer),
                    cast(asyncio.Task[object], presentation_consumer),
                    cast(asyncio.Task[object], control_task),
                    cast(asyncio.Task[object], resource_task),
                ]
                if audio_task is not None:
                    side_tasks.append(cast(asyncio.Task[object], audio_task))
                side_results = await asyncio.gather(*side_tasks, return_exceptions=True)
                audio_result = side_results[-1] if audio_task is not None else 0
                elapsed = measure_ended - wall_started
                cpu_elapsed = cpu_ended - cpu_started
                for result in (*results, audio_result):
                    if isinstance(result, BaseException):
                        errors.append(f"worker:{type(result).__name__}:{result}")
                typed_video_results = [
                    result for result in results if not isinstance(result, BaseException)
                ]
                if len(typed_video_results) != 5 or not isinstance(audio_result, int):
                    raise RuntimeError("one or more media workers failed")
                await close_peers(studio, videos, audio)
                videos.clear()
                audio = None
                gc.collect()
                await asyncio.sleep(1.0)
                rss_end = rss_bytes()
                tasks_end = len(asyncio.all_tasks())
                main_frames, main_width, main_height = typed_video_results[0]
                thumbnail_fps = tuple(result[0] / elapsed for result in typed_video_results[1:])
                five_minute_samples = 30
                growth_origin = max(0, len(rss_samples) - five_minute_samples - 1)
                rss_growth_last_5m = rss_samples[-1] - rss_samples[growth_origin]
                return CombinedBenchmarkResult(
                    duration_s=elapsed,
                    with_audio=with_audio,
                    cpu_percent=cpu_elapsed / elapsed * 100,
                    main_fps=main_frames / elapsed,
                    main_width=main_width,
                    main_height=main_height,
                    thumbnail_fps=thumbnail_fps,
                    audio_frames=audio_result,
                    presentation_events=presentation.get("consumed", 0),
                    presentation_queue_max=presentation.get("queue_max", 0),
                    control_requests=len(latencies_ms),
                    control_p95_ms=percentile_95(latencies_ms),
                    rss_start_mib=rss_start / 1024 / 1024,
                    rss_loaded_mib=rss_loaded / 1024 / 1024,
                    rss_end_mib=rss_end / 1024 / 1024,
                    rss_samples_mib=tuple(rss_samples),
                    rss_growth_last_5m_mib=rss_growth_last_5m,
                    task_count_loaded=tasks_loaded,
                    task_count_end=tasks_end,
                    task_count_max=max(task_samples, default=tasks_loaded),
                    video_sessions_end=gateway.video.session_count,
                    video_sources_end=gateway.video.source_count,
                    audio_sessions_end=gateway.audio.session_count,
                    audio_sources_end=gateway.audio.source_count,
                    errors=tuple(errors),
                )
        finally:
            stop.set()
            if videos or audio is not None:
                await close_peers(studio, videos, audio)
            await studio.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the combined P3 video, audio, 3D-event, and control benchmark.")
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-audio", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def main() -> int:
    args = parse_args()
    result = asyncio.run(
        run_benchmark(cast(float, args.duration), with_audio=cast(bool, args.with_audio))
    )
    payload = asdict(result)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    output = cast(Path | None, args.output)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    if result.errors:
        raise RuntimeError(f"combined benchmark recorded errors: {result.errors}")
    if result.control_p95_ms > 200:
        raise RuntimeError(f"control response p95 exceeded 200 ms: {result.control_p95_ms:.2f} ms")
    if result.duration_s >= 600 and result.rss_growth_last_5m_mib > 50:
        raise RuntimeError(
            f"RSS grew more than 50 MiB during the final five minutes: {result.rss_growth_last_5m_mib:.1f} MiB"
        )
    if any((result.video_sessions_end, result.video_sources_end, result.audio_sessions_end, result.audio_sources_end)):
        raise RuntimeError("combined benchmark leaked media resources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
