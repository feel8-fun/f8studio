from __future__ import annotations

import asyncio
import logging
import math
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol
from uuid import uuid4

from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from f8pysdk.audio_transport import (
    SAMPLE_FORMAT_F32LE,
    LatestAudioChunk,
    ZenohLatestAudioChunkTransport,
)

from f8media_protocol.models import AudioSessionAnswer, AudioSessionOffer

from .peer_lifecycle import PeerCloseQueue


logger = logging.getLogger(__name__)
AUDIO_SAMPLE_RATE = 48_000
AUDIO_TIME_BASE = Fraction(1, AUDIO_SAMPLE_RATE)
SYNTHETIC_AUDIO_FRAMES = 960


@dataclass(frozen=True)
class RawAudioChunk:
    sample_rate: int
    channels: int
    frames: int
    sequence: int
    frame_index: int
    timestamp_ms: int
    payload: bytes


class AsyncAudioProducer(Protocol):
    async def read(self) -> RawAudioChunk | None: ...

    async def close(self) -> None: ...


class SyntheticToneProducer:
    def __init__(self, *, frequency_hz: float = 440.0, channels: int = 1) -> None:
        if channels not in {1, 2}:
            raise ValueError("synthetic audio supports one or two channels")
        self._frequency_hz = frequency_hz
        self._channels = channels
        self._sequence = 0
        self._frame_index = 0
        self._closed = False

    async def read(self) -> RawAudioChunk | None:
        if self._closed:
            return None
        await asyncio.sleep(SYNTHETIC_AUDIO_FRAMES / AUDIO_SAMPLE_RATE)
        start = self._frame_index
        samples = bytearray(SYNTHETIC_AUDIO_FRAMES * self._channels * 4)
        for frame_offset in range(SYNTHETIC_AUDIO_FRAMES):
            phase = 2.0 * math.pi * self._frequency_hz * (start + frame_offset) / AUDIO_SAMPLE_RATE
            sample = 0.2 * math.sin(phase)
            for channel in range(self._channels):
                struct.pack_into("<f", samples, (frame_offset * self._channels + channel) * 4, sample)
        self._sequence += 1
        self._frame_index += SYNTHETIC_AUDIO_FRAMES
        return RawAudioChunk(
            sample_rate=AUDIO_SAMPLE_RATE,
            channels=self._channels,
            frames=SYNTHETIC_AUDIO_FRAMES,
            sequence=self._sequence,
            frame_index=self._frame_index,
            timestamp_ms=int(time.time() * 1_000),
            payload=bytes(samples),
        )

    async def close(self) -> None:
        self._closed = True


class ZenohAudioProducer:
    def __init__(self, source: str) -> None:
        self._transport = ZenohLatestAudioChunkTransport.open_subscriber(source)
        self._closed = False

    async def read(self) -> RawAudioChunk | None:
        if self._closed:
            return None
        chunk = await asyncio.to_thread(self._transport.wait_latest, 250)
        if chunk is None:
            return None
        return self._copy_chunk(chunk)

    @staticmethod
    def _copy_chunk(chunk: LatestAudioChunk) -> RawAudioChunk:
        try:
            if chunk.fmt != SAMPLE_FORMAT_F32LE:
                raise ValueError(f"unsupported audio sample format {chunk.fmt}; expected f32le")
            return RawAudioChunk(
                sample_rate=chunk.sample_rate,
                channels=chunk.channels,
                frames=chunk.frames,
                sequence=chunk.seq,
                frame_index=chunk.frame_index,
                timestamp_ms=chunk.ts_ms,
                payload=chunk.payload_copy(),
            )
        finally:
            chunk.release()

    async def close(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._transport.close)


def create_audio_producer(source: str) -> AsyncAudioProducer:
    normalized = source.strip()
    if normalized == "synthetic://tone":
        return SyntheticToneProducer()
    if normalized == "synthetic://tone-stereo":
        return SyntheticToneProducer(channels=2)
    if not normalized.startswith("f8/") or len(normalized) > 512:
        raise ValueError(
            "audio source must be synthetic://tone, synthetic://tone-stereo, or an f8/ Zenoh key"
        )
    return ZenohAudioProducer(normalized)


def audio_frame(chunk: RawAudioChunk) -> AudioFrame:
    if chunk.sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError(f"audio sample rate must be 48000 Hz, received {chunk.sample_rate}")
    if chunk.channels not in {1, 2}:
        raise ValueError(f"audio channels must be mono or stereo, received {chunk.channels}")
    expected_bytes = chunk.frames * chunk.channels * 4
    if chunk.frames <= 0 or len(chunk.payload) != expected_bytes:
        raise ValueError("audio payload size does not match frames and channels")

    pcm = bytearray(chunk.frames * chunk.channels * 2)
    for index in range(chunk.frames * chunk.channels):
        sample = struct.unpack_from("<f", chunk.payload, index * 4)[0]
        if not math.isfinite(sample):
            sample = 0.0
        value = round(max(-1.0, min(1.0, sample)) * 32_767.0)
        struct.pack_into("<h", pcm, index * 2, value)
    frame = AudioFrame(format="s16", layout="mono" if chunk.channels == 1 else "stereo", samples=chunk.frames)
    frame.sample_rate = AUDIO_SAMPLE_RATE
    frame.planes[0].update(bytes(pcm))
    return frame


@dataclass
class LatestAudioHub:
    source: str
    producer: AsyncAudioProducer
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition, init=False)
    _latest: RawAudioChunk | None = field(default=None, init=False)
    _version: int = field(default=0, init=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False)
    _closed: bool = field(default=False, init=False)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"audio-source:{self.source}")

    async def _run(self) -> None:
        try:
            while not self._closed:
                chunk = await self.producer.read()
                if chunk is None:
                    continue
                audio_frame(chunk)
                async with self._condition:
                    self._latest = chunk
                    self._version += 1
                    self._condition.notify_all()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("audio source failed source=%s", self.source)
        finally:
            async with self._condition:
                self._closed = True
                self._condition.notify_all()

    async def next_chunk(self, after_version: int) -> tuple[int, RawAudioChunk]:
        async with self._condition:
            await self._condition.wait_for(lambda: self._version > after_version or self._closed)
            if self._latest is None or self._closed:
                raise MediaStreamError
            return self._version, self._latest

    async def close(self) -> None:
        self._closed = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.producer.close()
        async with self._condition:
            self._condition.notify_all()


class LatestAudioTrack(AudioStreamTrack):
    def __init__(self, hub: LatestAudioHub) -> None:
        super().__init__()
        self._hub = hub
        self._version = 0
        self._pts = 0
        self._last_sequence: int | None = None
        self.dropped_chunks = 0

    async def recv(self) -> AudioFrame:
        self._version, chunk = await self._hub.next_chunk(self._version)
        if self._last_sequence is not None and chunk.sequence > self._last_sequence + 1:
            self.dropped_chunks += chunk.sequence - self._last_sequence - 1
        self._last_sequence = chunk.sequence
        frame = audio_frame(chunk)
        frame.pts = self._pts
        frame.time_base = AUDIO_TIME_BASE
        self._pts += chunk.frames
        return frame


@dataclass
class AudioSession:
    session_id: str
    source: str
    peer: RTCPeerConnection
    track: LatestAudioTrack


class AudioSessionManager:
    def __init__(
        self,
        *,
        producer_factory: Callable[[str], AsyncAudioProducer] = create_audio_producer,
        disconnected_grace_s: float = 10.0,
    ) -> None:
        self._sessions: dict[str, AudioSession] = {}
        self._hubs: dict[str, tuple[LatestAudioHub, int]] = {}
        self._lock = asyncio.Lock()
        self._janitor: asyncio.Task[None] | None = None
        self._producer_factory = producer_factory
        self._disconnected_grace_s = disconnected_grace_s
        self._disconnected_since: dict[str, float] = {}
        self._closed_dropped_chunks = 0
        self._peer_closer = PeerCloseQueue()

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def source_count(self) -> int:
        return len(self._hubs)

    @property
    def dropped_chunks(self) -> int:
        return self._closed_dropped_chunks + sum(session.track.dropped_chunks for session in self._sessions.values())

    async def create(self, offer: AudioSessionOffer) -> AudioSessionAnswer:
        source = offer.source.strip()
        if offer.type != "offer" or not offer.sdp.strip():
            raise ValueError("a non-empty WebRTC offer SDP is required")
        hub = await self._acquire_hub(source)
        peer = RTCPeerConnection()
        session_id = uuid4().hex

        @peer.on("iceconnectionstatechange")
        async def ice_connection_state_changed() -> None:
            logger.info(
                "audio ICE state changed session_id=%s source=%s state=%s",
                session_id,
                source,
                peer.iceConnectionState,
            )

        @peer.on("connectionstatechange")
        async def connection_state_changed() -> None:
            logger.info(
                "audio peer state changed session_id=%s source=%s state=%s",
                session_id,
                source,
                peer.connectionState,
            )

        track = LatestAudioTrack(hub)
        peer.addTrack(track)
        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
        except Exception:
            logger.exception("failed to negotiate audio session source=%s", source)
            track.stop()
            await peer.close()
            await self._release_hub(source)
            raise

        async with self._lock:
            self._sessions[session_id] = AudioSession(
                session_id=session_id, source=source, peer=peer, track=track
            )
            if self._janitor is None:
                self._janitor = asyncio.create_task(self._run_janitor(), name="audio-session-janitor")
        local = peer.localDescription
        return AudioSessionAnswer(
            session_id=session_id,
            source=source,
            sdp=local.sdp,
            type=local.type,
            sample_rate=AUDIO_SAMPLE_RATE,
            channels=1 if source == "synthetic://tone" else 2 if source == "synthetic://tone-stereo" else 0,
            transport_policy="latest-chunk; sequence gaps are dropped and counted",
        )

    async def _acquire_hub(self, source: str) -> LatestAudioHub:
        async with self._lock:
            existing = self._hubs.get(source)
            if existing is not None:
                hub, references = existing
                self._hubs[source] = (hub, references + 1)
                return hub
            producer = self._producer_factory(source)
            hub = LatestAudioHub(source=source, producer=producer)
            hub.start()
            self._hubs[source] = (hub, 1)
            return hub

    async def _release_hub(self, source: str) -> None:
        hub_to_close: LatestAudioHub | None = None
        async with self._lock:
            existing = self._hubs.get(source)
            if existing is None:
                return
            hub, references = existing
            if references <= 1:
                del self._hubs[source]
                hub_to_close = hub
            else:
                self._hubs[source] = (hub, references - 1)
        if hub_to_close is not None:
            await hub_to_close.close()

    async def close_session(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            self._disconnected_since.pop(session_id, None)
        if session is None:
            return False
        self._closed_dropped_chunks += session.track.dropped_chunks
        session.track.stop()
        await self._peer_closer.close(session.peer, context=f"audio:{session_id}")
        await self._release_hub(session.source)
        return True

    async def _run_janitor(self) -> None:
        try:
            while True:
                await asyncio.sleep(2.0)
                now = asyncio.get_running_loop().time()
                async with self._lock:
                    stale: list[str] = []
                    for session_id, session in self._sessions.items():
                        state = session.peer.connectionState
                        if state in {"failed", "closed"}:
                            stale.append(session_id)
                        elif state == "disconnected":
                            disconnected_at = self._disconnected_since.setdefault(session_id, now)
                            if now - disconnected_at >= self._disconnected_grace_s:
                                stale.append(session_id)
                        else:
                            self._disconnected_since.pop(session_id, None)
                for session_id in stale:
                    await self.close_session(session_id)
        except asyncio.CancelledError:
            raise

    async def close(self) -> None:
        janitor = self._janitor
        self._janitor = None
        if janitor is not None:
            janitor.cancel()
            await asyncio.gather(janitor, return_exceptions=True)
        async with self._lock:
            session_ids = tuple(self._sessions)
        for session_id in session_ids:
            await self.close_session(session_id)
        await self._peer_closer.shutdown()


__all__ = [
    "AUDIO_SAMPLE_RATE",
    "AudioSessionManager",
    "LatestAudioHub",
    "LatestAudioTrack",
    "RawAudioChunk",
    "SyntheticToneProducer",
    "audio_frame",
]
