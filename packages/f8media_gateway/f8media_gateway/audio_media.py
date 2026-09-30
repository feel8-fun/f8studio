from __future__ import annotations

from .session_manager import SessionManager

import asyncio
import logging
import math
import struct
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol

from aiortc import AudioStreamTrack, RTCPeerConnection
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame
import numpy as np

from f8pysdk.audio_transport import (
    SAMPLE_FORMAT_F32LE,
    LatestAudioChunk,
    ZenohLatestAudioChunkTransport,
)

from f8media_protocol.models import MediaInputError
from f8media_protocol.models import AudioSessionAnswer, AudioSessionOffer

from f8pysdk.binary_stream_transport import SharedStreamSession


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
    def __init__(self, source: str, session: SharedStreamSession | None = None) -> None:
        self._transport = (
            ZenohLatestAudioChunkTransport.open_subscriber(source)
            if session is None
            else ZenohLatestAudioChunkTransport(
                key_expr=source, raw_transport=session.subscribe(source, log_context="audio", max_pending_samples=16)
            )
        )
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


def create_audio_producer(source: str, session: SharedStreamSession | None = None) -> AsyncAudioProducer:
    normalized = source.strip()
    if normalized == "synthetic://tone":
        return SyntheticToneProducer()
    if normalized == "synthetic://tone-stereo":
        return SyntheticToneProducer(channels=2)
    if not normalized.startswith("f8/") or len(normalized) > 512:
        raise MediaInputError("audio source must be synthetic://tone, synthetic://tone-stereo, or an f8/ Zenoh key")
    return ZenohAudioProducer(normalized, session)


def audio_frame(chunk: RawAudioChunk) -> AudioFrame:
    validate_audio_chunk(chunk)
    samples = np.frombuffer(chunk.payload, dtype="<f4").astype(np.float64)
    samples[~np.isfinite(samples)] = 0
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
    frame = AudioFrame(format="s16", layout="mono" if chunk.channels == 1 else "stereo", samples=chunk.frames)
    frame.sample_rate = AUDIO_SAMPLE_RATE
    frame.planes[0].update(pcm.tobytes())
    return frame


def validate_audio_chunk(chunk: RawAudioChunk) -> None:
    if chunk.sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError(f"audio sample rate must be 48000 Hz, received {chunk.sample_rate}")
    if chunk.channels not in {1, 2}:
        raise ValueError(f"audio channels must be mono or stereo, received {chunk.channels}")
    expected_bytes = chunk.frames * chunk.channels * 4
    if chunk.frames <= 0 or len(chunk.payload) != expected_bytes:
        raise ValueError("audio payload size does not match frames and channels")


@dataclass
class LatestAudioHub:
    source: str
    producer: AsyncAudioProducer
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition, init=False)
    _pending: deque[RawAudioChunk] = field(default_factory=lambda: deque(maxlen=16), init=False)
    _version: int = field(default=0, init=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False)
    _closed: bool = field(default=False, init=False)

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"audio-source:{self.source}")

    async def _run(self) -> None:
        try:
            while not self._closed:
                chunk = await self.producer.read()
                if chunk is None:
                    continue
                validate_audio_chunk(chunk)
                async with self._condition:
                    self._pending.append(chunk)
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
            if not self._pending or self._closed:
                raise MediaStreamError
            first_version = self._version - len(self._pending) + 1
            next_version = max(after_version + 1, first_version)
            return next_version, self._pending[next_version - first_version]

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
    hub: LatestAudioHub
    source: str
    peer: RTCPeerConnection
    track: LatestAudioTrack


class AudioSessionManager(SessionManager[LatestAudioHub, AudioSession]):
    def __init__(
        self,
        *,
        producer_factory: Callable[[str], AsyncAudioProducer] = create_audio_producer,
        disconnected_grace_s: float = 10.0,
    ) -> None:
        self._producer_factory = producer_factory
        super().__init__(hub_factory=self._make_hub, disconnected_grace_s=disconnected_grace_s, kind="audio")
        self._closed_dropped_chunks = 0

    @property
    def dropped_chunks(self) -> int:
        return self._closed_dropped_chunks + sum(session.track.dropped_chunks for session in self._sessions.values())

    async def create(self, offer: AudioSessionOffer) -> AudioSessionAnswer:
        source = offer.source.strip()

        def build(hub: LatestAudioHub, session_id: str, peer: RTCPeerConnection) -> AudioSession:
            return AudioSession(hub=hub, session_id=session_id, source=source, peer=peer, track=LatestAudioTrack(hub))

        session, local = await self._create_session(source=source, sdp=offer.sdp, offer_type=offer.type, build=build)
        return AudioSessionAnswer(
            session_id=session.session_id,
            source=source,
            sdp=local.sdp,
            type="answer",
            sample_rate=AUDIO_SAMPLE_RATE,
            channels=1 if source == "synthetic://tone" else 2 if source == "synthetic://tone-stereo" else 0,
            transport_policy="bounded-queue-16; overflow gaps are dropped and counted",
        )

    def _session_closed(self, session: AudioSession) -> None:
        self._closed_dropped_chunks += session.track.dropped_chunks

    def _make_hub(self, source: str) -> LatestAudioHub:
        return LatestAudioHub(source=source, producer=self._producer_factory(source))


__all__ = [
    "AUDIO_SAMPLE_RATE",
    "AudioSessionManager",
    "LatestAudioHub",
    "LatestAudioTrack",
    "RawAudioChunk",
    "SyntheticToneProducer",
    "audio_frame",
]
