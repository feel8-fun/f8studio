from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from f8pysdk.audio_transport import (
    LatestAudioChunkTransport,
    SAMPLE_FORMAT_F32LE,
    ZenohLatestAudioChunkTransport,
)
from f8pysdk.bus import ServiceBus
from f8pysdk.codec import coerce_int
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode

from .constants import CORE_SCHEMA_VERSION
from .feature_math import compute_core_features, librosa_available

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoreDefaults:
    channel_mode: str = "mono_mix"
    window_ms: int = 768
    hop_ms: int = 64
    emit_every_hops: int = 1


class AudioCoreFeatureServiceNode(ServiceNode):
    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=["audio"],
            data_out_ports=["coreFeatures"],
            state_fields=[str(s.name) for s in list(node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._active = True
        self._task: asyncio.Task[object] | None = None

        self._channel_mode = self._coerce_channel_mode(self._initial_state.get("channelMode"))
        self._window_ms = coerce_int(self._initial_state.get("windowMs"), default=CoreDefaults.window_ms, minimum=64)
        self._hop_ms = coerce_int(self._initial_state.get("hopMs"), default=CoreDefaults.hop_ms, minimum=8)
        self._emit_every_hops = coerce_int(
            self._initial_state.get("emitEveryHops"), default=CoreDefaults.emit_every_hops, minimum=1
        )

        self._reader: LatestAudioChunkTransport | None = None
        self._opened_audio_key = ""
        self._last_seq = 0
        self._emit_seq = 0
        self._hop_counter = 0
        self._sample_ring = np.asarray([], dtype=np.float32)
        self._onset_history: deque[float] = deque(maxlen=256)

        self._last_error = ""
        self._last_error_signature = ""
        self._last_error_log_ms = 0
        self._zenoh_config_path: str | None = None
        self._zenoh_connect: tuple[str, ...] = ()
        self._zenoh_listen: tuple[str, ...] = ()
        self._zenoh_shm_pool_bytes = 256 * 1024 * 1024

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        if isinstance(bus, ServiceBus):
            cfg = bus.config
            self._zenoh_config_path = cfg.zenoh_config_path
            self._zenoh_connect = cfg.zenoh_connect
            self._zenoh_listen = cfg.zenoh_listen
            self._zenoh_shm_pool_bytes = cfg.zenoh_shm_pool_bytes
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._loop(), name=f"audiofeat:core:{self.node_id}")

    async def close(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._close_reader()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if field == "channelMode":
            self._channel_mode = self._coerce_channel_mode(value)
            return
        if field == "windowMs":
            self._window_ms = coerce_int(value, default=self._window_ms, minimum=64)
            return
        if field == "hopMs":
            self._hop_ms = coerce_int(value, default=self._hop_ms, minimum=8)
            return
        if field == "emitEveryHops":
            self._emit_every_hops = coerce_int(value, default=self._emit_every_hops, minimum=1)
            return

    @staticmethod
    def _coerce_channel_mode(value: Any) -> str:
        raw = str(value or CoreDefaults.channel_mode).strip().lower()
        if raw == "left":
            return "left"
        if raw == "right":
            return "right"
        return "mono_mix"

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000.0)

    async def _set_last_error(self, msg: str, *, signature: str, exc: BaseException | None = None) -> None:
        if self._last_error != msg:
            self._last_error = msg
            await self.report_error(
                "AUDIOFEAT_CORE_RUNTIME",
                msg,
                severity="error",
                fingerprint=f"audiofeat-core:{signature}",
            )

        now_ms = self._now_ms()
        if signature == self._last_error_signature and (now_ms - self._last_error_log_ms) < 2000:
            return

        self._last_error_signature = signature
        self._last_error_log_ms = now_ms
        if exc is None:
            logger.error("[%s] %s", self.node_id, msg)
            return
        logger.error("[%s] %s", self.node_id, msg, exc_info=(type(exc), exc, exc.__traceback__))

    async def _clear_last_error(self) -> None:
        if not self._last_error:
            return
        self._last_error = ""
        self._last_error_signature = ""
        await self.clear_error()

    def _close_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._opened_audio_key = ""
        self._last_seq = 0
        self._sample_ring = np.asarray([], dtype=np.float32)

    def _ensure_reader(self) -> None:
        audio_key = str(self.input_zenoh_key("audio") or "").strip()
        if self._reader is not None and self._opened_audio_key == audio_key:
            return
        self._close_reader()
        reader = ZenohLatestAudioChunkTransport.open_subscriber(
            audio_key,
            config_path=self._zenoh_config_path,
            connect=self._zenoh_connect,
            listen=self._zenoh_listen,
            shm_pool_bytes=self._zenoh_shm_pool_bytes,
        )
        self._reader = reader
        self._opened_audio_key = audio_key

    def _chunk_to_mono(self, payload: memoryview, *, frames: int, channels: int) -> np.ndarray:
        samples = np.frombuffer(payload, dtype=np.float32)
        matrix = samples.reshape((int(frames), int(channels)))
        if self._channel_mode == "left":
            return matrix[:, 0].astype(np.float32, copy=True)
        if self._channel_mode == "right":
            idx = 1 if channels > 1 else 0
            return matrix[:, idx].astype(np.float32, copy=True)
        if channels == 1:
            return matrix[:, 0].astype(np.float32, copy=True)
        return np.asarray(np.mean(matrix, axis=1, dtype=np.float32), dtype=np.float32)

    async def _loop(self) -> None:
        if not librosa_available():
            await self._set_last_error("librosa not available", signature="missing_librosa")
        while True:
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._set_last_error("core loop failure", signature=f"loop:{type(exc).__name__}:{exc}", exc=exc)
                await asyncio.sleep(0.05)

    async def _step(self) -> None:
        if not self._active:
            await asyncio.sleep(0.02)
            return

        audio_key = str(self.input_zenoh_key("audio") or "").strip()
        if not audio_key:
            await self._set_last_error("missing audio data input", signature="missing_audio_input")
            await asyncio.sleep(0.05)
            return

        try:
            self._ensure_reader()
        except FileNotFoundError as exc:
            await self._set_last_error(
                "audio input open failed", signature="input_open:FileNotFoundError", exc=exc
            )
            await asyncio.sleep(0.05)
            return
        except (OSError, RuntimeError, ValueError) as exc:
            await self._set_last_error("audio input open failed", signature=f"input_open:{type(exc).__name__}", exc=exc)
            await asyncio.sleep(0.05)
            return

        reader = self._reader
        if reader is None:
            await asyncio.sleep(0.05)
            return

        chunk = reader.poll_latest()
        if chunk is None:
            await asyncio.sleep(0.01)
            return

        if int(chunk.fmt) != int(SAMPLE_FORMAT_F32LE):
            chunk.release()
            await self._set_last_error("audio format must be f32le", signature="bad_format")
            await asyncio.sleep(0.05)
            return
        seq = int(chunk.seq)
        frames = int(chunk.frames)
        channels = int(chunk.channels)
        if seq <= 0 or seq == int(self._last_seq) or frames <= 0 or channels <= 0:
            chunk.release()
            await asyncio.sleep(0.005)
            return

        try:
            mono = self._chunk_to_mono(chunk.payload, frames=frames, channels=channels)
        finally:
            chunk.release()
        self._sample_ring = np.concatenate((self._sample_ring, mono))

        sample_rate = int(chunk.sample_rate)
        window_length = max(32, int(round(sample_rate * float(self._window_ms) / 1000.0)))
        hop_length = max(8, int(round(sample_rate * float(self._hop_ms) / 1000.0)))

        max_ring = max(window_length * 3, window_length + hop_length)
        if int(self._sample_ring.size) > int(max_ring):
            self._sample_ring = self._sample_ring[-int(max_ring) :]

        if int(self._sample_ring.size) < int(window_length):
            self._last_seq = seq
            await asyncio.sleep(0.001)
            return

        rms, centroid_hz, onset_env = compute_core_features(
            mono=self._sample_ring,
            sample_rate=sample_rate,
            window_length=window_length,
            hop_length=hop_length,
        )
        onset_strength = float(onset_env[-1]) if onset_env.size > 0 else 0.0
        self._onset_history.append(onset_strength)

        self._hop_counter += 1
        self._last_seq = seq
        await self._clear_last_error()

        if (self._hop_counter % int(self._emit_every_hops)) != 0:
            await asyncio.sleep(0.001)
            return

        self._emit_seq += 1
        payload_out = {
            "schemaVersion": CORE_SCHEMA_VERSION,
            "tsMs": int(chunk.ts_ms),
            "seq": int(self._emit_seq),
            "sampleRate": int(sample_rate),
            "hopLength": int(hop_length),
            "windowLength": int(window_length),
            "rms": float(rms),
            "spectralCentroidHz": float(centroid_hz),
            "onsetStrength": float(onset_strength),
            "onsetEnvelope": [float(v) for v in self._onset_history],
        }
        await self.emit("coreFeatures", payload_out, ts_ms=int(chunk.ts_ms))
        await asyncio.sleep(0.001)
