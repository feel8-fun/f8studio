from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import ServiceNode
from f8pysdk.shm.audio import (
    AUDIO_SHM_MAGIC,
    AUDIO_SHM_VERSION,
    SAMPLE_FORMAT_F32LE,
    AudioShmReader,
)

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
            data_in_ports=[],
            data_out_ports=["coreFeatures"],
            state_fields=[str(s.name) for s in list(node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._active = True
        self._task: asyncio.Task[object] | None = None

        self._audio_shm_name = self._coerce_str(self._initial_state.get("audioShmName"), default="")
        self._channel_mode = self._coerce_channel_mode(self._initial_state.get("channelMode"))
        self._window_ms = self._coerce_int(self._initial_state.get("windowMs"), default=CoreDefaults.window_ms, minimum=64)
        self._hop_ms = self._coerce_int(self._initial_state.get("hopMs"), default=CoreDefaults.hop_ms, minimum=8)
        self._emit_every_hops = self._coerce_int(
            self._initial_state.get("emitEveryHops"), default=CoreDefaults.emit_every_hops, minimum=1
        )

        self._reader: AudioShmReader | None = None
        self._opened_shm_name = ""
        self._last_seq = 0
        self._emit_seq = 0
        self._hop_counter = 0
        self._sample_ring = np.asarray([], dtype=np.float32)
        self._onset_history: deque[float] = deque(maxlen=256)

        self._last_error = ""
        self._last_error_signature = ""
        self._last_error_log_ms = 0

    def attach(self, bus: Any) -> None:
        super().attach(bus)
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
        if field == "audioShmName":
            self._audio_shm_name = self._coerce_str(value, default="")
            self._close_reader()
            return
        if field == "channelMode":
            self._channel_mode = self._coerce_channel_mode(value)
            return
        if field == "windowMs":
            self._window_ms = self._coerce_int(value, default=self._window_ms, minimum=64)
            return
        if field == "hopMs":
            self._hop_ms = self._coerce_int(value, default=self._hop_ms, minimum=8)
            return
        if field == "emitEveryHops":
            self._emit_every_hops = self._coerce_int(value, default=self._emit_every_hops, minimum=1)
            return

    @staticmethod
    def _coerce_str(value: Any, *, default: str) -> str:
        if value is None:
            return default
        text = str(value).strip()
        if text:
            return text
        return default

    @staticmethod
    def _coerce_int(value: Any, *, default: int, minimum: int) -> int:
        try:
            out = int(value)
        except (TypeError, ValueError):
            out = int(default)
        if out < int(minimum):
            return int(minimum)
        return out

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
            await self.set_state("lastError", msg)

        now_ms = self._now_ms()
        if signature == self._last_error_signature and (now_ms - self._last_error_log_ms) < 2000:
            return

        self._last_error_signature = signature
        self._last_error_log_ms = now_ms
        if exc is None:
            logger.error("[%s] %s", self.node_id, msg)
            return
        logger.exception("[%s] %s", self.node_id, msg, exc_info=exc)

    async def _clear_last_error(self) -> None:
        if not self._last_error:
            return
        self._last_error = ""
        self._last_error_signature = ""
        await self.set_state("lastError", "")

    def _close_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._opened_shm_name = ""
        self._last_seq = 0
        self._sample_ring = np.asarray([], dtype=np.float32)

    def _ensure_reader(self) -> None:
        if self._reader is not None and self._opened_shm_name == self._audio_shm_name:
            return
        self._close_reader()
        reader = AudioShmReader(self._audio_shm_name)
        reader.open(use_event=False)
        self._reader = reader
        self._opened_shm_name = self._audio_shm_name

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
        return np.mean(matrix, axis=1, dtype=np.float32)

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

        if not self._audio_shm_name:
            await self._set_last_error("missing audioShmName", signature="missing_shm")
            await asyncio.sleep(0.05)
            return

        try:
            self._ensure_reader()
        except FileNotFoundError as exc:
            await self._set_last_error("audio shm not found", signature="shm_not_found", exc=exc)
            await asyncio.sleep(0.05)
            return
        except RuntimeError as exc:
            await self._set_last_error("audio shm open failed", signature="shm_open_runtime", exc=exc)
            await asyncio.sleep(0.05)
            return

        reader = self._reader
        if reader is None:
            await asyncio.sleep(0.05)
            return

        hdr = reader.read_header()
        if hdr is None:
            await self._set_last_error("audio shm header unavailable", signature="bad_header")
            await asyncio.sleep(0.02)
            return

        if hdr.magic != AUDIO_SHM_MAGIC or hdr.version != AUDIO_SHM_VERSION:
            await self._set_last_error("audio shm header mismatch", signature="bad_magic")
            await asyncio.sleep(0.05)
            return

        if int(hdr.fmt) != int(SAMPLE_FORMAT_F32LE):
            await self._set_last_error("audio shm format must be f32le", signature="bad_format")
            await asyncio.sleep(0.05)
            return

        seq = int(hdr.write_seq)
        if seq <= 0 or seq == int(self._last_seq):
            await asyncio.sleep(0.01)
            return

        hdr2, chunk_hdr, payload = reader.read_chunk_f32(seq)
        if hdr2 is None or chunk_hdr is None or payload is None:
            await asyncio.sleep(0.005)
            return

        frames = int(chunk_hdr.frames)
        channels = int(hdr2.channels)
        if frames <= 0 or channels <= 0:
            await asyncio.sleep(0.005)
            return

        mono = self._chunk_to_mono(payload, frames=frames, channels=channels)
        self._sample_ring = np.concatenate((self._sample_ring, mono))

        sample_rate = int(hdr2.sample_rate)
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
            "tsMs": int(chunk_hdr.ts_ms),
            "seq": int(self._emit_seq),
            "sampleRate": int(sample_rate),
            "hopLength": int(hop_length),
            "windowLength": int(window_length),
            "rms": float(rms),
            "spectralCentroidHz": float(centroid_hz),
            "onsetStrength": float(onset_strength),
            "onsetEnvelope": [float(v) for v in self._onset_history],
        }
        await self.emit("coreFeatures", payload_out, ts_ms=int(chunk_hdr.ts_ms))
        await asyncio.sleep(0.001)
