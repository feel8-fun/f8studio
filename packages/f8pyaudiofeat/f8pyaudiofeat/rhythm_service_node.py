from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import ServiceNode

from .constants import CORE_SCHEMA_VERSION, RHYTHM_SCHEMA_VERSION
from .feature_math import compute_pulse_clarity, compute_tempo_bpm, librosa_available, select_recent_onset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RhythmDefaults:
    tempo_window_sec: float = 8.0
    pulse_window_sec: float = 6.0
    emit_every: int = 1


class AudioRhythmFeatureServiceNode(ServiceNode):
    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=["coreFeatures"],
            data_out_ports=["rhythmFeatures"],
            state_fields=[str(s.name) for s in list(node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._active = True

        self._tempo_window_sec = self._coerce_float(
            self._initial_state.get("tempoWindowSec"), default=RhythmDefaults.tempo_window_sec, minimum=1.0
        )
        self._pulse_window_sec = self._coerce_float(
            self._initial_state.get("pulseWindowSec"), default=RhythmDefaults.pulse_window_sec, minimum=1.0
        )
        self._emit_every = self._coerce_int(self._initial_state.get("emitEvery"), default=RhythmDefaults.emit_every, minimum=1)

        self._emit_counter = 0
        self._emit_seq = 0

        self._last_error = ""
        self._last_error_signature = ""
        self._last_error_log_ms = 0

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if field == "tempoWindowSec":
            self._tempo_window_sec = self._coerce_float(value, default=self._tempo_window_sec, minimum=1.0)
            return
        if field == "pulseWindowSec":
            self._pulse_window_sec = self._coerce_float(value, default=self._pulse_window_sec, minimum=1.0)
            return
        if field == "emitEvery":
            self._emit_every = self._coerce_int(value, default=self._emit_every, minimum=1)
            return

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if port != "coreFeatures":
            return
        if not self._active:
            return
        if not isinstance(value, dict):
            await self._set_last_error("coreFeatures payload must be object", signature="bad_payload_type")
            return
        await self._process_core_payload(value)

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
    def _coerce_float(value: Any, *, default: float, minimum: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = float(default)
        if out < float(minimum):
            return float(minimum)
        return out

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

    async def _process_core_payload(self, payload: dict[str, Any]) -> None:
        try:
            schema_version = str(payload.get("schemaVersion") or "")
            if schema_version != CORE_SCHEMA_VERSION:
                await self._set_last_error("unsupported coreFeatures schemaVersion", signature="bad_schema")
                return

            sample_rate = int(payload.get("sampleRate") or 0)
            hop_length = int(payload.get("hopLength") or 0)
            ts_ms = int(payload.get("tsMs") or self._now_ms())

            onset_raw = payload.get("onsetEnvelope")
            if not isinstance(onset_raw, list):
                await self._set_last_error("onsetEnvelope must be array", signature="bad_onset")
                return

            onset_values: list[float] = []
            for item in onset_raw:
                onset_values.append(float(item))

            if sample_rate <= 0 or hop_length <= 0:
                await self._set_last_error("sampleRate/hopLength must be positive", signature="bad_sr_hop")
                return

            if not librosa_available():
                await self._set_last_error("librosa not available", signature="missing_librosa")
                return

            hops_per_second = float(sample_rate) / float(hop_length)
            tempo_hops = max(4, int(round(float(self._tempo_window_sec) * hops_per_second)))
            pulse_hops = max(4, int(round(float(self._pulse_window_sec) * hops_per_second)))

            onset_arr = np.asarray(onset_values, dtype=np.float32)
            onset_for_tempo = select_recent_onset(onset_arr, hops=tempo_hops)
            onset_for_pulse = select_recent_onset(onset_arr, hops=pulse_hops)

            tempo_bpm = compute_tempo_bpm(onset_envelope=onset_for_tempo, sample_rate=sample_rate, hop_length=hop_length)
            beat_period_ms = 0.0
            if tempo_bpm > 1e-6:
                beat_period_ms = 60000.0 / float(tempo_bpm)
            pulse_clarity = compute_pulse_clarity(onset_for_pulse)
            onset_mean = float(np.mean(onset_for_pulse)) if onset_for_pulse.size > 0 else 0.0
            onset_std = float(np.std(onset_for_pulse)) if onset_for_pulse.size > 0 else 0.0

            self._emit_counter += 1
            if (self._emit_counter % int(self._emit_every)) != 0:
                await self._clear_last_error()
                return

            self._emit_seq += 1
            out = {
                "schemaVersion": RHYTHM_SCHEMA_VERSION,
                "tsMs": int(ts_ms),
                "seq": int(self._emit_seq),
                "tempoBpm": float(tempo_bpm),
                "beatPeriodMs": float(beat_period_ms),
                "pulseClarity": float(pulse_clarity),
                "onsetStrengthMean": float(onset_mean),
                "onsetStrengthStd": float(onset_std),
            }
            await self.emit("rhythmFeatures", out, ts_ms=int(ts_ms))
            await self._clear_last_error()
        except (TypeError, ValueError) as exc:
            await self._set_last_error("invalid coreFeatures payload", signature=f"payload:{type(exc).__name__}", exc=exc)
        except Exception as exc:
            await self._set_last_error("rhythm processing failed", signature=f"process:{type(exc).__name__}:{exc}", exc=exc)
