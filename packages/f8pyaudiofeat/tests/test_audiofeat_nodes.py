from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

PKG_AUDIOFEAT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_AUDIOFEAT, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)

from f8pyaudiofeat.constants import CORE_SCHEMA_VERSION, RHYTHM_SCHEMA_VERSION  # noqa: E402
from f8pyaudiofeat.core_service_node import AudioCoreFeatureServiceNode  # noqa: E402
from f8pyaudiofeat.feature_math import librosa_available  # noqa: E402
from f8pyaudiofeat.rhythm_service_node import AudioRhythmFeatureServiceNode  # noqa: E402
from f8pysdk.runtime_node import RuntimeNode  # noqa: E402
from f8pysdk.service_bus.state_read import StateRead  # noqa: E402
from f8pysdk.shm.audio import AudioShmWriter  # noqa: E402


@dataclass(frozen=True)
class _StateField:
    name: str


@dataclass(frozen=True)
class _NodeStub:
    stateFields: list[_StateField]


class _FakeBus:
    def __init__(self) -> None:
        self.state_values: dict[str, Any] = {}
        self.emits: list[tuple[str, str, Any, int | None]] = []

    async def emit_data(self, node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        self.emits.append((node_id, port, value, ts_ms))

    async def publish_state_runtime(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del node_id
        del ts_ms
        self.state_values[str(field)] = value

    async def get_state(self, node_id: str, field: str) -> StateRead:
        del node_id
        key = str(field)
        if key in self.state_values:
            return StateRead(found=True, value=self.state_values[key], ts_ms=0)
        return StateRead(found=False, value=None, ts_ms=None)

    def get_state_cached(self, node_id: str, field: str, default: Any) -> Any:
        del node_id
        return self.state_values.get(str(field), default)


@unittest.skipUnless(librosa_available(), "librosa is required")
class AudioFeatNodeTests(unittest.TestCase):
    def test_missing_audio_shm_name_sets_error(self) -> None:
        async def _run() -> None:
            node = AudioCoreFeatureServiceNode(node_id="audio_core", node=_NodeStub(stateFields=[]), initial_state={})
            bus = _FakeBus()
            RuntimeNode.attach(node, bus)
            await node._step()
            self.assertEqual(bus.state_values.get("lastError"), "missing audioShmName")

        asyncio.run(_run())

    def test_duplicate_frame_is_not_emitted_twice(self) -> None:
        async def _run() -> None:
            shm_name = f"shm_test_audiofeat_{int(time.time() * 1000)}"
            writer = AudioShmWriter(
                shm_name=shm_name,
                size=8 * 1024 * 1024,
                sample_rate=48_000,
                channels=2,
                frames_per_chunk=480,
                chunk_count=200,
            )
            writer.open()
            try:
                node = AudioCoreFeatureServiceNode(
                    node_id="audio_core",
                    node=_NodeStub(stateFields=[]),
                    initial_state={"audioShmName": shm_name, "windowMs": 64, "hopMs": 16},
                )
                bus = _FakeBus()
                RuntimeNode.attach(node, bus)

                sine = np.sin(2.0 * np.pi * 440.0 * (np.arange(480, dtype=np.float32) / 48_000.0)).astype(np.float32)
                interleaved = np.empty((480 * 2,), dtype=np.float32)
                interleaved[0::2] = sine
                interleaved[1::2] = sine
                for _ in range(12):
                    writer.write_chunk_f32(interleaved.tobytes(), frames=480)
                    await node._step()
                emit_count_1 = len(bus.emits)
                self.assertGreaterEqual(emit_count_1, 1)

                for _ in range(3):
                    await node._step()
                emit_count_2 = len(bus.emits)
                self.assertEqual(emit_count_1, emit_count_2)
            finally:
                writer.close(unlink=False)

        asyncio.run(_run())

    def test_core_to_rhythm_chain(self) -> None:
        async def _run() -> None:
            shm_name = f"shm_test_audiofeat_chain_{int(time.time() * 1000)}"
            writer = AudioShmWriter(
                shm_name=shm_name,
                size=8 * 1024 * 1024,
                sample_rate=48_000,
                channels=2,
                frames_per_chunk=480,
                chunk_count=200,
            )
            writer.open()
            try:
                core = AudioCoreFeatureServiceNode(
                    node_id="audio_core",
                    node=_NodeStub(stateFields=[]),
                    initial_state={"audioShmName": shm_name, "windowMs": 64, "hopMs": 16},
                )
                rhythm = AudioRhythmFeatureServiceNode(
                    node_id="audio_rhythm",
                    node=_NodeStub(stateFields=[]),
                    initial_state={},
                )
                core_bus = _FakeBus()
                rhythm_bus = _FakeBus()
                RuntimeNode.attach(core, core_bus)
                RuntimeNode.attach(rhythm, rhythm_bus)

                for i in range(60):
                    burst = np.zeros((480,), dtype=np.float32)
                    if (i % 8) == 0:
                        burst[:64] = 0.95
                    interleaved = np.empty((480 * 2,), dtype=np.float32)
                    interleaved[0::2] = burst
                    interleaved[1::2] = burst
                    writer.write_chunk_f32(interleaved.tobytes(), frames=480)
                    await core._step()

                core_payloads = [item for item in core_bus.emits if item[1] == "coreFeatures"]
                self.assertGreaterEqual(len(core_payloads), 1)
                latest_core = core_payloads[-1][2]
                self.assertEqual(latest_core.get("schemaVersion"), CORE_SCHEMA_VERSION)

                await rhythm.on_data("coreFeatures", latest_core)
                rhythm_payloads = [item for item in rhythm_bus.emits if item[1] == "rhythmFeatures"]
                self.assertGreaterEqual(len(rhythm_payloads), 1)
                latest_rhythm = rhythm_payloads[-1][2]
                self.assertEqual(latest_rhythm.get("schemaVersion"), RHYTHM_SCHEMA_VERSION)
                self.assertIn("tempoBpm", latest_rhythm)
                self.assertIn("pulseClarity", latest_rhythm)
            finally:
                writer.close(unlink=False)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
