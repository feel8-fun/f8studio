from __future__ import annotations

import asyncio
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any

PKG_AUDIOFEAT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PKG_AUDIOFEAT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from f8pyaudiofeat.core_service_node import AudioCoreFeatureServiceNode  # noqa: E402
from f8pyaudiofeat.feature_math import librosa_available  # noqa: E402
from f8pysdk.nodes import RuntimeNode  # noqa: E402
from f8pysdk.state import StateRead  # noqa: E402


@dataclass(frozen=True)
class _StateField:
    name: str


@dataclass(frozen=True)
class _NodeStub:
    stateFields: list[_StateField]


class _FakeBus:
    def __init__(self) -> None:
        self.state_values: dict[str, Any] = {}
        self.errors: list[tuple[str, str, str]] = []
        self.emits: list[tuple[str, str, Any, int | None, str | int | None]] = []

    async def emit_data(
        self,
        node_id: str,
        port: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        ctx_id: str | int | None = None,
    ) -> None:
        self.emits.append((node_id, port, value, ts_ms, ctx_id))

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

    def data_input_zenoh_key(self, node_id: str, port: str) -> str | None:
        del node_id, port
        return None

    def report_error(
        self,
        node_id: str,
        code: str,
        message: str,
        severity: str = "error",
        fingerprint: str | None = None,
        ts_ms: int | None = None,
    ) -> None:
        del severity, fingerprint, ts_ms
        self.errors.append((str(node_id), str(code), str(message)))

    def clear_error(self, node_id: str, fingerprint: str | None = None, ts_ms: int | None = None) -> None:
        del node_id, fingerprint, ts_ms
        self.errors.clear()


@unittest.skipUnless(librosa_available(), "librosa is required")
class AudioFeatNodeTests(unittest.TestCase):
    def test_missing_audio_data_input_sets_error(self) -> None:
        async def _run() -> None:
            node = AudioCoreFeatureServiceNode(node_id="audio_core", node=_NodeStub(stateFields=[]), initial_state={})
            bus = _FakeBus()
            RuntimeNode.attach(node, bus)
            await node._step()
            self.assertEqual(bus.errors[-1][2], "missing audio data input")

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
