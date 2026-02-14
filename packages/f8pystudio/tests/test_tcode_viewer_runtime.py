import asyncio
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_STUDIO, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pystudio.operators.tcode_viewer import PyStudioTCodeViewerRuntimeNode  # noqa: E402
from f8pystudio.ui_bus import UiCommand, set_ui_command_sink  # noqa: E402


@dataclass(frozen=True)
class _FakePort:
    name: str


@dataclass(frozen=True)
class _FakeState:
    name: str


@dataclass(frozen=True)
class _FakeNode:
    dataInPorts: list[_FakePort]
    dataOutPorts: list[_FakePort]
    stateFields: list[_FakeState]


def _make_runtime(*, initial_state: dict[str, Any] | None = None) -> PyStudioTCodeViewerRuntimeNode:
    fake = _FakeNode(
        dataInPorts=[_FakePort(name="tcode")],
        dataOutPorts=[],
        stateFields=[
            _FakeState(name="model"),
            _FakeState(name="throttleMs"),
            _FakeState(name="maxLineLength"),
        ],
    )
    return PyStudioTCodeViewerRuntimeNode(node_id="n1", node=fake, initial_state=initial_state or {})


class TCodeViewerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        set_ui_command_sink(None)

    async def test_normalize_newline_and_truncate(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))
        n = _make_runtime(initial_state={"throttleMs": 0})
        await n.on_state("maxLineLength", 32, ts_ms=900)
        await n.on_data("tcode", "L0000I500\r", ts_ms=1000)
        long_line = "R1234567890123456789012345678901234567890"
        await n.on_data("tcode", long_line, ts_ms=1010)

        write_cmds = [c for c in cmds if c.command == "tcode_viewer.write"]
        self.assertGreaterEqual(len(write_cmds), 2)
        self.assertEqual(write_cmds[0].payload.get("line"), "L0000I500\n")
        self.assertEqual(write_cmds[1].payload.get("line"), long_line[:32] + "\n")
        await n.close()

    async def test_model_change_set_model_and_reset_without_replay(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))
        n = _make_runtime(initial_state={"throttleMs": 0, "model": "SR6"})
        await n.on_data("tcode", "L0000I500", ts_ms=1000)
        await n.on_data("tcode", "L1000I500", ts_ms=1010)
        await n.on_data("tcode", "L2000I500", ts_ms=1020)
        writes_before_model_change = [c for c in cmds if c.command == "tcode_viewer.write"]
        await n.on_state("model", "OSR2", ts_ms=1030)

        commands = [c.command for c in cmds]
        self.assertIn("tcode_viewer.set_model", commands)
        self.assertIn("tcode_viewer.reset", commands)
        writes_after_model_change = [c for c in cmds if c.command == "tcode_viewer.write"]
        self.assertEqual(len(writes_after_model_change), len(writes_before_model_change))
        await n.close()

    async def test_throttle_batches_writes(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))
        n = _make_runtime(initial_state={"throttleMs": 80})
        await n.on_data("tcode", "L0000I500", ts_ms=1000)
        await n.on_data("tcode", "L1000I500", ts_ms=1010)
        await asyncio.sleep(0.11)

        writes = [c for c in cmds if c.command == "tcode_viewer.write"]
        self.assertGreaterEqual(len(writes), 2)
        self.assertEqual(writes[0].payload.get("line"), "L0000I500\n")
        self.assertEqual(writes[1].payload.get("line"), "L1000I500\n")
        await n.close()

    async def test_close_emits_detach(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))
        n = _make_runtime(initial_state={"throttleMs": 0})
        await n.on_data("tcode", "L0000I500", ts_ms=1000)
        await n.close()
        self.assertEqual(cmds[-1].command, "tcode_viewer.detach")


if __name__ == "__main__":
    unittest.main()
