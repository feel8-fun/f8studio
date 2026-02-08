import asyncio
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.generated import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.service_bus.state_write import StateWriteOrigin  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.lovense_wave import (  # noqa: E402
    LovenseThrustingWaveRuntimeNode,
    register_operator,
)


class LovenseWaveNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_thrusting_phase_continuity_on_event_update(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")

        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)

        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        def mk_event(*, thrusting: int, depth: int, event_id: str) -> dict[str, object]:
            return {
                "eventId": event_id,
                "summary": {
                    "type": "solace_thrusting",
                    "thrusting": thrusting,
                    "depth": depth,
                    "timeSec": 0,
                    "loopRunningSec": None,
                    "loopPauseSec": None,
                    "apiVer": 1,
                },
                "raw": {"command": "Function", "action": f"Thrusting:{thrusting},Depth:{depth}"},
            }

        op = F8RuntimeNode(
            nodeId="thr1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=LovenseThrustingWaveRuntimeNode.SPEC.operatorClass,
            stateFields=list(LovenseThrustingWaveRuntimeNode.SPEC.stateFields or []),
            stateValues={
                # Fixed frequency so the test is deterministic-ish.
                "minHz": 2.0,
                "maxHz": 2.0,
                "slewMs": 0,
                "lovenseEvent": mk_event(thrusting=1, depth=10, event_id="e1"),
            },
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("thr1")
        self.assertIsInstance(node, LovenseThrustingWaveRuntimeNode)
        assert isinstance(node, LovenseThrustingWaveRuntimeNode)

        # Drive phase into a range where a reset to 0 would be a large jump.
        phase = None
        for i in range(200):
            phase = await node.compute_output("phase", ctx_id=i)
            await asyncio.sleep(0.01)
            if isinstance(phase, float) and 0.25 <= phase <= 0.45:
                break
        self.assertIsInstance(phase, float)
        phase_before = float(phase)

        # Publish a new event and ensure phase keeps accumulating (no reset).
        await bus._publish_state(
            "thr1",
            "lovenseEvent",
            mk_event(thrusting=5, depth=15, event_id="e2"),
            origin=StateWriteOrigin.external,
            source="test",
        )
        await asyncio.sleep(0.02)
        phase_after = await node.compute_output("phase", ctx_id="after")
        self.assertIsInstance(phase_after, float)

        # Compute modular delta. If phase resets to 0, delta ~= 1-phase_before (>~0.55 here).
        delta = (float(phase_after) - phase_before) % 1.0
        self.assertLess(delta, 0.3)

        out = await node.compute_output("out", ctx_id="out")
        self.assertIsInstance(out, float)
        self.assertGreaterEqual(float(out), 0.0)
        self.assertLessEqual(float(out), 1.0)


if __name__ == "__main__":
    unittest.main()

