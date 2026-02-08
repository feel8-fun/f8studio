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
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.mix_silence_fill import (  # noqa: E402
    MixSilenceFillRuntimeNode,
    register_operator,
)


class MixSilenceFillTests(unittest.IsolatedAsyncioTestCase):
    async def test_switches_to_b_after_silence_and_back_to_a(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="mix1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=MixSilenceFillRuntimeNode.SPEC.operatorClass,
            stateFields=list(MixSilenceFillRuntimeNode.SPEC.stateFields or []),
            stateValues={"silenceMs": 50, "deltaThreshold": 0.1, "fadeMs": 0},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("mix1")
        self.assertIsInstance(node, MixSilenceFillRuntimeNode)
        assert isinstance(node, MixSilenceFillRuntimeNode)

        # Feed inputs: A=1, B=0.
        bus._buffer_input("mix1", "A", 1.0, ts_ms=now_ms(), edge=None, ctx_id=None)
        bus._buffer_input("mix1", "B", 0.0, ts_ms=now_ms(), edge=None, ctx_id=None)

        out0 = await node.compute_output("out", ctx_id=0)
        self.assertAlmostEqual(float(out0), 1.0, places=6)

        # Keep A constant; after silenceMs it should switch to B (fadeMs=0 => instant).
        await asyncio.sleep(0.07)
        out1 = await node.compute_output("out", ctx_id=1)
        self.assertAlmostEqual(float(out1), 0.0, places=6)

        # Make A active (big change) -> should switch back to A.
        bus._buffer_input("mix1", "A", 0.4, ts_ms=now_ms(), edge=None, ctx_id=None)
        out2 = await node.compute_output("out", ctx_id=2)
        self.assertAlmostEqual(float(out2), 0.4, places=6)


if __name__ == "__main__":
    unittest.main()

