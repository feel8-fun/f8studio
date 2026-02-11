import os
import sys
import time
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

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.sequence_player import SequencePlayerRuntimeNode, register_operator  # noqa: E402


class SequencePlayerTests(unittest.IsolatedAsyncioTestCase):
    async def test_outputs_expected_value_for_elapsed_time(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        now_ms = int(time.time() * 1000.0)
        start_ms = now_ms - 250  # safely inside step index=2 for stepMs=100

        op = F8RuntimeNode(
            nodeId="seq1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=SequencePlayerRuntimeNode.SPEC.operatorClass,
            stateFields=list(SequencePlayerRuntimeNode.SPEC.stateFields or []),
            stateValues={"sequence": {"tsMs": start_ms, "stepMs": 100, "values": [1.0, 2.0, 3.0], "timeSec": 0}},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("seq1")
        self.assertIsInstance(node, SequencePlayerRuntimeNode)
        assert isinstance(node, SequencePlayerRuntimeNode)

        idx = await node.compute_output("index", ctx_id="i")
        value = await node.compute_output("value", ctx_id="v")
        self.assertIsInstance(idx, int)
        self.assertEqual(int(idx), 2)
        self.assertAlmostEqual(float(value), 3.0, places=6)

    async def test_done_outputs_zero(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        now_ms = int(time.time() * 1000.0)
        start_ms = now_ms - 600  # timeSec=0.2 => done by now

        op = F8RuntimeNode(
            nodeId="seq1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=SequencePlayerRuntimeNode.SPEC.operatorClass,
            stateFields=list(SequencePlayerRuntimeNode.SPEC.stateFields or []),
            stateValues={
                "sequence": {
                    "tsMs": start_ms,
                    "stepMs": 100,
                    "values": [1.0, 2.0, 3.0],
                    "timeSec": 0.2,
                }
            },
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("seq1")
        value = await node.compute_output("value", ctx_id="v")
        active = await node.compute_output("active", ctx_id="a")
        done = await node.compute_output("done", ctx_id="d")
        self.assertAlmostEqual(float(value), 0.0, places=6)
        self.assertEqual(bool(active), False)
        self.assertEqual(bool(done), True)

    async def test_index_wraps_when_looping(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        now_ms = int(time.time() * 1000.0)
        start_ms = now_ms - 1250  # stepMs=100 => idx_raw=12 -> wraps to 0 for n=3

        op = F8RuntimeNode(
            nodeId="seq1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=SequencePlayerRuntimeNode.SPEC.operatorClass,
            stateFields=list(SequencePlayerRuntimeNode.SPEC.stateFields or []),
            stateValues={"sequence": {"tsMs": start_ms, "stepMs": 100, "values": [1.0, 2.0, 3.0], "timeSec": 0}},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("seq1")
        idx = await node.compute_output("index", ctx_id="i")
        value = await node.compute_output("value", ctx_id="v")
        self.assertEqual(int(idx), 0)
        self.assertAlmostEqual(float(value), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
