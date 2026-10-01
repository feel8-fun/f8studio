import asyncio
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.testing import buffer_input, ServiceBusHarness  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.silence_detector import SilenceDetectorRuntimeNode, register_operator  # noqa: E402


class SilenceDetectorTests(unittest.IsolatedAsyncioTestCase):
    def test_spec_excludes_activity_timestamp_state(self) -> None:
        state_names = {str(field.name or "") for field in list(SilenceDetectorRuntimeNode.SPEC.stateFields or [])}
        self.assertNotIn("lastActiveTsMs", state_names)

    async def test_sets_is_silent_after_input_stops_changing(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="sil1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=SilenceDetectorRuntimeNode.SPEC.operatorClass,
            stateFields=list(SilenceDetectorRuntimeNode.SPEC.stateFields or []),
            stateValues={"silenceMs": 50, "deltaThreshold": 0.1},
            execInPorts=list(SilenceDetectorRuntimeNode.SPEC.execInPorts or []),
            execOutPorts=list(SilenceDetectorRuntimeNode.SPEC.execOutPorts or []),
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("sil1")
        self.assertIsInstance(node, SilenceDetectorRuntimeNode)
        assert isinstance(node, SilenceDetectorRuntimeNode)

        buffer_input(bus, "sil1", "value", 1.0, ts_ms=now_ms(), edge=None, ctx_id=None)
        await node.on_exec(0)
        self.assertFalse(bool(await node.get_state_value("isSilent")))

        await asyncio.sleep(0.07)
        await node.on_exec(1)
        self.assertTrue(bool(await node.get_state_value("isSilent")))

        buffer_input(bus, "sil1", "value", 1.4, ts_ms=now_ms(), edge=None, ctx_id=None)
        await node.on_exec(2)
        self.assertFalse(bool(await node.get_state_value("isSilent")))


if __name__ == "__main__":
    unittest.main()
