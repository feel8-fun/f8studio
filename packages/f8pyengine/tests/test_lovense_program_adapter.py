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

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.lovense_program_adapter import LovenseProgramAdapterRuntimeNode, register_operator  # noqa: E402


class LovenseProgramAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_converts_solace_thrusting_event_to_program_and_amplitude(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="a1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=LovenseProgramAdapterRuntimeNode.SPEC.operatorClass,
            stateFields=list(LovenseProgramAdapterRuntimeNode.SPEC.stateFields or []),
            stateValues={
                "minHz": 1.0,
                "maxHz": 5.0,
                "thrustingMax": 20.0,
                "depthMax": 20.0,
                "speedGamma": 1.0,
            },
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        event = {
            "tsMs": 1700000000000,
            "command": {"name": "Function", "apiVer": 1, "kind": "solace_thrusting"},
            "params": {"action": "Thrusting:10,Depth:5", "timeSec": 20, "loopRunningSec": 9, "loopPauseSec": 4},
        }

        await bus.publish_state_external("a1", "lovenseEvent", event, source="test")

        node = bus.get_node("a1")
        self.assertIsInstance(node, LovenseProgramAdapterRuntimeNode)
        assert isinstance(node, LovenseProgramAdapterRuntimeNode)

        program = await node.get_state_value("program")
        amp = await node.get_state_value("amplitude")
        seq = await node.get_state_value("sequence")

        self.assertIsInstance(program, dict)
        assert isinstance(program, dict)
        self.assertEqual(int(program.get("tsMs")), 1700000000000)
        self.assertAlmostEqual(float(program.get("timeSec")), 20.0, places=6)
        self.assertAlmostEqual(float(program.get("loopRunningSec")), 9.0, places=6)
        self.assertAlmostEqual(float(program.get("loopPauseSec")), 4.0, places=6)
        # thrusting=10/20 => hz = 1 + 0.5*(5-1) = 3
        self.assertAlmostEqual(float(program.get("hz")), 3.0, places=6)
        # depth=5/20 => 0.25
        self.assertAlmostEqual(float(amp), 0.25, places=6)
        self.assertIsNone(seq)

    async def test_converts_pattern_event_to_hz_sequence(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="a1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=LovenseProgramAdapterRuntimeNode.SPEC.operatorClass,
            stateFields=list(LovenseProgramAdapterRuntimeNode.SPEC.stateFields or []),
            stateValues={"patternMinHz": 0.0, "patternMaxHz": 4.0, "patternStrengthMax": 20.0},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        event = {
            "tsMs": 1700000000000,
            "command": {"name": "Pattern", "apiVer": 2, "kind": "vibration_pattern"},
            "params": {
                "rule": "V:1;F:v,r,p,t,f,s,d,o;S:200#",
                "strength": "2;3;4",
                "timeSec": 10,
            },
        }
        await bus.publish_state_external("a1", "lovenseEvent", event, source="test")

        node = bus.get_node("a1")
        seq = await node.get_state_value("sequence")
        self.assertIsInstance(seq, dict)
        assert isinstance(seq, dict)
        self.assertEqual(int(seq.get("tsMs")), 1700000000000)
        self.assertAlmostEqual(float(seq.get("stepMs")), 200.0, places=6)
        values = seq.get("values")
        self.assertIsInstance(values, list)
        assert isinstance(values, list)
        # Map 0..20 -> 0..4Hz: 2=>0.4, 3=>0.6, 4=>0.8
        self.assertAlmostEqual(float(values[0]), 0.4, places=6)
        self.assertAlmostEqual(float(values[1]), 0.6, places=6)
        self.assertAlmostEqual(float(values[2]), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
