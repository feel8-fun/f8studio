from f8pysdk.specs import F8DataPayloadSpec, F8DataPortPayloadKind
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

from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.specs import F8DataPortSpec, F8RuntimeGraph, F8RuntimeNode, number_schema  # noqa: E402
from f8pysdk.testing import buffer_input, ServiceBusHarness  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.switch_mixer import SwitchMixerRuntimeNode, register_operator  # noqa: E402


class SwitchMixerTests(unittest.IsolatedAsyncioTestCase):
    async def test_switches_between_named_channels(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="mix1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=SwitchMixerRuntimeNode.SPEC.operatorClass,
            stateFields=list(SwitchMixerRuntimeNode.SPEC.stateFields or []),
            dataInPorts=[
                F8DataPortSpec(name="main", payload=F8DataPayloadSpec(kind=F8DataPortPayloadKind.json, valueSchema=number_schema())),
                F8DataPortSpec(name="fallback", payload=F8DataPayloadSpec(kind=F8DataPortPayloadKind.json, valueSchema=number_schema())),
                F8DataPortSpec(name="manual", payload=F8DataPayloadSpec(kind=F8DataPortPayloadKind.json, valueSchema=number_schema())),
            ],
            stateValues={"currentChannel": "main", "fadeMs": 0},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("mix1")
        self.assertIsInstance(node, SwitchMixerRuntimeNode)
        assert isinstance(node, SwitchMixerRuntimeNode)

        buffer_input(bus, "mix1", "main", 1.0, ts_ms=now_ms(), edge=None, ctx_id=None)
        buffer_input(bus, "mix1", "fallback", 0.0, ts_ms=now_ms(), edge=None, ctx_id=None)
        buffer_input(bus, "mix1", "manual", 0.75, ts_ms=now_ms(), edge=None, ctx_id=None)

        out_main = await node.compute_output("out", ctx_id=0)
        self.assertAlmostEqual(float(out_main), 1.0, places=6)
        self.assertEqual(str(await node.get_state_value("resolvedChannel")), "main")

        await bus.publish_state_external("mix1", "currentChannel", "manual", source="test")
        out_manual = await node.compute_output("out", ctx_id=1)
        self.assertAlmostEqual(float(out_manual), 0.75, places=6)
        self.assertEqual(str(await node.get_state_value("resolvedChannel")), "manual")

    async def test_crossfades_and_holds_last_value_for_missing_updates(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="mix1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=SwitchMixerRuntimeNode.SPEC.operatorClass,
            stateFields=list(SwitchMixerRuntimeNode.SPEC.stateFields or []),
            dataInPorts=[
                F8DataPortSpec(name="track_a", payload=F8DataPayloadSpec(kind=F8DataPortPayloadKind.json, valueSchema=number_schema())),
                F8DataPortSpec(name="track_b", payload=F8DataPayloadSpec(kind=F8DataPortPayloadKind.json, valueSchema=number_schema())),
            ],
            stateValues={"currentChannel": "track_a", "fadeMs": 90},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("mix1")
        self.assertIsInstance(node, SwitchMixerRuntimeNode)
        assert isinstance(node, SwitchMixerRuntimeNode)

        buffer_input(bus, "mix1", "track_a", 1.0, ts_ms=now_ms(), edge=None, ctx_id=None)
        buffer_input(bus, "mix1", "track_b", 0.2, ts_ms=now_ms(), edge=None, ctx_id=None)

        out0 = await node.compute_output("out", ctx_id=0)
        self.assertAlmostEqual(float(out0), 1.0, places=6)

        await bus.publish_state_external("mix1", "currentChannel", "track_b", source="test")
        await asyncio.sleep(0.03)
        mid = await node.compute_output("out", ctx_id=1)
        self.assertGreater(float(mid), 0.2)
        self.assertLess(float(mid), 1.0)

        # Simulate the selected channel no longer receiving a valid sample.
        buffer_input(bus, "mix1", "track_b", None, ts_ms=now_ms(), edge=None, ctx_id=None)
        await asyncio.sleep(0.10)
        end = await node.compute_output("out", ctx_id=2)
        self.assertAlmostEqual(float(end), 0.2, places=1)


if __name__ == "__main__":
    unittest.main()
