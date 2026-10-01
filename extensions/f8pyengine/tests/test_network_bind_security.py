import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.lovense_mock_server import (  # noqa: E402
    LovenseMockServerRuntimeNode,
    register_operator as register_lovense_mock_server,
)
from f8pyengine.operators.udp_in import UdpInRuntimeNode, register_operator as register_udp_in  # noqa: E402


class NetworkBindSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_lovense_mock_server_rejects_non_loopback_by_default(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_lovense_mock_server(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="lov1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=LovenseMockServerRuntimeNode.SPEC.operatorClass,
            stateFields=list(LovenseMockServerRuntimeNode.SPEC.stateFields or []),
            stateValues={"bindAddress": "127.0.0.1", "port": 30010},
        )
        await bus.set_rungraph(F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[]))
        try:
            with self.assertRaises(ValueError):
                await bus.publish_state_external("lov1", "bindAddress", "0.0.0.0", source="test")

            await bus.publish_state_external("lov1", "allowNonLoopbackBind", True, source="test")
            await bus.publish_state_external("lov1", "bindAddress", "0.0.0.0", source="test")
        finally:
            node = bus.get_node("lov1")
            if isinstance(node, LovenseMockServerRuntimeNode):
                await node.close()

    async def test_udp_in_rejects_non_loopback_by_default(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_udp_in(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="udp1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=UdpInRuntimeNode.SPEC.operatorClass,
            stateFields=list(UdpInRuntimeNode.SPEC.stateFields or []),
            stateValues={"bindAddress": "127.0.0.1", "port": 39541},
        )
        await bus.set_rungraph(F8RuntimeGraph(graphId="g2", revision="r1", nodes=[op], edges=[]))
        try:
            with self.assertRaises(ValueError):
                await bus.publish_state_external("udp1", "bindAddress", "0.0.0.0", source="test")

            await bus.publish_state_external("udp1", "allowNonLoopbackBind", True, source="test")
            await bus.publish_state_external("udp1", "bindAddress", "0.0.0.0", source="test")
        finally:
            node = bus.get_node("udp1")
            if isinstance(node, UdpInRuntimeNode):
                await node.close()


if __name__ == "__main__":
    unittest.main()
