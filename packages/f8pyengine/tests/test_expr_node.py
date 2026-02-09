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
from f8pysdk.service_bus.routing_data import buffer_input  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.expr import ExprRuntimeNode, register_operator  # noqa: E402
from f8pysdk import F8DataPortSpec, any_schema  # noqa: E402


class ExprNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_extracts_nested_fields_via_attribute_access(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="e1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(ExprRuntimeNode.SPEC.stateFields or []),
            stateValues={"code": "input.center.x"},
            dataInPorts=[
                F8DataPortSpec(name="input", description="", valueSchema=any_schema(), required=False),
            ],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="", valueSchema=any_schema(), required=False),
            ],
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("e1")
        self.assertIsInstance(node, ExprRuntimeNode)
        assert isinstance(node, ExprRuntimeNode)

        buffer_input(bus, "e1", "input", {"center": {"x": 3.25, "y": 9}}, ts_ms=0, edge=None, ctx_id=None)
        out = await node.compute_output("out", ctx_id=1)
        self.assertAlmostEqual(float(out), 3.25, places=6)

    async def test_math_expression(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="e2",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(ExprRuntimeNode.SPEC.stateFields or []),
            stateValues={"code": "a + b - c**2"},
            dataInPorts=[
                F8DataPortSpec(name="a", description="", valueSchema=any_schema(), required=False),
                F8DataPortSpec(name="b", description="", valueSchema=any_schema(), required=False),
                F8DataPortSpec(name="c", description="", valueSchema=any_schema(), required=False),
            ],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="", valueSchema=any_schema(), required=False),
            ],
        )
        graph = F8RuntimeGraph(graphId="g2", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("e2")
        assert isinstance(node, ExprRuntimeNode)

        buffer_input(bus, "e2", "a", 2, ts_ms=0, edge=None, ctx_id=None)
        buffer_input(bus, "e2", "b", 3, ts_ms=0, edge=None, ctx_id=None)
        buffer_input(bus, "e2", "c", 4, ts_ms=0, edge=None, ctx_id=None)

        out = await node.compute_output("out", ctx_id=2)
        self.assertEqual(out, -11)


if __name__ == "__main__":
    unittest.main()
