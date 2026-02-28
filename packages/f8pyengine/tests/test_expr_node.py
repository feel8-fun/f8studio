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
from f8pyengine.operators import expr as expr_mod  # noqa: E402
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

    async def test_list_comprehension_over_list_input(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="e3",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(ExprRuntimeNode.SPEC.stateFields or []),
            stateValues={"code": "[x * 2 for x in input if x % 2 == 0]"},
            dataInPorts=[
                F8DataPortSpec(name="input", description="", valueSchema=any_schema(), required=False),
            ],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="", valueSchema=any_schema(), required=False),
            ],
        )
        graph = F8RuntimeGraph(graphId="g3", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("e3")
        assert isinstance(node, ExprRuntimeNode)

        buffer_input(bus, "e3", "input", [1, 2, 3, 4], ts_ms=0, edge=None, ctx_id=None)
        out = await node.compute_output("out", ctx_id=3)
        self.assertEqual(out, [4, 8])

    async def test_list_comprehension_over_nested_objects(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="e4",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(ExprRuntimeNode.SPEC.stateFields or []),
            stateValues={"code": "[p.x for p in input.points if p.x >= 0]"},
            dataInPorts=[
                F8DataPortSpec(name="input", description="", valueSchema=any_schema(), required=False),
            ],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="", valueSchema=any_schema(), required=False),
            ],
        )
        graph = F8RuntimeGraph(graphId="g4", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("e4")
        assert isinstance(node, ExprRuntimeNode)

        payload = {"points": [{"x": -1}, {"x": 0}, {"x": 2}]}
        buffer_input(bus, "e4", "input", payload, ts_ms=0, edge=None, ctx_id=None)
        out = await node.compute_output("out", ctx_id=4)
        self.assertEqual(out, [0, 2])

    async def test_numpy_disabled_by_default(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="e5",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(ExprRuntimeNode.SPEC.stateFields or []),
            stateValues={"code": "np.clip(input, 0, 1)"},
            dataInPorts=[
                F8DataPortSpec(name="input", description="", valueSchema=any_schema(), required=False),
            ],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="", valueSchema=any_schema(), required=False),
            ],
        )
        graph = F8RuntimeGraph(graphId="g5", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("e5")
        assert isinstance(node, ExprRuntimeNode)
        buffer_input(bus, "e5", "input", 0.5, ts_ms=0, edge=None, ctx_id=None)
        out = await node.compute_output("out", ctx_id=5)
        self.assertIsNone(out)

    @unittest.skipIf(expr_mod.np is None, "numpy not available in test environment")
    async def test_numpy_enabled_allows_np_calls(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="e6",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(ExprRuntimeNode.SPEC.stateFields or []),
            stateValues={"allowNumpy": True, "code": "np.clip(input, 0, 1)"},
            dataInPorts=[
                F8DataPortSpec(name="input", description="", valueSchema=any_schema(), required=False),
            ],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="", valueSchema=any_schema(), required=False),
            ],
        )
        graph = F8RuntimeGraph(graphId="g6", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("e6")
        assert isinstance(node, ExprRuntimeNode)
        buffer_input(bus, "e6", "input", 1.5, ts_ms=0, edge=None, ctx_id=None)
        out = await node.compute_output("out", ctx_id=6)
        self.assertEqual(float(out), 1.0)

    async def test_dict_result_without_unpack_emits_single_out_value(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="e7",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(ExprRuntimeNode.SPEC.stateFields or []),
            stateValues={"code": "{'a': input + 1, 'b': input + 2}", "unpackDictOutputs": False},
            dataInPorts=[
                F8DataPortSpec(name="input", description="", valueSchema=any_schema(), required=False),
            ],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="", valueSchema=any_schema(), required=False),
                F8DataPortSpec(name="a", description="", valueSchema=any_schema(), required=False),
            ],
        )
        graph = F8RuntimeGraph(graphId="g7", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("e7")
        assert isinstance(node, ExprRuntimeNode)
        buffer_input(bus, "e7", "input", 10, ts_ms=0, edge=None, ctx_id=None)

        out = await node.compute_output("out", ctx_id=7)
        out_a = await node.compute_output("a", ctx_id=7)
        self.assertEqual(out, {"a": 11, "b": 12})
        self.assertIsNone(out_a)

    async def test_dict_result_with_unpack_maps_to_matching_output_ports(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="e8",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(ExprRuntimeNode.SPEC.stateFields or []),
            stateValues={"code": "{'a': input + 1, 'b': input + 2, 'z': 999}", "unpackDictOutputs": True},
            dataInPorts=[
                F8DataPortSpec(name="input", description="", valueSchema=any_schema(), required=False),
            ],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="", valueSchema=any_schema(), required=False),
                F8DataPortSpec(name="a", description="", valueSchema=any_schema(), required=False),
                F8DataPortSpec(name="b", description="", valueSchema=any_schema(), required=False),
            ],
        )
        graph = F8RuntimeGraph(graphId="g8", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("e8")
        assert isinstance(node, ExprRuntimeNode)
        buffer_input(bus, "e8", "input", 20, ts_ms=0, edge=None, ctx_id=None)

        out_a = await node.compute_output("a", ctx_id=8)
        out_b = await node.compute_output("b", ctx_id=8)
        out_default = await node.compute_output("out", ctx_id=8)
        self.assertEqual(out_a, 21)
        self.assertEqual(out_b, 22)
        self.assertIsNone(out_default)


if __name__ == "__main__":
    unittest.main()
