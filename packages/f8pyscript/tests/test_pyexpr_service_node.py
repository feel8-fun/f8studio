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

from f8pysdk import F8DataPortSpec, F8Edge, F8EdgeKindEnum, F8EdgeStrategyEnum, any_schema  # noqa: E402
from f8pysdk.generated import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyscript.constants import EXPR_SERVICE_CLASS  # noqa: E402
from f8pyscript.expr_node_registry import register_expr_specs  # noqa: E402
from f8pyscript.expr_service_node import PythonExprServiceNode, np  # noqa: E402
from f8pyscript.main_expr import PythonExprServiceProgram  # noqa: E402


def _expr_node(
    *,
    service_id: str,
    state_values: dict[str, object] | None = None,
    data_in_ports: list[F8DataPortSpec] | None = None,
    data_out_ports: list[F8DataPortSpec] | None = None,
) -> F8RuntimeNode:
    spec = RuntimeNodeRegistry.instance().service_spec(EXPR_SERVICE_CLASS)
    assert spec is not None
    state = {"code": "inputs['in']"}
    if state_values is not None:
        state.update(state_values)
    return F8RuntimeNode(
        nodeId=service_id,
        serviceId=service_id,
        serviceClass=EXPR_SERVICE_CLASS,
        operatorClass=None,
        dataInPorts=list(data_in_ports if data_in_ports is not None else (spec.dataInPorts or [])),
        dataOutPorts=list(data_out_ports if data_out_ports is not None else (spec.dataOutPorts or [])),
        stateFields=list(spec.stateFields or []),
        stateValues=state,
    )


def _any_port(name: str) -> F8DataPortSpec:
    return F8DataPortSpec(name=name, description="", valueSchema=any_schema(), required=False)


class PyExprServiceNodeTests(unittest.IsolatedAsyncioTestCase):
    def _register_runtime(self, bus: object) -> None:
        reg = RuntimeNodeRegistry.instance()
        register_expr_specs(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=EXPR_SERVICE_CLASS), registry=reg)

    def test_program_defaults_data_delivery_to_both(self) -> None:
        program = PythonExprServiceProgram()
        cfg = program.build_runtime_config(service_id="svcExpr", nats_url="mem://")
        self.assertEqual(str(cfg.bus.data_delivery), "both")

    async def test_core_expression_evaluation_on_data(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcExpr")
        self._register_runtime(bus)

        graph = F8RuntimeGraph(
            graphId="g_core",
            revision="r1",
            nodes=[_expr_node(service_id="svcExpr", state_values={"code": "inputs['in'] * 2"})],
            edges=[
                F8Edge(
                    edgeId="e_out",
                    fromServiceId="svcExpr",
                    fromOperatorId="svcExpr",
                    fromPort="out",
                    toServiceId="svcExpr",
                    toOperatorId="tap",
                    toPort="in",
                    kind=F8EdgeKindEnum.data,
                    strategy=F8EdgeStrategyEnum.latest,
                    timeoutMs=None,
                )
            ],
        )
        await bus.set_rungraph(graph)
        node = bus.get_node("svcExpr")
        self.assertIsInstance(node, PythonExprServiceNode)
        assert isinstance(node, PythonExprServiceNode)
        await node.on_state("code", "inputs['in'] * 2", ts_ms=1)

        await node.on_data("in", 7, ts_ms=2)
        out = await bus.pull_data("tap", "in", ctx_id="ctx-core")
        self.assertEqual(out, 14)

    async def test_dict_unpack_off_emits_single_default_output(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcExpr")
        self._register_runtime(bus)

        graph = F8RuntimeGraph(
            graphId="g_unpack_off",
            revision="r1",
            nodes=[
                _expr_node(
                    service_id="svcExpr",
                    state_values={"code": "{'a': inputs['in'] + 1, 'b': inputs['in'] + 2}", "unpackDictOutputs": False},
                )
            ],
            edges=[
                F8Edge(
                    edgeId="e_out_only",
                    fromServiceId="svcExpr",
                    fromOperatorId="svcExpr",
                    fromPort="out",
                    toServiceId="svcExpr",
                    toOperatorId="tap",
                    toPort="in",
                    kind=F8EdgeKindEnum.data,
                    strategy=F8EdgeStrategyEnum.latest,
                    timeoutMs=None,
                )
            ],
        )
        await bus.set_rungraph(graph)
        node = bus.get_node("svcExpr")
        assert isinstance(node, PythonExprServiceNode)
        await node.on_state("code", "{'a': inputs['in'] + 1, 'b': inputs['in'] + 2}", ts_ms=1)
        await node.on_state("unpackDictOutputs", False, ts_ms=2)

        await node.on_data("in", 3, ts_ms=3)
        out = await bus.pull_data("tap", "in", ctx_id="ctx-unpack-off")
        self.assertEqual(out, {"a": 4, "b": 5})

    async def test_dict_unpack_on_routes_matching_ports_only(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcExpr")
        self._register_runtime(bus)

        graph = F8RuntimeGraph(
            graphId="g_unpack_on",
            revision="r1",
            nodes=[
                _expr_node(
                    service_id="svcExpr",
                    state_values={
                        "code": "{'a': inputs['in'] + 1, 'b': inputs['in'] + 2, 'z': 999}",
                        "unpackDictOutputs": True,
                    },
                    data_out_ports=[_any_port("out"), _any_port("a"), _any_port("b")],
                )
            ],
            edges=[
                F8Edge(
                    edgeId="e_a",
                    fromServiceId="svcExpr",
                    fromOperatorId="svcExpr",
                    fromPort="a",
                    toServiceId="svcExpr",
                    toOperatorId="tapA",
                    toPort="in",
                    kind=F8EdgeKindEnum.data,
                    strategy=F8EdgeStrategyEnum.latest,
                    timeoutMs=None,
                ),
                F8Edge(
                    edgeId="e_b",
                    fromServiceId="svcExpr",
                    fromOperatorId="svcExpr",
                    fromPort="b",
                    toServiceId="svcExpr",
                    toOperatorId="tapB",
                    toPort="in",
                    kind=F8EdgeKindEnum.data,
                    strategy=F8EdgeStrategyEnum.latest,
                    timeoutMs=None,
                ),
                F8Edge(
                    edgeId="e_out_fallback",
                    fromServiceId="svcExpr",
                    fromOperatorId="svcExpr",
                    fromPort="out",
                    toServiceId="svcExpr",
                    toOperatorId="tapOut",
                    toPort="in",
                    kind=F8EdgeKindEnum.data,
                    strategy=F8EdgeStrategyEnum.latest,
                    timeoutMs=None,
                ),
            ],
        )
        await bus.set_rungraph(graph)
        node = bus.get_node("svcExpr")
        assert isinstance(node, PythonExprServiceNode)
        node.data_out_ports = ["out", "a", "b"]
        await node.on_state("code", "{'a': inputs['in'] + 1, 'b': inputs['in'] + 2, 'z': 999}", ts_ms=1)
        await node.on_state("unpackDictOutputs", True, ts_ms=2)

        await node.on_data("in", 10, ts_ms=3)
        out_a = await bus.pull_data("tapA", "in", ctx_id="ctx-unpack-on-a")
        out_b = await bus.pull_data("tapB", "in", ctx_id="ctx-unpack-on-b")
        out_fallback = await bus.pull_data("tapOut", "in", ctx_id="ctx-unpack-on-out")
        self.assertEqual(out_a, 11)
        self.assertEqual(out_b, 12)
        self.assertIsNone(out_fallback)

    async def test_numpy_disabled_sets_last_error(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcExpr")
        self._register_runtime(bus)

        graph = F8RuntimeGraph(
            graphId="g_np_off",
            revision="r1",
            nodes=[
                _expr_node(service_id="svcExpr", state_values={"code": "np.clip(inputs['in'], 0, 1)", "allowNumpy": False})
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)
        node = bus.get_node("svcExpr")
        assert isinstance(node, PythonExprServiceNode)
        await node.on_state("code", "np.clip(inputs['in'], 0, 1)", ts_ms=1)
        await node.on_state("allowNumpy", False, ts_ms=2)

        await node.on_data("in", 1.5, ts_ms=4)
        last_error = await node.get_state_value("lastError")
        self.assertTrue(str(last_error or "").strip() != "")

    @unittest.skipIf(np is None, "numpy not available in test environment")
    async def test_numpy_enabled_evaluates_expression(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcExpr")
        self._register_runtime(bus)

        graph = F8RuntimeGraph(
            graphId="g_np_on",
            revision="r1",
            nodes=[
                _expr_node(service_id="svcExpr", state_values={"code": "np.clip(inputs['in'], 0, 1)", "allowNumpy": True})
            ],
            edges=[
                F8Edge(
                    edgeId="e_np_out",
                    fromServiceId="svcExpr",
                    fromOperatorId="svcExpr",
                    fromPort="out",
                    toServiceId="svcExpr",
                    toOperatorId="tap",
                    toPort="in",
                    kind=F8EdgeKindEnum.data,
                    strategy=F8EdgeStrategyEnum.latest,
                    timeoutMs=None,
                )
            ],
        )
        await bus.set_rungraph(graph)
        node = bus.get_node("svcExpr")
        assert isinstance(node, PythonExprServiceNode)
        await node.on_state("code", "np.clip(inputs['in'], 0, 1)", ts_ms=1)
        await node.on_state("allowNumpy", True, ts_ms=2)

        await node.on_data("in", 1.5, ts_ms=5)
        out = await bus.pull_data("tap", "in", ctx_id="ctx-np-on")
        self.assertEqual(float(out), 1.0)

    async def test_cross_service_data_edge_triggers_eval(self) -> None:
        harness = ServiceBusHarness()
        bus_src = harness.create_bus("srcSvc")
        bus_dst = harness.create_bus("dstSvc")
        self._register_runtime(bus_dst)

        graph_dst = F8RuntimeGraph(
            graphId="g_cross",
            revision="r1",
            nodes=[_expr_node(service_id="dstSvc", state_values={"code": "inputs['in'] + 100"})],
            edges=[
                F8Edge(
                    edgeId="e_cross_in",
                    fromServiceId="srcSvc",
                    fromOperatorId="srcNode",
                    fromPort="out",
                    toServiceId="dstSvc",
                    toOperatorId="dstSvc",
                    toPort="in",
                    kind=F8EdgeKindEnum.data,
                    strategy=F8EdgeStrategyEnum.latest,
                    timeoutMs=None,
                ),
                F8Edge(
                    edgeId="e_cross_out",
                    fromServiceId="dstSvc",
                    fromOperatorId="dstSvc",
                    fromPort="out",
                    toServiceId="dstSvc",
                    toOperatorId="tap",
                    toPort="in",
                    kind=F8EdgeKindEnum.data,
                    strategy=F8EdgeStrategyEnum.latest,
                    timeoutMs=None,
                ),
            ],
        )
        await bus_dst.set_rungraph(graph_dst)
        bus_dst.set_data_delivery("both", source="test")
        node = bus_dst.get_node("dstSvc")
        assert isinstance(node, PythonExprServiceNode)
        await node.on_state("code", "inputs['in'] + 100", ts_ms=1)
        await bus_src.emit_data("srcNode", "out", 23, ts_ms=123)
        await asyncio.sleep(0.05)
        out = await bus_dst.pull_data("tap", "in", ctx_id="ctx-cross")
        self.assertEqual(out, 123)


if __name__ == "__main__":
    unittest.main()
