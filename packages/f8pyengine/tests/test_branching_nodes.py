import asyncio
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.capabilities import EntrypointNode  # noqa: E402
from f8pysdk.executors.exec_flow import EntrypointContext  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.nodes import OperatorNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.specs import (  # noqa: E402
    F8DataPortSpec,
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    any_schema,
)
from f8pysdk.testing import ServiceBusHarness, buffer_input  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.branching import (  # noqa: E402
    DataMuxRuntimeNode,
    ExecBranchRuntimeNode,
    ExecMergeRuntimeNode,
    register_operator,
)
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402
from f8pyengine.pyengine_service import PyEngineService  # noqa: E402


@dataclass
class _RuntimeStub:
    bus: object


class _ManualEntrypointNode(OperatorNode, EntrypointNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=node_id,
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
        )
        self.ctx: EntrypointContext | None = None

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        del exec_id, in_port
        return []

    async def start_entrypoint(self, ctx: EntrypointContext) -> None:
        self.ctx = ctx

    async def stop_entrypoint(self) -> None:
        self.ctx = None


class _RecordingStepNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        state = dict(initial_state or {})
        super().__init__(
            node_id=node_id,
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
            data_out_ports=[str(port.name) for port in list(node.dataOutPorts or [])],
            state_fields=[str(field.name) for field in list(node.stateFields or [])],
        )
        self.calls: list[str] = []
        self.value = state.get("value", str(node_id).upper())

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        del exec_id
        self.calls.append(str(in_port or ""))
        if "done" in self.exec_out_ports:
            return ["done"]
        if "exec" in self.exec_out_ports:
            return ["exec"]
        return []

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        del ctx_id
        if str(port or "") == "out":
            return self.value
        return None


class _FinalSinkNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=node_id,
            data_in_ports=[str(port.name) for port in list(node.dataInPorts or [])],
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
        )
        self.calls: list[Any] = []

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        del in_port
        value = await self.pull("input", ctx_id=exec_id)
        self.calls.append(value)
        return []


def _runtime_node(
    *,
    node_id: str,
    operator_class: str,
    exec_in: list[str] | None = None,
    exec_out: list[str] | None = None,
    data_in: list[str] | None = None,
    data_out: list[str] | None = None,
    state_fields: object | None = None,
    state_values: dict[str, Any] | None = None,
) -> F8RuntimeNode:
    return F8RuntimeNode(
        nodeId=node_id,
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=operator_class,
        execInPorts=list(exec_in or []),
        execOutPorts=list(exec_out or []),
        dataInPorts=[F8DataPortSpec(name=name, valueSchema=any_schema(), definitionProtected=False) for name in list(data_in or [])],
        dataOutPorts=[
            F8DataPortSpec(name=name, valueSchema=any_schema(), definitionProtected=False) for name in list(data_out or [])
        ],
        stateFields=list(state_fields or []),
        stateValues=dict(state_values or {}),
    )


def _exec_edge(*, edge_id: str, from_node: str, from_port: str, to_node: str, to_port: str) -> F8Edge:
    return F8Edge(
        edgeId=edge_id,
        fromServiceId="svcA",
        fromOperatorId=from_node,
        fromPort=from_port,
        toServiceId="svcA",
        toOperatorId=to_node,
        toPort=to_port,
        kind=F8EdgeKindEnum.exec,
        strategy=F8EdgeStrategyEnum.latest,
    )


def _data_edge(*, edge_id: str, from_node: str, from_port: str, to_node: str, to_port: str) -> F8Edge:
    return F8Edge(
        edgeId=edge_id,
        fromServiceId="svcA",
        fromOperatorId=from_node,
        fromPort=from_port,
        toServiceId="svcA",
        toOperatorId=to_node,
        toPort=to_port,
        kind=F8EdgeKindEnum.data,
        strategy=F8EdgeStrategyEnum.latest,
    )


class _PullRecordingDataMux(DataMuxRuntimeNode):
    def __init__(self) -> None:
        node = _runtime_node(
            node_id="mux",
            operator_class=str(DataMuxRuntimeNode.SPEC.operatorClass),
            exec_in=["exec"],
            exec_out=["exec"],
            data_in=["branch_a", "branch_b", "default"],
            data_out=["out"],
            state_fields=DataMuxRuntimeNode.SPEC.stateFields,
            state_values={"selectedInput": "branch_b"},
        )
        super().__init__(node_id="mux", node=node, initial_state={"selectedInput": "branch_b"})
        self.pull_calls: list[tuple[str, str | int | None]] = []
        self.values: dict[str, Any] = {"branch_a": "A", "branch_b": "B", "default": "D"}

    async def pull(self, port: str, *, ctx_id: str | int | None = None) -> Any:
        self.pull_calls.append((str(port), ctx_id))
        return self.values.get(str(port))


class BranchingNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_exec_branch_emits_selected_branch(self) -> None:
        node = _runtime_node(
            node_id="branch",
            operator_class=str(ExecBranchRuntimeNode.SPEC.operatorClass),
            exec_in=["exec"],
            exec_out=["branch_a", "branch_b", "default"],
            state_fields=ExecBranchRuntimeNode.SPEC.stateFields,
            state_values={"selectedBranch": "branch_b"},
        )
        branch = ExecBranchRuntimeNode(node_id="branch", node=node, initial_state={"selectedBranch": "branch_b"})

        self.assertEqual(await branch.on_exec(1, "exec"), ["branch_b"])

    async def test_exec_branch_invalid_selection_uses_default(self) -> None:
        node = _runtime_node(
            node_id="branch",
            operator_class=str(ExecBranchRuntimeNode.SPEC.operatorClass),
            exec_in=["exec"],
            exec_out=["branch_a", "default"],
            state_fields=ExecBranchRuntimeNode.SPEC.stateFields,
            state_values={"selectedBranch": "missing"},
        )
        branch = ExecBranchRuntimeNode(node_id="branch", node=node, initial_state={"selectedBranch": "missing"})

        self.assertEqual(await branch.on_exec(1, "exec"), ["default"])

    async def test_exec_branch_invalid_selection_without_default_emits_nothing(self) -> None:
        node = _runtime_node(
            node_id="branch",
            operator_class=str(ExecBranchRuntimeNode.SPEC.operatorClass),
            exec_in=["exec"],
            exec_out=["branch_a"],
            state_fields=ExecBranchRuntimeNode.SPEC.stateFields,
            state_values={"selectedBranch": "missing"},
        )
        branch = ExecBranchRuntimeNode(node_id="branch", node=node, initial_state={"selectedBranch": "missing"})

        self.assertEqual(await branch.on_exec(1, "exec"), [])

    async def test_exec_branch_publishes_resolved_branch(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = _runtime_node(
            node_id="branch",
            operator_class=str(ExecBranchRuntimeNode.SPEC.operatorClass),
            exec_in=["exec"],
            exec_out=["branch_a", "branch_b", "default"],
            state_fields=ExecBranchRuntimeNode.SPEC.stateFields,
            state_values={"selectedBranch": "branch_b"},
        )
        graph = F8RuntimeGraph(graphId="g_branch", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        branch = bus.get_node("branch")
        self.assertIsInstance(branch, ExecBranchRuntimeNode)
        assert isinstance(branch, ExecBranchRuntimeNode)
        self.assertEqual(await branch.on_exec(1, "exec"), ["branch_b"])
        self.assertEqual(await branch.get_state_value("resolvedBranch"), "branch_b")

    async def test_exec_merge_any_input_returns_single_exec(self) -> None:
        node = _runtime_node(
            node_id="merge",
            operator_class=str(ExecMergeRuntimeNode.SPEC.operatorClass),
            exec_in=["in_a", "in_b"],
            exec_out=["exec"],
        )
        merge = ExecMergeRuntimeNode(node_id="merge", node=node, initial_state={})

        self.assertEqual(await merge.on_exec(1, "in_a"), ["exec"])
        self.assertEqual(await merge.on_exec(2, "in_b"), ["exec"])

    async def test_data_mux_pulls_only_selected_input_when_value_exists(self) -> None:
        mux = _PullRecordingDataMux()

        self.assertEqual(await mux.compute_output("out", ctx_id=10), "B")
        self.assertEqual(mux.pull_calls, [("branch_b", 10)])

    async def test_data_mux_invalid_selection_uses_default(self) -> None:
        mux = _PullRecordingDataMux()
        await mux.on_state("selectedInput", "missing")

        self.assertEqual(await mux.compute_output("out", ctx_id=11), "D")
        self.assertEqual(mux.pull_calls, [("default", 11)])

    async def test_data_mux_invalid_selection_without_default_returns_none(self) -> None:
        node = _runtime_node(
            node_id="mux",
            operator_class=str(DataMuxRuntimeNode.SPEC.operatorClass),
            exec_in=["exec"],
            exec_out=["exec"],
            data_in=["branch_a"],
            data_out=["out"],
            state_fields=DataMuxRuntimeNode.SPEC.stateFields,
            state_values={"selectedInput": "missing"},
        )
        mux = DataMuxRuntimeNode(node_id="mux", node=node, initial_state={"selectedInput": "missing"})

        self.assertIsNone(await mux.compute_output("out", ctx_id=12))

    async def test_data_mux_caches_outputs_per_context(self) -> None:
        mux = _PullRecordingDataMux()

        self.assertEqual(await mux.compute_output("out", ctx_id=13), "B")
        mux.values["branch_b"] = "B2"
        self.assertEqual(await mux.compute_output("out", ctx_id=13), "B")
        self.assertEqual(await mux.compute_output("out", ctx_id=14), "B2")
        self.assertEqual(mux.pull_calls, [("branch_b", 13), ("branch_b", 14)])

    async def test_data_mux_publishes_resolved_input(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = _runtime_node(
            node_id="mux",
            operator_class=str(DataMuxRuntimeNode.SPEC.operatorClass),
            exec_in=["exec"],
            exec_out=["exec"],
            data_in=["branch_a", "branch_b", "default"],
            data_out=["out"],
            state_fields=DataMuxRuntimeNode.SPEC.stateFields,
            state_values={"selectedInput": "branch_b"},
        )
        graph = F8RuntimeGraph(graphId="g_mux", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        mux = bus.get_node("mux")
        self.assertIsInstance(mux, DataMuxRuntimeNode)
        assert isinstance(mux, DataMuxRuntimeNode)
        buffer_input(bus, "mux", "branch_b", "B", ts_ms=now_ms(), edge=None, ctx_id=1)
        self.assertEqual(await mux.compute_output("out", ctx_id=1), "B")
        self.assertEqual(await mux.get_state_value("resolvedInput"), "branch_b")

    async def test_branch_merge_mux_integration_runs_only_selected_branch(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        reg.register_operator_factory(
            SERVICE_CLASS,
            "f8.test_manual_entrypoint",
            lambda node_id, node, initial_state: _ManualEntrypointNode(
                node_id=node_id,
                node=node,
                initial_state=initial_state,
            ),
            overwrite=True,
        )
        reg.register_operator_factory(
            SERVICE_CLASS,
            "f8.test_recording_step",
            lambda node_id, node, initial_state: _RecordingStepNode(
                node_id=node_id,
                node=node,
                initial_state=initial_state,
            ),
            overwrite=True,
        )
        reg.register_operator_factory(
            SERVICE_CLASS,
            "f8.test_final_sink",
            lambda node_id, node, initial_state: _FinalSinkNode(
                node_id=node_id,
                node=node,
                initial_state=initial_state,
            ),
            overwrite=True,
        )
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        service = PyEngineService()
        runtime = _RuntimeStub(bus=bus)
        await service.setup(runtime)  # type: ignore[arg-type]
        try:
            nodes = [
                _runtime_node(
                    node_id="start",
                    operator_class="f8.test_manual_entrypoint",
                    exec_in=[],
                    exec_out=["exec"],
                ),
                _runtime_node(
                    node_id="branch",
                    operator_class=str(ExecBranchRuntimeNode.SPEC.operatorClass),
                    exec_in=["exec"],
                    exec_out=["branch_a", "branch_b", "branch_c", "default"],
                    state_fields=ExecBranchRuntimeNode.SPEC.stateFields,
                    state_values={"selectedBranch": "branch_b"},
                ),
                _runtime_node(
                    node_id="a",
                    operator_class="f8.test_recording_step",
                    exec_in=["exec"],
                    exec_out=["done"],
                    data_out=["out"],
                ),
                _runtime_node(
                    node_id="b",
                    operator_class="f8.test_recording_step",
                    exec_in=["exec"],
                    exec_out=["done"],
                    data_out=["out"],
                ),
                _runtime_node(
                    node_id="c",
                    operator_class="f8.test_recording_step",
                    exec_in=["exec"],
                    exec_out=["done"],
                    data_out=["out"],
                ),
                _runtime_node(
                    node_id="merge",
                    operator_class=str(ExecMergeRuntimeNode.SPEC.operatorClass),
                    exec_in=["branch_a", "branch_b", "branch_c", "default"],
                    exec_out=["exec"],
                ),
                _runtime_node(
                    node_id="mux",
                    operator_class=str(DataMuxRuntimeNode.SPEC.operatorClass),
                    exec_in=[],
                    exec_out=[],
                    data_in=["branch_a", "branch_b", "branch_c", "default"],
                    data_out=["out"],
                    state_fields=DataMuxRuntimeNode.SPEC.stateFields,
                    state_values={"selectedInput": "branch_b"},
                ),
                _runtime_node(
                    node_id="sink",
                    operator_class="f8.test_final_sink",
                    exec_in=["exec"],
                    exec_out=[],
                    data_in=["input"],
                ),
            ]
            edges = [
                _exec_edge(edge_id="e_start_branch", from_node="start", from_port="exec", to_node="branch", to_port="exec"),
                _exec_edge(edge_id="e_branch_a", from_node="branch", from_port="branch_a", to_node="a", to_port="exec"),
                _exec_edge(edge_id="e_branch_b", from_node="branch", from_port="branch_b", to_node="b", to_port="exec"),
                _exec_edge(edge_id="e_branch_c", from_node="branch", from_port="branch_c", to_node="c", to_port="exec"),
                _exec_edge(edge_id="e_a_merge", from_node="a", from_port="done", to_node="merge", to_port="branch_a"),
                _exec_edge(edge_id="e_b_merge", from_node="b", from_port="done", to_node="merge", to_port="branch_b"),
                _exec_edge(edge_id="e_c_merge", from_node="c", from_port="done", to_node="merge", to_port="branch_c"),
                _exec_edge(edge_id="e_merge_sink", from_node="merge", from_port="exec", to_node="sink", to_port="exec"),
                _data_edge(edge_id="d_a_mux", from_node="a", from_port="out", to_node="mux", to_port="branch_a"),
                _data_edge(edge_id="d_b_mux", from_node="b", from_port="out", to_node="mux", to_port="branch_b"),
                _data_edge(edge_id="d_c_mux", from_node="c", from_port="out", to_node="mux", to_port="branch_c"),
                _data_edge(edge_id="d_mux_sink", from_node="mux", from_port="out", to_node="sink", to_port="input"),
            ]
            graph = F8RuntimeGraph(graphId="g_integration", revision="r1", nodes=nodes, edges=edges)
            await bus.set_rungraph(graph)

            start = bus.get_node("start")
            self.assertIsInstance(start, _ManualEntrypointNode)
            assert isinstance(start, _ManualEntrypointNode)
            self.assertIsNotNone(start.ctx)
            assert start.ctx is not None
            await start.ctx.emit_exec("exec", exec_id=101)
            for _ in range(5):
                await asyncio.sleep(0)

            a = bus.get_node("a")
            b = bus.get_node("b")
            c = bus.get_node("c")
            sink = bus.get_node("sink")
            self.assertIsInstance(a, _RecordingStepNode)
            self.assertIsInstance(b, _RecordingStepNode)
            self.assertIsInstance(c, _RecordingStepNode)
            self.assertIsInstance(sink, _FinalSinkNode)
            assert isinstance(a, _RecordingStepNode)
            assert isinstance(b, _RecordingStepNode)
            assert isinstance(c, _RecordingStepNode)
            assert isinstance(sink, _FinalSinkNode)
            self.assertEqual(a.calls, [])
            self.assertEqual(b.calls, ["exec"])
            self.assertEqual(c.calls, [])
            self.assertEqual(sink.calls, ["B"])
        finally:
            await service.teardown(runtime)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
