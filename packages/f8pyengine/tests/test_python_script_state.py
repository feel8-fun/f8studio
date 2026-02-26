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

from f8pysdk import F8StateAccess, F8StateSpec, any_schema  # noqa: E402
from f8pysdk.generated import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.python_script import PythonScriptRuntimeNode, register_operator  # noqa: E402


def _runtime_python_script_node(
    *,
    node_id: str,
    code: str,
    state_fields: list[F8StateSpec] | None = None,
) -> F8RuntimeNode:
    spec = PythonScriptRuntimeNode.SPEC
    return F8RuntimeNode(
        nodeId=node_id,
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=spec.operatorClass,
        execInPorts=list(spec.execInPorts or []),
        execOutPorts=list(spec.execOutPorts or []),
        dataInPorts=list(spec.dataInPorts or []),
        dataOutPorts=list(spec.dataOutPorts or []),
        stateFields=list(state_fields if state_fields is not None else (spec.stateFields or [])),
        stateValues={"code": code},
    )


class PythonScriptStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_on_state_runs_and_can_write_state(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onState(ctx, field, value, tsMs=None):\n"
            "    if field == 'foo':\n"
            "        ctx['set_state']('bar', int(value) * 2)\n"
        )

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        state_fields.append(F8StateSpec(name="foo", label="foo", description="", valueSchema=any_schema(), access=F8StateAccess.rw))
        state_fields.append(F8StateSpec(name="bar", label="bar", description="", valueSchema=any_schema(), access=F8StateAccess.ro))

        op = _runtime_python_script_node(node_id="ps1", code=code, state_fields=state_fields)
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        await bus.publish_state_external("ps1", "foo", 21, source="test")
        await asyncio.sleep(0.05)
        node = bus.get_node("ps1")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        bar = await node.get_state_value("bar")
        self.assertEqual(int(bar), 42)

    async def test_python_script_exec_enabled_by_default(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        state_fields.append(
            F8StateSpec(name="booted", label="booted", description="", valueSchema=any_schema(), access=F8StateAccess.ro)
        )
        op = _runtime_python_script_node(
            node_id="ps2",
            code="def onStart(ctx):\n    ctx['set_state']('booted', True)\n",
            state_fields=state_fields,
        )
        graph = F8RuntimeGraph(graphId="g2", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        await asyncio.sleep(0.05)

        node = bus.get_node("ps2")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        booted = await node.get_state_value("booted")
        self.assertTrue(bool(booted))

    async def test_compute_output_uses_on_exec_outputs_in_pull_mode(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onExec(ctx, execIn, inputs):\n"
            "    return {'outputs': {'out': 123}}\n"
        )
        op = _runtime_python_script_node(node_id="ps3", code=code)
        graph = F8RuntimeGraph(graphId="g3", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps3")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)

        out = await node.compute_output("out", ctx_id="ctx-1")
        self.assertEqual(out, 123)

    async def test_compute_output_falls_back_to_on_msg_when_on_exec_missing(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onMsg(ctx, inputs):\n"
            "    return 99\n"
        )
        op = _runtime_python_script_node(node_id="ps4", code=code)
        graph = F8RuntimeGraph(graphId="g4", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps4")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)

        out = await node.compute_output("out", ctx_id="ctx-2")
        self.assertEqual(out, 99)

    async def test_ctx_locals_persists_between_calls(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onMsg(ctx, inputs):\n"
            "    c = int(ctx['locals'].get('count') or 0)\n"
            "    c += 1\n"
            "    ctx['locals']['count'] = c\n"
            "    return {'outputs': {'out': c}}\n"
        )
        op = _runtime_python_script_node(node_id="ps5", code=code)
        graph = F8RuntimeGraph(graphId="g5", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps5")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)

        out1 = await node.compute_output("out", ctx_id="ctx-5a")
        out2 = await node.compute_output("out", ctx_id="ctx-5b")
        self.assertEqual(out1, 1)
        self.assertEqual(out2, 2)

    async def test_ctx_locals_is_isolated_from_system_state(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        state_fields.append(F8StateSpec(name="x", label="x", description="", valueSchema=any_schema(), access=F8StateAccess.rw))
        code = (
            "async def onExec(ctx, execIn, inputs):\n"
            "    ctx['locals']['x'] = 1\n"
            "    await ctx['set_state_async']('x', 2)\n"
            "    v = await ctx['get_state']('x')\n"
            "    return {'outputs': {'out': v}}\n"
        )
        op = _runtime_python_script_node(node_id="ps6", code=code, state_fields=state_fields)
        graph = F8RuntimeGraph(graphId="g6", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps6")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)

        out = await node.compute_output("out", ctx_id="ctx-6a")
        self.assertEqual(out, 2)
        state_x = await node.get_state_value("x")
        self.assertEqual(state_x, 2)
        self.assertEqual(node._locals.get("x"), 1)

if __name__ == "__main__":
    unittest.main()
