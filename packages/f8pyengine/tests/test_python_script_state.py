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

        op = F8RuntimeNode(
            nodeId="ps1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=PythonScriptRuntimeNode.SPEC.operatorClass,
            stateFields=state_fields,
            stateValues={"allowUnsafeExec": True, "code": code},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        await bus.publish_state_external("ps1", "foo", 21, source="test")
        await asyncio.sleep(0.05)
        node = bus.get_node("ps1")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        bar = await node.get_state_value("bar")
        self.assertEqual(int(bar), 42)

    async def test_unsafe_exec_disabled_by_default(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        op = F8RuntimeNode(
            nodeId="ps2",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=PythonScriptRuntimeNode.SPEC.operatorClass,
            stateFields=state_fields,
            stateValues={"code": "def onStart(ctx):\n    ctx['set_state']('lastError', 'should_not_run')\n"},
        )
        graph = F8RuntimeGraph(graphId="g2", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        await asyncio.sleep(0.05)

        node = bus.get_node("ps2")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        self.assertIn("unsafe python exec is disabled", str(node._last_error or ""))


if __name__ == "__main__":
    unittest.main()
