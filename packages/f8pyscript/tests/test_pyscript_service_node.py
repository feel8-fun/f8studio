import asyncio
import os
import sys
import unittest
import uuid

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
from f8pysdk.shm.video import VideoShmWriter  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyscript.constants import SERVICE_CLASS  # noqa: E402
from f8pyscript.main import PythonScriptServiceProgram  # noqa: E402
from f8pyscript.node_registry import register_specs  # noqa: E402
from f8pyscript.service_node import PythonScriptServiceNode  # noqa: E402


def _service_node(*, code: str, state_fields: list[F8StateSpec] | None = None, state_values: dict[str, object] | None = None) -> F8RuntimeNode:
    spec = RuntimeNodeRegistry.instance().service_spec(SERVICE_CLASS)
    assert spec is not None
    merged_state = {"code": code}
    if state_values is not None:
        merged_state.update(state_values)
    return F8RuntimeNode(
        nodeId="svcA",
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=None,
        dataInPorts=list(spec.dataInPorts or []),
        dataOutPorts=list(spec.dataOutPorts or []),
        stateFields=list(state_fields if state_fields is not None else (spec.stateFields or [])),
        stateValues=merged_state,
    )


class PyScriptServiceNodeTests(unittest.IsolatedAsyncioTestCase):
    def test_program_defaults_data_delivery_to_both(self) -> None:
        program = PythonScriptServiceProgram()
        cfg = program.build_runtime_config(service_id="svcA", nats_url="mem://")
        self.assertEqual(str(cfg.bus.data_delivery), "both")

    async def _build_runtime(self) -> tuple[object, object, PythonScriptServiceNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_specs(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[_service_node(code="")], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        self.assertIsInstance(node, PythonScriptServiceNode)
        assert isinstance(node, PythonScriptServiceNode)
        return harness, bus, node

    async def test_on_start_and_lifecycle_hooks(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_specs(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onStart(ctx):\n"
            "    ctx['set_state']('startedCount', 1)\n"
            "\n"
            "def onPause(ctx, meta=None):\n"
            "    c = int(ctx['locals'].get('pauseCount') or 0) + 1\n"
            "    ctx['locals']['pauseCount'] = c\n"
            "    ctx['set_state']('pauseCount', c)\n"
            "\n"
            "def onResume(ctx, meta=None):\n"
            "    c = int(ctx['locals'].get('resumeCount') or 0) + 1\n"
            "    ctx['locals']['resumeCount'] = c\n"
            "    ctx['set_state']('resumeCount', c)\n"
        )

        fields = list(RuntimeNodeRegistry.instance().service_spec(SERVICE_CLASS).stateFields or [])  # type: ignore[union-attr]
        fields.append(F8StateSpec(name="startedCount", label="", description="", valueSchema=any_schema(), access=F8StateAccess.rw))
        fields.append(F8StateSpec(name="pauseCount", label="", description="", valueSchema=any_schema(), access=F8StateAccess.rw))
        fields.append(F8StateSpec(name="resumeCount", label="", description="", valueSchema=any_schema(), access=F8StateAccess.rw))

        graph = F8RuntimeGraph(graphId="g2", revision="r1", nodes=[_service_node(code="", state_fields=fields)], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)
        await node.on_state("code", code, ts_ms=1)
        await asyncio.sleep(0.05)

        self.assertEqual(int(await node.get_state_value("startedCount") or 0), 1)

        await bus.set_active(False, source="test")
        await asyncio.sleep(0.05)
        self.assertEqual(int(await node.get_state_value("pauseCount") or 0), 1)

        await bus.set_active(True, source="test")
        await asyncio.sleep(0.05)
        self.assertEqual(int(await node.get_state_value("resumeCount") or 0), 1)

    async def test_tick_pause_resume(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_specs(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onTick(ctx, tick):\n"
            "    c = int(ctx['locals'].get('tickCount') or 0) + 1\n"
            "    ctx['locals']['tickCount'] = c\n"
            "    ctx['set_state']('tickCount', c)\n"
        )

        fields = list(RuntimeNodeRegistry.instance().service_spec(SERVICE_CLASS).stateFields or [])  # type: ignore[union-attr]
        fields.append(F8StateSpec(name="tickCount", label="", description="", valueSchema=any_schema(), access=F8StateAccess.rw))

        graph = F8RuntimeGraph(
            graphId="g3",
            revision="r1",
            nodes=[_service_node(code="", state_fields=fields, state_values={"tickEnabled": False, "tickMs": 100})],
            edges=[],
        )
        await bus.set_rungraph(graph)

        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)
        await node.on_state("code", code, ts_ms=1)
        await node.on_state("tickEnabled", True, ts_ms=2)
        await node.on_state("tickMs", 20, ts_ms=3)

        await asyncio.sleep(0.12)
        before_pause = int(await node.get_state_value("tickCount") or 0)
        self.assertGreaterEqual(before_pause, 2)

        await bus.set_active(False, source="test")
        await asyncio.sleep(0.08)
        during_pause = int(await node.get_state_value("tickCount") or 0)
        self.assertLessEqual(during_pause, before_pause + 1)
        await asyncio.sleep(0.08)
        stable_pause = int(await node.get_state_value("tickCount") or 0)
        self.assertEqual(stable_pause, during_pause)

        await bus.set_active(True, source="test")
        await asyncio.sleep(0.1)
        after_resume = int(await node.get_state_value("tickCount") or 0)
        self.assertGreater(after_resume, during_pause)

    async def test_command_grant_and_revoke_local_exec(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_specs(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "async def onCommand(ctx, name, args, meta=None):\n"
            "    if name == 'run_echo':\n"
            "        import sys\n"
            "        return await ctx['exec_local'](sys.executable, ['-c', \"print('hello')\"])\n"
            "    return {'name': name}\n"
        )

        graph = F8RuntimeGraph(graphId="g4", revision="r1", nodes=[_service_node(code="")], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)
        await node.on_state("code", code, ts_ms=1)

        with self.assertRaises(PermissionError):
            await node.on_command("run_echo", {})

        grant_reply = await node.on_command("grant_local_exec", {"ttlMs": 2000}, meta={"reqId": "r1"})
        self.assertTrue(bool((grant_reply or {}).get("ok")))

        run_reply = await node.on_command("run_echo", {})
        run_result = (run_reply or {}).get("result") if isinstance(run_reply, dict) else {}
        self.assertIsInstance(run_result, dict)
        self.assertEqual(int((run_result or {}).get("returncode", -1)), 0)
        self.assertIn("hello", str((run_result or {}).get("stdout") or ""))

        revoke_reply = await node.on_command("revoke_local_exec", {})
        self.assertTrue(bool((revoke_reply or {}).get("ok")))

        with self.assertRaises(PermissionError):
            await node.on_command("run_echo", {})

    async def test_video_shm_subscription(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_specs(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        shm_name = f"test.shm.pyscript.{uuid.uuid4().hex}"
        writer = VideoShmWriter(shm_name=shm_name, size=1024 * 1024, slot_count=2)
        writer.open()
        try:
            code = (
                f"def onStart(ctx):\n"
                f"    ctx['subscribe_video_shm']('v', '{shm_name}', decode='none')\n"
                "\n"
                "def onCommand(ctx, name, args, meta=None):\n"
                "    if name != 'video':\n"
                "        return {'ok': False}\n"
                "    pkt = ctx['get_video_shm']('v')\n"
                "    if pkt is None:\n"
                "        return {'frameId': 0}\n"
                "    header = pkt.get('header') or {}\n"
                "    return {'frameId': int(header.get('frameId') or 0), 'rawLen': len(pkt.get('raw') or b'')}\n"
            )

            graph = F8RuntimeGraph(graphId="g5", revision="r1", nodes=[_service_node(code="")], edges=[])
            await bus.set_rungraph(graph)
            node = bus.get_node("svcA")
            assert isinstance(node, PythonScriptServiceNode)
            await node.on_state("code", code, ts_ms=1)

            payload = bytes((i % 251 for i in range(16)))
            writer.write_frame_bgra(width=2, height=2, pitch=8, payload=payload)

            await asyncio.sleep(0.1)
            out = await node.on_command("video", {})
            out_result = (out or {}).get("result") if isinstance(out, dict) else {}
            self.assertIsInstance(out_result, dict)
            self.assertGreater(int((out_result or {}).get("frameId") or 0), 0)
            self.assertEqual(int((out_result or {}).get("rawLen") or 0), len(payload))
        finally:
            writer.close(unlink=True)

    async def test_get_state_cached_sync_snapshot(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_specs(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        fields = list(RuntimeNodeRegistry.instance().service_spec(SERVICE_CLASS).stateFields or [])  # type: ignore[union-attr]
        fields.append(F8StateSpec(name="myState", label="", description="", valueSchema=any_schema(), access=F8StateAccess.rw))

        code = (
            "def onCommand(ctx, name, args, meta=None):\n"
            "    if name != 'cached':\n"
            "        return {'ok': False}\n"
            "    return {'value': ctx['get_state_cached']('myState', 99)}\n"
        )

        graph = F8RuntimeGraph(graphId="g6", revision="r1", nodes=[_service_node(code="", state_fields=fields)], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)
        await node.on_state("code", code, ts_ms=1)

        out1 = await node.on_command("cached", {})
        out1_result = (out1 or {}).get("result") if isinstance(out1, dict) else {}
        self.assertIsInstance(out1_result, dict)
        self.assertEqual(int((out1_result or {}).get("value") or 0), 99)

        await bus.publish_state_external("svcA", "myState", 123, source="test")
        await asyncio.sleep(0.05)
        out2 = await node.on_command("cached", {})
        out2_result = (out2 or {}).get("result") if isinstance(out2, dict) else {}
        self.assertIsInstance(out2_result, dict)
        self.assertEqual(int((out2_result or {}).get("value") or 0), 123)

    async def test_commands_state_normalization_tolerates_empty_values(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_specs(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        graph = F8RuntimeGraph(
            graphId="g7",
            revision="r1",
            nodes=[_service_node(code="", state_values={"commands": None})],
            edges=[],
        )
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)

        await node.on_state("commands", None, ts_ms=1)
        self.assertEqual(node._declared_commands, [])

        await node.on_state("commands", "", ts_ms=2)
        self.assertEqual(node._declared_commands, [])

        await node.on_state("commands", {}, ts_ms=3)
        self.assertEqual(node._declared_commands, [])

        await node.on_state("commands", {"name": "ping"}, ts_ms=4)
        self.assertEqual(len(node._declared_commands), 1)
        self.assertEqual(str(node._declared_commands[0].get("name") or ""), "ping")


if __name__ == "__main__":
    unittest.main()
