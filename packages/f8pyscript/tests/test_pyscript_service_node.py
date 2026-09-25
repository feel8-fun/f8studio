import asyncio
import os
import sys
import unittest
from types import MethodType
from unittest.mock import patch

import msgspec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.specs import F8StateAccess, F8StateSpec, any_schema  # noqa: E402
from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.codec import dump_json  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402
from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32, LatestVideoFrame, ZenohLatestVideoFrameTransport  # noqa: E402

from f8pyscript.constants import SERVICE_CLASS  # noqa: E402
from f8pyscript.main_script import build_app  # noqa: E402
from f8pyscript.script_node_registry import create_pyscript_registry  # noqa: E402
from f8pyscript.script_service_node import PythonScriptServiceNode  # noqa: E402


def _monitor_error_message(bus: object) -> str:
    snapshot = bus.monitor_collector._build_snapshot(ts_ms=int(now_ms()))
    return str(snapshot.error.currentMessage or "")


def _service_node(*, code: str, state_fields: list[F8StateSpec] | None = None, state_values: dict[str, object] | None = None) -> F8RuntimeNode:
    desc = create_pyscript_registry().describe(SERVICE_CLASS)
    spec = desc.service
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


class _FakeLatestVideoTransport:
    def __init__(self, *, payload: bytes) -> None:
        self._payload = bytes(payload)
        self._delivered = False
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def publish_frame(
        self,
        *,
        width: int,
        height: int,
        pitch: int,
        payload: bytes | bytearray | memoryview,
        fmt: int,
        ts_ms: int | None = None,
    ) -> None:
        del width, height, pitch, payload, fmt, ts_ms
        raise RuntimeError("fake transport is subscriber-only")

    def poll_latest(self) -> LatestVideoFrame | None:
        if self._delivered:
            return None
        self._delivered = True
        return LatestVideoFrame(
            width=2,
            height=2,
            pitch=8,
            fmt=VIDEO_FORMAT_BGRA32,
            frame_id=57,
            ts_ms=4321,
            payload=memoryview(self._payload),
        )

    def wait_latest(self, timeout_ms: int) -> LatestVideoFrame | None:
        del timeout_ms
        return self.poll_latest()


class PyScriptServiceNodeTests(unittest.IsolatedAsyncioTestCase):
    def test_service_spec_contains_editor_assist_protocol(self) -> None:
        reg = create_pyscript_registry()
        spec = reg.describe(SERVICE_CLASS).service
        state_fields_by_name = {str(field.name or ""): field for field in list(spec.stateFields or [])}
        self.assertIn("tickEnabled", state_fields_by_name)
        self.assertIn("tickMs", state_fields_by_name)
        self.assertTrue(bool(state_fields_by_name["tickEnabled"].valueRequired))
        self.assertTrue(bool(state_fields_by_name["tickMs"].valueRequired))
        self.assertIn("code", state_fields_by_name)
        self.assertTrue(bool(state_fields_by_name["code"].valueRequired))
        data_in_ports = {str(port.name or ""): port for port in list(spec.dataInPorts or [])}
        data_out_ports = {str(port.name or ""): port for port in list(spec.dataOutPorts or [])}
        self.assertIn("in", data_in_ports)
        self.assertFalse(bool(data_in_ports["in"].definitionProtected))
        self.assertIn("out", data_out_ports)
        self.assertFalse(bool(data_out_ports["out"].definitionProtected))
        self.assertIn("monitor", data_out_ports)
        self.assertTrue(bool(data_out_ports["monitor"].definitionProtected))
        code_field = next((f for f in list(spec.stateFields or []) if str(f.name or "").strip() == "code"), None)
        self.assertIsNotNone(code_field)
        assert code_field is not None
        editor_assist = code_field.editorAssist
        self.assertIsNotNone(editor_assist)
        python_payload = None
        if editor_assist is not None and not isinstance(editor_assist.python, msgspec.UnsetType):
            python_payload = dump_json(editor_assist.python, mode="json")
        self.assertIsInstance(python_payload, dict)
        support_files = (python_payload or {}).get("support_files") if isinstance(python_payload, dict) else None
        self.assertIsInstance(support_files, dict)
        api_stub = str((support_files or {}).get("f8_script_api.pyi") or "")
        self.assertIn("from f8_dynamic_states import F8States as F8States", api_stub)
        self.assertIn("rw/ro/wo fields", api_stub)
        dynamic_bindings = (python_payload or {}).get("dynamic_bindings") if isinstance(python_payload, dict) else None
        self.assertIsInstance(dynamic_bindings, dict)
        inputs_binding = (dynamic_bindings or {}).get("inputs") if isinstance(dynamic_bindings, dict) else None
        self.assertIsInstance(inputs_binding, dict)
        self.assertTrue(bool((inputs_binding or {}).get("enabled")))
        states_binding = (dynamic_bindings or {}).get("states") if isinstance(dynamic_bindings, dict) else None
        self.assertIsInstance(states_binding, dict)
        self.assertTrue(bool((states_binding or {}).get("enabled")))

    def test_program_defaults_data_delivery_to_callback(self) -> None:
        app = build_app()
        cfg = app.build_runtime_config(service_id="svcA")
        self.assertEqual(str(cfg.bus.data_delivery), "callback")

    async def _build_runtime(self) -> tuple[object, object, PythonScriptServiceNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
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
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onStart(ctx):\n"
            "    ctx.set_state('startedCount', 1)\n"
            "\n"
            "def onPause(ctx, meta=None):\n"
            "    c = int(ctx.locals.get('pauseCount') or 0) + 1\n"
            "    ctx.locals['pauseCount'] = c\n"
            "    ctx.set_state('pauseCount', c)\n"
            "\n"
            "def onResume(ctx, meta=None):\n"
            "    c = int(ctx.locals.get('resumeCount') or 0) + 1\n"
            "    ctx.locals['resumeCount'] = c\n"
            "    ctx.set_state('resumeCount', c)\n"
        )

        fields = list(create_pyscript_registry().describe(SERVICE_CLASS).service.stateFields or [])
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
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onTick(ctx, tick):\n"
            "    c = int(ctx.locals.get('tickCount') or 0) + 1\n"
            "    ctx.locals['tickCount'] = c\n"
            "    ctx.set_state('tickCount', c)\n"
        )

        fields = list(create_pyscript_registry().describe(SERVICE_CLASS).service.stateFields or [])
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
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "async def onCommand(ctx, name, args, meta=None):\n"
            "    if name == 'run_echo':\n"
            "        import sys\n"
            "        return await ctx.exec_local(sys.executable, ['-c', \"print('hello')\"])\n"
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

    async def test_video_latest_subscription_uses_zenoh_transport(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        payload = bytes((i % 251 for i in range(16)))
        opened: list[tuple[str, dict[str, object]]] = []

        def _fake_open_subscriber(key_expr: str, **kwargs: object) -> _FakeLatestVideoTransport:
            opened.append((str(key_expr), dict(kwargs)))
            return _FakeLatestVideoTransport(payload=payload)

        code = (
            "def onStart(ctx):\n"
            "    ctx.subscribe_video_latest('v', stream_key='f8/test/pyscript/video', decode='none')\n"
            "\n"
            "def onCommand(ctx, name, args, meta=None):\n"
            "    if name != 'video':\n"
            "        return {'ok': False}\n"
            "    pkt = ctx.get_video_latest('v')\n"
            "    items = ctx.list_video_latest_subscriptions()\n"
            "    if pkt is None:\n"
            "        return {'ok': False, 'items': items}\n"
            "    header = pkt.get('header') or {}\n"
            "    meta = pkt.get('meta') or {}\n"
            "    return {\n"
            "        'ok': True,\n"
            "        'frameId': int(header.get('frameId') or 0),\n"
            "        'rawLen': len(pkt.get('raw') or b''),\n"
            "        'transport': str(meta.get('transport') or ''),\n"
            "        'streamKey': str(meta.get('streamKey') or ''),\n"
            "        'itemTransport': str((items[0] if items else {}).get('transport') or ''),\n"
            "        'itemStreamKey': str((items[0] if items else {}).get('streamKey') or ''),\n"
            "    }\n"
        )

        with patch.object(ZenohLatestVideoFrameTransport, "open_subscriber", side_effect=_fake_open_subscriber):
            graph = F8RuntimeGraph(graphId="g-video-latest", revision="r1", nodes=[_service_node(code="")], edges=[])
            await bus.set_rungraph(graph)
            node = bus.get_node("svcA")
            assert isinstance(node, PythonScriptServiceNode)
            await node.on_state("code", code, ts_ms=1)

            await asyncio.sleep(0.1)
            out = await node.on_command("video", {})
            out_result = (out or {}).get("result") if isinstance(out, dict) else {}
            self.assertIsInstance(out_result, dict)
            self.assertTrue(bool((out_result or {}).get("ok")))
            self.assertEqual(int((out_result or {}).get("frameId") or 0), 57)
            self.assertEqual(int((out_result or {}).get("rawLen") or 0), len(payload))
            self.assertEqual(str((out_result or {}).get("transport") or ""), "zenoh")
            self.assertEqual(str((out_result or {}).get("streamKey") or ""), "f8/test/pyscript/video")
            self.assertEqual(str((out_result or {}).get("itemTransport") or ""), "zenoh")
            self.assertEqual(str((out_result or {}).get("itemStreamKey") or ""), "f8/test/pyscript/video")
            self.assertEqual(opened[0][0], "f8/test/pyscript/video")
            await node.close()

    async def test_get_state_cached_sync_snapshot(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        fields = list(create_pyscript_registry().service_spec(SERVICE_CLASS).stateFields or [])  # type: ignore[union-attr]
        fields.append(F8StateSpec(name="myState", label="", description="", valueSchema=any_schema(), access=F8StateAccess.rw))

        code = (
            "def onCommand(ctx, name, args, meta=None):\n"
            "    if name != 'cached':\n"
            "        return {'ok': False}\n"
            "    v = ctx.states.get('myState')\n"
            "    if v is None:\n"
            "        v = 99\n"
            "    return {'value': v}\n"
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

    async def test_states_view_supports_object_and_mapping_access_and_exposes_wo(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        fields = list(create_pyscript_registry().service_spec(SERVICE_CLASS).stateFields or [])  # type: ignore[union-attr]
        fields.append(F8StateSpec(name="my_rw", label="", description="", valueSchema=any_schema(), access=F8StateAccess.rw))
        fields.append(F8StateSpec(name="my_ro", label="", description="", valueSchema=any_schema(), access=F8StateAccess.ro))
        fields.append(F8StateSpec(name="my_wo", label="", description="", valueSchema=any_schema(), access=F8StateAccess.wo))

        graph = F8RuntimeGraph(
            graphId="g9",
            revision="r1",
            nodes=[_service_node(code="", state_fields=fields)],
            edges=[],
        )
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)

        code = (
            "def onStart(ctx):\n"
            "    ctx.set_state('my_ro', 42)\n"
            "\n"
            "def onCommand(ctx, name, args, meta=None):\n"
            "    if name != 'state_view':\n"
            "        return {'ok': False}\n"
            "    dot_v = ctx.states.my_rw\n"
            "    map_v = ctx.states['my_ro']\n"
            "    get_v = ctx.states.get('my_ro')\n"
            "    wo_v = ctx.states.get('my_wo')\n"
            "    has_wo = 'my_wo' in ctx.states\n"
            "    return {'values': [dot_v, map_v, get_v, wo_v, has_wo]}\n"
        )
        await node.on_state("code", code, ts_ms=1)
        await asyncio.sleep(0.05)
        await bus.publish_state_external("svcA", "my_rw", 41, source="test")
        await asyncio.sleep(0.05)

        out = await node.on_command("state_view", {})
        out_result = (out or {}).get("result") if isinstance(out, dict) else {}
        self.assertIsInstance(out_result, dict)
        self.assertEqual((out_result or {}).get("values"), [41, 42, 42, None, True])

    async def test_states_view_fallback_declared_state_names_exposes_wo(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        fields = list(create_pyscript_registry().service_spec(SERVICE_CLASS).stateFields or [])  # type: ignore[union-attr]
        fields.append(F8StateSpec(name="my_rw", label="", description="", valueSchema=any_schema(), access=F8StateAccess.rw))
        fields.append(F8StateSpec(name="my_ro", label="", description="", valueSchema=any_schema(), access=F8StateAccess.ro))
        fields.append(F8StateSpec(name="my_wo", label="", description="", valueSchema=any_schema(), access=F8StateAccess.wo))

        graph = F8RuntimeGraph(
            graphId="g9f",
            revision="r1",
            nodes=[_service_node(code="", state_fields=fields)],
            edges=[],
        )
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)

        view = node._state_access.build_states_view(())
        self.assertTrue("my_wo" in view)
        self.assertIsNone(view.get("my_wo"))

    async def test_ctx_dict_access_reports_monitor_error(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        graph = F8RuntimeGraph(graphId="g8", revision="r1", nodes=[_service_node(code="")], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)

        code = (
            "def onStart(ctx):\n"
            "    ctx['log']('dict syntax')\n"
        )
        await node.on_state("code", code, ts_ms=1)
        await asyncio.sleep(0.05)
        self.assertIn("not subscriptable", _monitor_error_message(bus))

    async def test_on_data_passes_raw_value(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        graph = F8RuntimeGraph(graphId="g11", revision="r1", nodes=[_service_node(code="")], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)

        code = (
            "def onData(ctx, port, value, ts_ms=None):\n"
            "    if port == 'in':\n"
            "        ctx.locals['v'] = value\n"
            "\n"
            "def onCommand(ctx, name, args, meta=None):\n"
            "    if name != 'get':\n"
            "        return {'ok': False}\n"
            "    return {'v': ctx.locals.get('v')}\n"
        )
        await node.on_state("code", code, ts_ms=1)
        await asyncio.sleep(0.05)

        await node.on_data("in", 123, ts_ms=2)
        out = await node.on_command("get", {})
        out_result = (out or {}).get("result") if isinstance(out, dict) else {}
        self.assertIsInstance(out_result, dict)
        self.assertEqual((out_result or {}).get("v"), 123)

    async def test_on_data_ignored_while_inactive(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        graph = F8RuntimeGraph(graphId="g11_inactive", revision="r1", nodes=[_service_node(code="")], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)

        code = (
            "def onData(ctx, port, value, ts_ms=None):\n"
            "    ctx.locals['v'] = value\n"
            "\n"
            "def onCommand(ctx, name, args, meta=None):\n"
            "    if name != 'get':\n"
            "        return {'ok': False}\n"
            "    return {'v': ctx.locals.get('v')}\n"
        )
        await node.on_state("code", code, ts_ms=1)
        await asyncio.sleep(0.05)
        await node.on_lifecycle(False, {})

        await node.on_data("in", 123, ts_ms=2)
        out = await node.on_command("get", {})
        out_result = (out or {}).get("result") if isinstance(out, dict) else {}

        self.assertIsInstance(out_result, dict)
        self.assertIsNone((out_result or {}).get("v"))

    async def test_outputs_unwrap_state_object_view_to_dict(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        fields = list(create_pyscript_registry().service_spec(SERVICE_CLASS).stateFields or [])  # type: ignore[union-attr]
        fields.append(F8StateSpec(name="pose", label="", description="", valueSchema=any_schema(), access=F8StateAccess.rw))

        graph = F8RuntimeGraph(
            graphId="g12",
            revision="r1",
            nodes=[
                _service_node(
                    code="",
                    state_fields=fields,
                    state_values={
                        "pose": {
                            "bones": [
                                {"name": "Head", "position": [0, 1, 2]},
                                {"name": "Hips", "position": [3, 4, 5]},
                            ]
                        }
                    },
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)

        code = (
            "def onData(ctx, port, value, ts_ms=None):\n"
            "    for b in ctx.states.pose.bones:\n"
            "        if b.name == 'Hips':\n"
            "            return {'outputs': {'out': b}}\n"
            "    return {'outputs': {}}\n"
        )
        await node.on_state("code", code, ts_ms=1)
        await asyncio.sleep(0.05)

        await bus.publish_state_external(
            "svcA",
            "pose",
            {
                "bones": [
                    {"name": "Head", "position": [0, 1, 2]},
                    {"name": "Hips", "position": [3, 4, 5]},
                ]
            },
            source="test",
        )
        await asyncio.sleep(0.05)

        captured: dict[str, object] = {}

        async def _capture_emit(self: PythonScriptServiceNode, port: str, value: object, *, ts_ms: int | None = None) -> None:
            del self, ts_ms
            captured[str(port)] = value

        node.emit = MethodType(_capture_emit, node)
        await node.on_data("in", {"trigger": True}, ts_ms=2)
        out_value = captured.get("out")
        self.assertIsInstance(out_value, dict)
        assert isinstance(out_value, dict)
        self.assertEqual(out_value.get("name"), "Hips")
        self.assertEqual(out_value.get("position"), [3, 4, 5])

    async def test_emit_outputs_reports_emit_failure(self) -> None:
        class FailingEmitPythonScriptServiceNode(PythonScriptServiceNode):
            async def emit(
                self,
                port: str,
                value: object,
                *,
                ts_ms: int | None = None,
                ctx_id: str | int | None = None,
            ) -> None:
                del port, value, ts_ms, ctx_id
                raise RuntimeError("emit failed")

        node = FailingEmitPythonScriptServiceNode(
            node_id="svcA",
            node=_service_node(code=""),
            initial_state={"code": "def onStart(ctx):\n    pass\n"},
        )
        await node._emit_outputs({"outputs": {"out": 123}})

        self.assertIsNotNone(node._error_reporter.last_error)
        self.assertIn("result:emit:out", str(node._error_reporter.last_error))
        self.assertIn("emit failed", str(node._error_reporter.last_error))

    async def test_hook_async_flags_and_invoke_context_reuse(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_pyscript_registry()
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        graph = F8RuntimeGraph(graphId="g10", revision="r1", nodes=[_service_node(code="")], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("svcA")
        assert isinstance(node, PythonScriptServiceNode)

        code = (
            "def onData(ctx, port, value, ts_ms=None):\n"
            "    return {'outputs': {'out': value}}\n"
            "\n"
            "async def onCommand(ctx, name, args, meta=None):\n"
            "    return {'name': name}\n"
        )
        await node.on_state("code", code, ts_ms=1)
        await asyncio.sleep(0.05)

        self.assertFalse(node._hook_runtime.is_data_hook_async())
        self.assertTrue(node._hook_runtime.is_command_hook_async())

        invoke_ctx_a = node._build_invoke_ctx()
        invoke_ctx_b = node._build_invoke_ctx()
        self.assertIs(invoke_ctx_a, invoke_ctx_b)


if __name__ == "__main__":
    unittest.main()
