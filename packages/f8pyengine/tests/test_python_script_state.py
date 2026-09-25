from f8pysdk.codec import dump_json
import asyncio
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.specs import (  # noqa: E402
    F8ArrayTypeSchema,
    F8ComplexObjectTypeSchema,
    F8DataPortSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    string_schema,
)
from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.service_bus.runtime import ServiceBus  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402
from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32, LatestVideoFrame, ZenohLatestVideoFrameTransport  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.python_script import (  # noqa: E402
    PythonScriptRuntimeNode,
    register_operator,
)
from f8pyengine.operators.script_utils.input_binding import script_uses_inputs_object_access  # noqa: E402


def _runtime_python_script_node(
    *,
    node_id: str,
    code: str,
    state_fields: list[F8StateSpec] | None = None,
    state_values: dict[str, object] | None = None,
    data_in_ports: list[F8DataPortSpec] | None = None,
    data_out_ports: list[F8DataPortSpec] | None = None,
) -> F8RuntimeNode:
    spec = PythonScriptRuntimeNode.SPEC
    merged_state_values: dict[str, object] = {"code": code}
    if state_values is not None:
        merged_state_values.update(state_values)
    return F8RuntimeNode(
        nodeId=node_id,
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=spec.operatorClass,
        execInPorts=[port.name for port in (spec.execInPorts or [])],
        execOutPorts=[port.name for port in (spec.execOutPorts or [])],
        dataInPorts=list(data_in_ports if data_in_ports is not None else (spec.dataInPorts or [])),
        dataOutPorts=list(data_out_ports if data_out_ports is not None else (spec.dataOutPorts or [])),
        stateFields=list(state_fields if state_fields is not None else (spec.stateFields or [])),
        stateValues=merged_state_values,
    )


def _monitor_current_error_message(bus: ServiceBus) -> str:
    snapshot = bus.monitor_collector._build_snapshot(ts_ms=int(now_ms()))
    return str(snapshot.error.currentMessage or "")


class _FailingEmitPythonScriptRuntimeNode(PythonScriptRuntimeNode):
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


class _FakeLatestVideoTransport:
    def __init__(self, *, payload: bytes) -> None:
        self._payload = bytes(payload)
        self._delivered = False
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def set_min_sample_interval_ms(self, min_sample_interval_ms: int) -> None:
        del min_sample_interval_ms

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


class PythonScriptStateTests(unittest.IsolatedAsyncioTestCase):
    def test_inputs_access_mode_detection(self) -> None:
        mapping_only = (
            "def onMsg(ctx, inputs):\n"
            "    return inputs.get('msg')\n"
        )
        dot_access = (
            "def onMsg(ctx, inputs):\n"
            "    return inputs.msg\n"
        )
        alias_dot_access = (
            "def onMsg(ctx, inputs):\n"
            "    payload = inputs\n"
            "    return payload.msg\n"
        )
        syntax_error = "def onMsg(ctx, inputs)\n    return 1\n"
        self.assertFalse(script_uses_inputs_object_access(mapping_only))
        self.assertTrue(script_uses_inputs_object_access(dot_access))
        self.assertTrue(script_uses_inputs_object_access(alias_dot_access))
        # Conservative fallback when parser cannot decide.
        self.assertTrue(script_uses_inputs_object_access(syntax_error))

    def test_spec_contains_editor_assist_protocol(self) -> None:
        spec = PythonScriptRuntimeNode.SPEC
        code_field = next((f for f in list(spec.stateFields or []) if str(f.name or "").strip() == "code"), None)
        input_mode_field = next((f for f in list(spec.stateFields or []) if str(f.name or "").strip() == "inputMode"), None)
        self.assertIsNotNone(code_field)
        self.assertIsNotNone(input_mode_field)
        assert code_field is not None
        assert input_mode_field is not None
        self.assertEqual(input_mode_field.access, F8StateAccess.rw)
        editor_assist = code_field.editorAssist
        self.assertIsNotNone(editor_assist)
        python_payload = dump_json(editor_assist.python, mode="json") if editor_assist is not None else None
        self.assertIsInstance(python_payload, dict)
        support_files = (python_payload or {}).get("support_files") if isinstance(python_payload, dict) else None
        self.assertIsInstance(support_files, dict)
        api_stub = str((support_files or {}).get("f8_script_api.pyi") or "")
        self.assertIn("from f8_dynamic_inputs import F8Inputs as F8Inputs", api_stub)
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

    async def test_on_state_runs_and_can_write_state(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onState(ctx, field, value, tsMs=None):\n"
            "    if field == 'foo':\n"
            "        ctx.set_state('bar', int(value) * 2)\n"
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
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        state_fields.append(
            F8StateSpec(name="booted", label="booted", description="", valueSchema=any_schema(), access=F8StateAccess.ro)
        )
        op = _runtime_python_script_node(
            node_id="ps2",
            code="def onStart(ctx):\n    ctx.set_state('booted', True)\n",
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
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onExec(ctx, exec_in, inputs):\n"
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

    async def test_on_exec_runs_without_on_start(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        state_fields.append(F8StateSpec(name="fired", label="fired", description="", valueSchema=string_schema(), access=F8StateAccess.rw))
        code = (
            "async def onExec(ctx, exec_in, inputs):\n"
            "    await ctx.set_state_async('fired', exec_in or 'missing')\n"
            "    return {'outputs': {'out': 321}}\n"
        )
        op = _runtime_python_script_node(node_id="ps3_exec", code=code, state_fields=state_fields)
        graph = F8RuntimeGraph(graphId="g3_exec", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps3_exec")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)

        out_ports = await node.on_exec("exec-1", "exec")
        fired = await node.get_state_value("fired")

        self.assertEqual(out_ports, ["exec"])
        self.assertEqual(fired, "exec")

    async def test_compute_output_falls_back_to_on_msg_when_on_exec_missing(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
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
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onMsg(ctx, inputs):\n"
            "    c = int(ctx.locals.get('count') or 0)\n"
            "    c += 1\n"
            "    ctx.locals['count'] = c\n"
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
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        state_fields.append(F8StateSpec(name="x", label="x", description="", valueSchema=any_schema(), access=F8StateAccess.rw))
        code = (
            "async def onExec(ctx, exec_in, inputs):\n"
            "    ctx.locals['x'] = 1\n"
            "    await ctx.set_state_async('x', 2)\n"
            "    v = await ctx.read_state('x')\n"
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

    async def test_get_state_cached_sync_snapshot(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        state_fields.append(F8StateSpec(name="x", label="x", description="", valueSchema=any_schema(), access=F8StateAccess.rw))
        code = (
            "def onExec(ctx, exec_in, inputs):\n"
            "    v = ctx.states.get('x')\n"
            "    if v is None:\n"
            "        v = 7\n"
            "    return {'outputs': {'out': v}}\n"
        )
        op = _runtime_python_script_node(node_id="ps7", code=code, state_fields=state_fields)
        graph = F8RuntimeGraph(graphId="g7", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps7")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)

        out1 = await node.compute_output("out", ctx_id="ctx-7a")
        self.assertEqual(out1, 7)

        await bus.publish_state_external("ps7", "x", 33, source="test")
        await asyncio.sleep(0.05)
        out2 = await node.compute_output("out", ctx_id="ctx-7b")
        self.assertEqual(out2, 33)

    async def test_states_view_supports_object_and_mapping_access_and_exposes_wo(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        state_fields.append(
            F8StateSpec(name="rw_state", label="rw_state", description="", valueSchema=any_schema(), access=F8StateAccess.rw)
        )
        state_fields.append(
            F8StateSpec(name="ro_state", label="ro_state", description="", valueSchema=any_schema(), access=F8StateAccess.ro)
        )
        state_fields.append(
            F8StateSpec(name="wo_state", label="wo_state", description="", valueSchema=any_schema(), access=F8StateAccess.wo)
        )
        code = (
            "def onStart(ctx):\n"
            "    ctx.set_state('ro_state', 8)\n"
            "\n"
            "def onExec(ctx, exec_in, inputs):\n"
            "    dot_v = ctx.states.rw_state\n"
            "    map_v = ctx.states['ro_state']\n"
            "    get_v = ctx.states.get('ro_state')\n"
            "    wo_v = ctx.states.get('wo_state')\n"
            "    has_wo = 'wo_state' in ctx.states\n"
            "    return {'outputs': {'out': [dot_v, map_v, get_v, wo_v, has_wo]}}\n"
        )
        op = _runtime_python_script_node(node_id="ps11", code=code, state_fields=state_fields)
        graph = F8RuntimeGraph(graphId="g11", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        await asyncio.sleep(0.05)
        await bus.publish_state_external("ps11", "rw_state", 7, source="test")
        await asyncio.sleep(0.05)

        node = bus.get_node("ps11")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        out = await node.compute_output("out", ctx_id="ctx-11")
        self.assertEqual(out, [7, 8, 8, None, True])

    async def test_states_view_fallback_declared_state_names_exposes_wo(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        state_fields = list(PythonScriptRuntimeNode.SPEC.stateFields or [])
        state_fields.append(
            F8StateSpec(name="rw_state", label="rw_state", description="", valueSchema=any_schema(), access=F8StateAccess.rw)
        )
        state_fields.append(
            F8StateSpec(name="ro_state", label="ro_state", description="", valueSchema=any_schema(), access=F8StateAccess.ro)
        )
        state_fields.append(
            F8StateSpec(name="wo_state", label="wo_state", description="", valueSchema=any_schema(), access=F8StateAccess.wo)
        )
        op = _runtime_python_script_node(node_id="ps11f", code="def onExec(ctx, exec_in, inputs): return None", state_fields=state_fields)
        graph = F8RuntimeGraph(graphId="g11f", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps11f")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        view = node._build_states_view(())
        self.assertTrue("wo_state" in view)
        self.assertIsNone(view.get("wo_state"))

    async def test_inputs_required_flag_does_not_enforce_runtime_non_null(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        data_in_ports = [
            F8DataPortSpec(name="req", description="", valueSchema=string_schema(), definitionProtected=True),
            F8DataPortSpec(name="opt", description="", valueSchema=string_schema(), definitionProtected=False),
        ]
        code = (
            "def onMsg(ctx, inputs):\n"
            "    return {'outputs': {'out': [inputs.req, inputs.opt]}}\n"
        )
        op = _runtime_python_script_node(node_id="ps9", code=code, data_in_ports=data_in_ports)
        graph = F8RuntimeGraph(graphId="g9", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps9")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        out_all_missing = await node._compute_outputs_for_pull({}, exec_in=None)
        self.assertEqual(out_all_missing.get("out"), [None, None])

        out_required_present = await node._compute_outputs_for_pull({"req": "hello"}, exec_in=None)
        self.assertEqual(out_required_present.get("out"), ["hello", None])

    async def test_required_input_accepts_none_value_for_transient_frames(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        msg_schema = F8ComplexObjectTypeSchema(
            properties={"bones": F8ArrayTypeSchema(items=any_schema())},
            required=["bones"],
        )
        data_in_ports = [F8DataPortSpec(name="msg", description="", valueSchema=msg_schema, definitionProtected=True)]
        code = (
            "def onExec(ctx, exec_in, inputs):\n"
            "    if inputs.msg is None:\n"
            "        return {'outputs': {'out': 'none'}}\n"
            "    return {'outputs': {'out': 'ok'}}\n"
        )
        op = _runtime_python_script_node(node_id="ps9n", code=code, data_in_ports=data_in_ports)
        graph = F8RuntimeGraph(graphId="g9n", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps9n")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        out = await node._compute_outputs_for_pull({"msg": None}, exec_in="exec")
        self.assertEqual(out.get("out"), "none")

    async def test_inputs_nested_object_and_array_support_dot_access(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        bone_schema = F8ComplexObjectTypeSchema(
            properties={"name": string_schema()},
            required=["name"],
        )
        msg_schema = F8ComplexObjectTypeSchema(
            properties={"bones": F8ArrayTypeSchema(items=bone_schema)},
            required=["bones"],
        )
        data_in_ports = [F8DataPortSpec(name="msg", description="", valueSchema=msg_schema, definitionProtected=True)]
        code = (
            "def onMsg(ctx, inputs):\n"
            "    return {'outputs': {'out': inputs.msg.bones[1].name}}\n"
        )
        op = _runtime_python_script_node(node_id="ps10", code=code, data_in_ports=data_in_ports)
        graph = F8RuntimeGraph(graphId="g10", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps10")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        out = await node._compute_outputs_for_pull(
            {"msg": {"bones": [{"name": "Head"}, {"name": "Hips"}]}},
            exec_in=None,
        )
        self.assertEqual(out.get("out"), "Hips")

    async def test_inputs_non_identifier_port_name_maps_to_attribute(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        data_in_ports = [F8DataPortSpec(name="hip-pos", description="", valueSchema=string_schema(), definitionProtected=True)]
        code = (
            "def onMsg(ctx, inputs):\n"
            "    return {'outputs': {'out': inputs.hip_pos}}\n"
        )
        op = _runtime_python_script_node(node_id="ps14", code=code, data_in_ports=data_in_ports)
        graph = F8RuntimeGraph(graphId="g14", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps14")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        out = await node._compute_outputs_for_pull({"hip-pos": "ok"}, exec_in=None)
        self.assertEqual(out.get("out"), "ok")

    async def test_inputs_name_collision_warns_and_keeps_both_fields(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        data_in_ports = [
            F8DataPortSpec(name="a-b", description="", valueSchema=string_schema(), definitionProtected=True),
            F8DataPortSpec(name="a b", description="", valueSchema=string_schema(), definitionProtected=True),
        ]
        code = (
            "def onMsg(ctx, inputs):\n"
            "    return {'outputs': {'out': [inputs.a_b, inputs.a_b_1]}}\n"
        )
        op = _runtime_python_script_node(node_id="ps15", code=code, data_in_ports=data_in_ports)
        graph = F8RuntimeGraph(graphId="g15", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps15")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        out = await node._compute_outputs_for_pull({"a-b": "x", "a b": "y"}, exec_in=None)
        self.assertEqual(out.get("out"), ["x", "y"])
        self.assertIn("inputModel:", str(node._last_error or ""))
        self.assertIn("collision", str(node._last_error or ""))

    async def test_outputs_unwrap_input_object_view_to_dict(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        msg_schema = F8ComplexObjectTypeSchema(
            properties={
                "modelName": string_schema(),
                "bones": F8ArrayTypeSchema(
                    items=F8ComplexObjectTypeSchema(
                        properties={
                            "name": string_schema(),
                            "position": F8ArrayTypeSchema(items=any_schema()),
                        }
                    )
                ),
            },
            required=["bones"],
        )
        data_in_ports = [F8DataPortSpec(name="msg", description="", valueSchema=msg_schema, definitionProtected=True)]
        code = (
            "def onExec(ctx, exec_in, inputs):\n"
            "    msg = inputs.msg\n"
            "    for b in msg.bones:\n"
            "        if b.name == 'Hips':\n"
            "            return {'outputs': {'out': b}}\n"
            "    return {'outputs': {}}\n"
        )
        op = _runtime_python_script_node(node_id="ps13", code=code, data_in_ports=data_in_ports)
        graph = F8RuntimeGraph(graphId="g13", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps13")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        payload = {
            "modelName": "m",
            "bones": [
                {"name": "Head", "position": [0, 1, 2]},
                {"name": "Hips", "position": [3, 4, 5]},
            ],
        }
        out = await node._compute_outputs_for_pull({"msg": payload}, exec_in="exec")
        value = out.get("out")
        self.assertIsInstance(value, dict)
        assert isinstance(value, dict)
        self.assertEqual(value.get("name"), "Hips")
        self.assertEqual(value.get("position"), [3, 4, 5])

    async def test_inputs_decode_error_sets_last_error_with_path(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        msg_schema = F8ComplexObjectTypeSchema(
            properties={"bones": F8ArrayTypeSchema(items=F8ComplexObjectTypeSchema(properties={"name": string_schema()}))},
            required=["bones"],
        )
        data_in_ports = [F8DataPortSpec(name="msg", description="", valueSchema=msg_schema, definitionProtected=True)]
        code = (
            "def onMsg(ctx, inputs):\n"
            "    _ = inputs.msg\n"
            "    return {'outputs': {'out': 1}}\n"
        )
        op = _runtime_python_script_node(
            node_id="ps12",
            code=code,
            data_in_ports=data_in_ports,
            state_values={"inputMode": "msgspec_struct"},
        )
        graph = F8RuntimeGraph(graphId="g12", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps12")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        self.assertEqual(node._input_binding.mode, "msgspec_struct")
        out = await node._compute_outputs_for_pull({"msg": {"bones": [None]}}, exec_in=None)
        self.assertEqual(out, {})
        error_text = str(node._last_error or "")
        self.assertIn("compute:inputs:", error_text)
        self.assertIn("$.msg.bones[0]", error_text)

    async def test_inputs_struct_supports_get_and_index_access(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onMsg(ctx, inputs):\n"
            "    x = inputs.get('msg')\n"
            "    y = inputs['msg']\n"
            "    return {'outputs': {'out': [x, y]}}\n"
        )
        op = _runtime_python_script_node(node_id="ps16", code=code)
        graph = F8RuntimeGraph(graphId="g16", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps16")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        self.assertEqual(node._input_binding.mode, "input_view")
        out = await node._compute_outputs_for_pull({"msg": 123}, exec_in=None)
        self.assertEqual(out.get("out"), [123, 123])

    async def test_any_input_inner_dict_supports_get_and_index_access(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        data_in_ports = [F8DataPortSpec(name="payload", description="", valueSchema=any_schema(), definitionProtected=True)]
        code = (
            "def onMsg(ctx, inputs):\n"
            "    p = inputs.payload\n"
            "    a = p.get('user')\n"
            "    b = p['user']\n"
            "    return {'outputs': {'out': [a, b]}}\n"
        )
        op = _runtime_python_script_node(node_id="ps17", code=code, data_in_ports=data_in_ports)
        graph = F8RuntimeGraph(graphId="g17", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps17")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        self.assertEqual(node._input_binding.mode, "input_view")
        out = await node._compute_outputs_for_pull({"payload": {"user": "alice"}}, exec_in=None)
        self.assertEqual(out.get("out"), ["alice", "alice"])

    async def test_dot_access_uses_default_input_view_mode(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onMsg(ctx, inputs):\n"
            "    return {'outputs': {'out': inputs.msg}}\n"
        )
        op = _runtime_python_script_node(node_id="ps18", code=code)
        graph = F8RuntimeGraph(graphId="g18", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps18")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        self.assertEqual(node._input_binding.mode, "input_view")
        out = await node._compute_outputs_for_pull({"msg": 123}, exec_in=None)
        self.assertEqual(out.get("out"), 123)

    async def test_raw_dict_mode_keeps_mapping_inputs(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onMsg(ctx, inputs):\n"
            "    return {'outputs': {'out': inputs.get('msg')}}\n"
        )
        op = _runtime_python_script_node(node_id="ps19", code=code, state_values={"inputMode": "raw_dict"})
        graph = F8RuntimeGraph(graphId="g19", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps19")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        self.assertEqual(node._input_binding.mode, "raw_dict")
        out = await node._compute_outputs_for_pull({"msg": 456}, exec_in=None)
        self.assertEqual(out.get("out"), 456)

    async def test_msgspec_mode_tracks_decode_metrics(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onMsg(ctx, inputs):\n"
            "    return {'outputs': {'out': inputs.msg}}\n"
        )
        op = _runtime_python_script_node(node_id="ps20", code=code, state_values={"inputMode": "msgspec_struct"})
        graph = F8RuntimeGraph(graphId="g20", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("ps20")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        self.assertEqual(node._input_binding.mode, "msgspec_struct")
        out = await node._compute_outputs_for_pull({"msg": 789}, exec_in=None)
        self.assertEqual(out.get("out"), 789)
        counters = node.get_performance_counters()
        self.assertGreaterEqual(float(counters.get("input_decode_time_us", 0.0)), 0.0)

    async def test_no_hooks_reports_monitor_error(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = "x = 1\n"
        op = _runtime_python_script_node(node_id="ps_no_hooks", code=code)
        graph = F8RuntimeGraph(graphId="g_no_hooks", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        await asyncio.sleep(0.05)

        node = bus.get_node("ps_no_hooks")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        error_text = _monitor_current_error_message(bus)
        self.assertIn("no hooks defined", error_text)

    async def test_monitor_error_clears_after_successful_recompile(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        bad_code = "def onMsg(ctx, inputs)\n    return 1\n"
        op = _runtime_python_script_node(node_id="ps_err_clear", code=bad_code)
        graph = F8RuntimeGraph(graphId="g_err_clear", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        await asyncio.sleep(0.05)

        node = bus.get_node("ps_err_clear")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        self.assertIn("compile:", _monitor_current_error_message(bus))

        good_code = "def onMsg(ctx, inputs):\n    return 1\n"
        await node.on_state("code", good_code, ts_ms=123)
        await asyncio.sleep(0.05)
        self.assertEqual(_monitor_current_error_message(bus), "")

    async def test_apply_result_reports_emit_failure(self) -> None:
        code = "def onMsg(ctx, inputs):\n    return {'outputs': {'out': 1}}\n"
        op = _runtime_python_script_node(node_id="ps_emit_fail", code=code)
        node = _FailingEmitPythonScriptRuntimeNode(
            node_id="ps_emit_fail",
            node=op,
            initial_state={"code": code},
        )

        result = await node._apply_result({"outputs": {"out": 1}})

        self.assertIsNone(result)
        error_text = str(node._last_error or "")
        self.assertIn("result:emit:out", error_text)
        self.assertIn("emit failed", error_text)

    async def test_video_latest_subscription_uses_zenoh_transport(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        payload = bytes((i % 251 for i in range(16)))
        opened: list[tuple[str, dict[str, object]]] = []

        def _fake_open_subscriber(key_expr: str, **kwargs: object) -> _FakeLatestVideoTransport:
            opened.append((str(key_expr), dict(kwargs)))
            return _FakeLatestVideoTransport(payload=payload)

        code = (
            "def onStart(ctx):\n"
            "    ctx.subscribe_video_latest('v', stream_key='f8/test/pyengine/video', decode='none')\n"
            "\n"
            "def onMsg(ctx, inputs):\n"
            "    pkt = ctx.get_video_latest('v')\n"
            "    items = ctx.list_video_latest_subscriptions()\n"
            "    if pkt is None:\n"
            "        return {'outputs': {'out': {'ok': False, 'items': items}}}\n"
            "    header = pkt.get('header') or {}\n"
            "    meta = pkt.get('meta') or {}\n"
            "    return {'outputs': {'out': {\n"
            "        'ok': True,\n"
            "        'frameId': int(header.get('frameId') or 0),\n"
            "        'rawLen': len(pkt.get('raw') or b''),\n"
            "        'transport': str(meta.get('transport') or ''),\n"
            "        'streamKey': str(meta.get('streamKey') or ''),\n"
            "        'itemTransport': str((items[0] if items else {}).get('transport') or ''),\n"
            "        'itemStreamKey': str((items[0] if items else {}).get('streamKey') or ''),\n"
            "    }}}\n"
        )
        op = _runtime_python_script_node(node_id="ps_video_latest", code=code)

        with patch.object(ZenohLatestVideoFrameTransport, "open_subscriber", side_effect=_fake_open_subscriber):
            graph = F8RuntimeGraph(graphId="g_video_latest", revision="r1", nodes=[op], edges=[])
            await bus.set_rungraph(graph)
            await asyncio.sleep(0.1)

            node = bus.get_node("ps_video_latest")
            self.assertIsInstance(node, PythonScriptRuntimeNode)
            assert isinstance(node, PythonScriptRuntimeNode)

            out = await node._compute_outputs_for_pull({"msg": None}, exec_in=None)

        out_value = out.get("out")
        self.assertIsInstance(out_value, dict)
        assert isinstance(out_value, dict)
        self.assertTrue(bool(out_value.get("ok")))
        self.assertEqual(int(out_value.get("frameId") or 0), 57)
        self.assertEqual(int(out_value.get("rawLen") or 0), len(payload))
        self.assertEqual(str(out_value.get("transport") or ""), "zenoh")
        self.assertEqual(str(out_value.get("streamKey") or ""), "f8/test/pyengine/video")
        self.assertEqual(str(out_value.get("itemTransport") or ""), "zenoh")
        self.assertEqual(str(out_value.get("itemStreamKey") or ""), "f8/test/pyengine/video")
        self.assertEqual(opened[0][0], "f8/test/pyengine/video")

    async def test_ctx_dict_access_reports_monitor_error(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        code = (
            "def onStart(ctx):\n"
            "    ctx['log']('dict syntax')\n"
        )
        op = _runtime_python_script_node(node_id="ps8", code=code)
        graph = F8RuntimeGraph(graphId="g8", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        await asyncio.sleep(0.05)

        node = bus.get_node("ps8")
        self.assertIsInstance(node, PythonScriptRuntimeNode)
        assert isinstance(node, PythonScriptRuntimeNode)
        error_text = _monitor_current_error_message(bus)
        self.assertIn("not subscriptable", error_text)

if __name__ == "__main__":
    unittest.main()
