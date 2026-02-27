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

from f8pysdk.generated import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.shm.video import VIDEO_FORMAT_FLOW2_F16, VideoShmWriter  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.python_script import PythonScriptRuntimeNode, register_operator  # noqa: E402


def _runtime_python_script_node(*, node_id: str, code: str) -> F8RuntimeNode:
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
        stateFields=list(spec.stateFields or []),
        stateValues={"code": code},
    )


class PythonScriptVideoShmTests(unittest.IsolatedAsyncioTestCase):
    async def test_subscribe_latest_and_decode_flow2_f16(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        shm_name = f"test.shm.ps.flow.{uuid.uuid4().hex}"
        writer = VideoShmWriter(shm_name=shm_name, size=1024 * 1024, slot_count=2)
        writer.open()
        try:
            code = (
                f"def onStart(ctx):\n"
                f"    ctx['subscribe_video_shm']('flow', '{shm_name}', decode='auto')\n\n"
                f"async def onExec(ctx, execIn, inputs):\n"
                f"    pkt = ctx['get_video_shm']('flow')\n"
                f"    if pkt is None:\n"
                f"        return {{'outputs': {{'out': {{'frameId': 0}}}}}}\n"
                f"    dec = pkt.get('decoded')\n"
                f"    kind = dec.get('kind') if isinstance(dec, dict) else ''\n"
                f"    shape = dec.get('shape') if isinstance(dec, dict) else []\n"
                f"    h = pkt.get('header') if isinstance(pkt, dict) else {{}}\n"
                f"    return {{'outputs': {{'out': {{\n"
                f"        'frameId': int((h or {{}}).get('frameId') or 0),\n"
                f"        'kind': kind,\n"
                f"        'shape': shape,\n"
                f"        'rawLen': len(pkt.get('raw') or b''),\n"
                f"    }}}}}}\n"
            )
            op = _runtime_python_script_node(node_id="psv1", code=code)
            graph = F8RuntimeGraph(graphId="gv1", revision="r1", nodes=[op], edges=[])
            await bus.set_rungraph(graph)

            width = 3
            height = 2
            pitch = width * 4
            payload = bytes((i % 251 for i in range(pitch * height)))
            writer.write_frame(width=width, height=height, pitch=pitch, payload=payload, fmt=VIDEO_FORMAT_FLOW2_F16)

            node = bus.get_node("psv1")
            self.assertIsInstance(node, PythonScriptRuntimeNode)
            assert isinstance(node, PythonScriptRuntimeNode)
            await asyncio.sleep(0.1)
            out1 = await node.compute_output("out", ctx_id="ctx-a")
            self.assertIsInstance(out1, dict)
            assert isinstance(out1, dict)
            frame_id_1 = int(out1.get("frameId") or 0)
            self.assertGreater(frame_id_1, 0)
            self.assertEqual(str(out1.get("kind") or ""), "flow2_f16")
            self.assertEqual(list(out1.get("shape") or []), [2, 3, 2])
            self.assertEqual(int(out1.get("rawLen") or 0), int(len(payload)))

            out2 = await node.compute_output("out", ctx_id="ctx-b")
            self.assertIsInstance(out2, dict)
            assert isinstance(out2, dict)
            frame_id_2 = int(out2.get("frameId") or 0)
            self.assertEqual(frame_id_2, frame_id_1)
            await node.close()
        finally:
            writer.close(unlink=True)

    async def test_decode_none_returns_raw_only(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        shm_name = f"test.shm.ps.raw.{uuid.uuid4().hex}"
        writer = VideoShmWriter(shm_name=shm_name, size=1024 * 1024, slot_count=2)
        writer.open()
        try:
            code = (
                f"def onStart(ctx):\n"
                f"    ctx['subscribe_video_shm']('video', '{shm_name}', decode='none')\n\n"
                f"async def onExec(ctx, execIn, inputs):\n"
                f"    pkt = ctx['get_video_shm']('video')\n"
                f"    if pkt is None:\n"
                f"        return {{'outputs': {{'out': {{'ok': False}}}}}}\n"
                f"    return {{'outputs': {{'out': {{\n"
                f"        'ok': True,\n"
                f"        'decodedIsNone': pkt.get('decoded') is None,\n"
                f"        'rawLen': len(pkt.get('raw') or b''),\n"
                f"    }}}}}}\n"
            )
            op = _runtime_python_script_node(node_id="psv2", code=code)
            graph = F8RuntimeGraph(graphId="gv2", revision="r1", nodes=[op], edges=[])
            await bus.set_rungraph(graph)

            width = 2
            height = 2
            pitch = width * 4
            payload = bytes((10 + i for i in range(pitch * height)))
            writer.write_frame_bgra(width=width, height=height, pitch=pitch, payload=payload)

            node = bus.get_node("psv2")
            self.assertIsInstance(node, PythonScriptRuntimeNode)
            assert isinstance(node, PythonScriptRuntimeNode)
            await asyncio.sleep(0.1)
            out = await node.compute_output("out", ctx_id="ctx-x")
            self.assertIsInstance(out, dict)
            assert isinstance(out, dict)
            self.assertTrue(bool(out.get("ok")))
            self.assertTrue(bool(out.get("decodedIsNone")))
            self.assertEqual(int(out.get("rawLen") or 0), int(len(payload)))
            await node.close()
        finally:
            writer.close(unlink=True)

    async def test_replace_subscription_same_key(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        shm_b = f"test.shm.ps.b.{uuid.uuid4().hex}"
        writer = VideoShmWriter(shm_name=shm_b, size=1024 * 1024, slot_count=2)
        writer.open()
        writer.write_frame_bgra(width=2, height=2, pitch=8, payload=bytes((i % 251 for i in range(16))))
        shm_a = f"test.shm.ps.a.{uuid.uuid4().hex}"
        code = (
            f"def onStart(ctx):\n"
            f"    ctx['subscribe_video_shm']('k', '{shm_a}', decode='none')\n"
            f"    ctx['subscribe_video_shm']('k', '{shm_b}', decode='none')\n\n"
            f"def onExec(ctx, execIn, inputs):\n"
            f"    items = ctx['list_video_shm_subscriptions']()\n"
            f"    return {{'outputs': {{'out': items}}}}\n"
        )
        try:
            op = _runtime_python_script_node(node_id="psv3", code=code)
            graph = F8RuntimeGraph(graphId="gv3", revision="r1", nodes=[op], edges=[])
            await bus.set_rungraph(graph)

            node = bus.get_node("psv3")
            self.assertIsInstance(node, PythonScriptRuntimeNode)
            assert isinstance(node, PythonScriptRuntimeNode)
            await asyncio.sleep(0.05)
            out = await node.compute_output("out", ctx_id="ctx-z")
            self.assertIsInstance(out, list)
            assert isinstance(out, list)
            self.assertEqual(len(out), 1)
            item = out[0]
            self.assertIsInstance(item, dict)
            assert isinstance(item, dict)
            self.assertEqual(str(item.get("key") or ""), "k")
            self.assertEqual(str(item.get("shmName") or ""), shm_b)
            await node.close()
        finally:
            writer.close(unlink=True)

    async def test_close_clears_video_subscriptions(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        shm_name = f"test.shm.ps.cleanup.{uuid.uuid4().hex}"
        writer = VideoShmWriter(shm_name=shm_name, size=1024 * 1024, slot_count=2)
        writer.open()
        writer.write_frame_bgra(width=2, height=2, pitch=8, payload=bytes((i % 251 for i in range(16))))
        code = (
            f"def onStart(ctx):\n"
            f"    ctx['subscribe_video_shm']('k', '{shm_name}', decode='none')\n\n"
            f"def onExec(ctx, execIn, inputs):\n"
            f"    return {{'outputs': {{'out': 1}}}}\n"
        )
        try:
            op = _runtime_python_script_node(node_id="psv4", code=code)
            graph = F8RuntimeGraph(graphId="gv4", revision="r1", nodes=[op], edges=[])
            await bus.set_rungraph(graph)

            node = bus.get_node("psv4")
            self.assertIsInstance(node, PythonScriptRuntimeNode)
            assert isinstance(node, PythonScriptRuntimeNode)
            await asyncio.sleep(0.05)
            self.assertGreaterEqual(len(node._video_subscriptions), 1)
            await node.close()
            self.assertEqual(len(node._video_subscriptions), 0)
        finally:
            writer.close(unlink=True)


if __name__ == "__main__":
    unittest.main()
