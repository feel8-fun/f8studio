import asyncio
import os
import sys
import unittest
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.generated import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.handy_out import (  # noqa: E402
    HandyOutRuntimeNode,
    _HttpResult,
    register_operator,
)


class HandyOutTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any]) -> tuple[Any, HandyOutRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        op = F8RuntimeNode(
            nodeId="handy1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=HandyOutRuntimeNode.SPEC.operatorClass,
            stateFields=list(HandyOutRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values),
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("handy1")
        self.assertIsInstance(node, HandyOutRuntimeNode)
        assert isinstance(node, HandyOutRuntimeNode)
        return bus, node

    async def test_maps_0_1_to_hdsp_and_ensures_mode_once(self) -> None:
        bus, node = await self._build_node(
            state_values={
                "enabled": True,
                "connectionKey": "abcde12345",
                "ensureHdspMode": True,
                "minPercent": 10,
                "maxPercent": 90,
                "invert": False,
            }
        )

        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, path: str, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append({"path": path, "payload": dict(payload)})
            if path == "mode":
                return _HttpResult(status_code=200, headers={}, json_body={"result": 0}, error_message="")
            return _HttpResult(status_code=200, headers={}, json_body={"result": 1}, error_message="")

        value_ref = {"value": 0.5}

        async def _pull(port: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            if port == "value":
                return value_ref["value"]
            return None

        node._http_put_json = _fake_http  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]

        await node.on_exec("e1")
        await asyncio.sleep(0.05)

        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0]["path"], "mode")
        self.assertEqual(calls[1]["path"], "hdsp/xpt")
        self.assertAlmostEqual(float(calls[1]["payload"]["position"]), 50.0, places=6)

        await bus.publish_state_external("handy1", "invert", True, source="test")
        value_ref["value"] = 0.0
        await node.on_exec("e2")
        await asyncio.sleep(0.05)
        self.assertEqual(calls[-1]["path"], "hdsp/xpt")
        self.assertAlmostEqual(float(calls[-1]["payload"]["position"]), 90.0, places=6)
        await node.close()

    async def test_disabled_or_none_value_sends_nothing(self) -> None:
        bus, node = await self._build_node(
            state_values={
                "enabled": False,
                "connectionKey": "abcde12345",
                "ensureHdspMode": False,
            }
        )
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, path: str, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append({"path": path, "payload": dict(payload)})
            return _HttpResult(status_code=200, headers={}, json_body={"result": 0}, error_message="")

        async def _pull_disabled(port: str, *, ctx_id: str | int | None = None) -> Any:
            del port, ctx_id
            return 0.5

        node._http_put_json = _fake_http  # type: ignore[method-assign]
        node.pull = _pull_disabled  # type: ignore[method-assign]
        await node.on_exec("e1")
        await asyncio.sleep(0.05)
        self.assertEqual(calls, [])

        await bus.publish_state_external("handy1", "enabled", True, source="test")

        async def _pull_none(port: str, *, ctx_id: str | int | None = None) -> Any:
            del port, ctx_id
            return None

        node.pull = _pull_none  # type: ignore[method-assign]
        await node.on_exec("e2")
        await asyncio.sleep(0.05)
        self.assertEqual(calls, [])
        await node.close()

    async def test_duration_input_overrides_default_duration(self) -> None:
        _bus, node = await self._build_node(
            state_values={
                "enabled": True,
                "connectionKey": "abcde12345",
                "ensureHdspMode": False,
                "defaultDurationMs": 100,
            }
        )
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, path: str, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append({"path": path, "payload": dict(payload)})
            return _HttpResult(status_code=200, headers={}, json_body={"result": 0}, error_message="")

        async def _pull(port: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            if port == "value":
                return 0.2
            if port == "durationMs":
                return 250
            return None

        node._http_put_json = _fake_http  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]
        await node.on_exec("e1")
        await asyncio.sleep(0.05)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["path"], "hdsp/xpt")
        self.assertEqual(int(calls[0]["payload"]["duration"]), 250)
        await node.close()

    async def test_api_error_updates_last_error(self) -> None:
        bus, node = await self._build_node(
            state_values={
                "enabled": True,
                "connectionKey": "abcde12345",
                "ensureHdspMode": False,
            }
        )

        async def _fake_http(*, cfg: Any, path: str, payload: dict[str, Any]) -> _HttpResult:
            del cfg, path, payload
            return _HttpResult(
                status_code=200,
                headers={},
                json_body={"error": {"code": 3000, "name": "HampError", "message": "HampError"}},
                error_message="",
            )

        async def _pull(port: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            if port == "value":
                return 0.5
            return None

        node._http_put_json = _fake_http  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]
        await node.on_exec("e1")
        await asyncio.sleep(0.05)

        last_error = (await bus.get_state("handy1", "lastError")).value
        self.assertIn("3000", str(last_error))
        self.assertEqual((await bus.get_state("handy1", "lastHttpStatus")).value, 200)
        await node.close()

    async def test_rate_limit_backoff_drops_following_exec(self) -> None:
        bus, node = await self._build_node(
            state_values={
                "enabled": True,
                "connectionKey": "abcde12345",
                "ensureHdspMode": False,
            }
        )
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, path: str, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append({"path": path, "payload": dict(payload)})
            return _HttpResult(
                status_code=200,
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "500"},
                json_body={"result": 0},
                error_message="",
            )

        async def _pull(port: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            if port == "value":
                return 0.7
            return None

        node._http_put_json = _fake_http  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]
        await node.on_exec("e1")
        await asyncio.sleep(0.05)
        self.assertEqual(len(calls), 1)

        await node.on_exec("e2")
        await asyncio.sleep(0.02)
        self.assertEqual(len(calls), 1)
        dropped = (await bus.get_state("handy1", "droppedCommands")).value
        self.assertGreaterEqual(int(dropped), 1)
        await node.close()

    async def test_validate_state_rejects_invalid_values(self) -> None:
        _bus, node = await self._build_node(
            state_values={
                "enabled": True,
                "connectionKey": "abcde12345",
                "maxPercent": 40.0,
            }
        )
        with self.assertRaises(ValueError):
            await node.validate_state("connectionKey", "bad!", ts_ms=1, meta={})
        with self.assertRaises(ValueError):
            await node.validate_state("baseUrl", "ftp://example.com", ts_ms=2, meta={})
        with self.assertRaises(ValueError):
            await node.validate_state("minPercent", 60.0, ts_ms=3, meta={})
        with self.assertRaises(ValueError):
            await node.validate_state("requestTimeoutMs", -1, ts_ms=4, meta={})
        await node.close()


if __name__ == "__main__":
    unittest.main()
