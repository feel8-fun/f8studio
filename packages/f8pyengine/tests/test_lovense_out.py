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

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.lovense_out import (  # noqa: E402
    LovenseOutRuntimeNode,
    _HttpResult,
    register_operator,
)


class LovenseOutTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any]) -> tuple[Any, LovenseOutRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        op = F8RuntimeNode(
            nodeId="lovense1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=LovenseOutRuntimeNode.SPEC.operatorClass,
            stateFields=list(LovenseOutRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values),
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("lovense1")
        self.assertIsInstance(node, LovenseOutRuntimeNode)
        assert isinstance(node, LovenseOutRuntimeNode)
        return bus, node

    async def test_exec_position_maps_and_sends_position_command(self) -> None:
        _bus, node = await self._build_node(state_values={"enabled": True, "minSendIntervalMs": 0, "toy": "toy-1"})
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append(dict(payload))
            return _HttpResult(status_code=200, headers={}, body={"code": 200, "type": "ok"}, raw_body="", error_message="")

        value_ref = {"value": 0.0}

        async def _pull(port: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            if port == "position":
                return value_ref["value"]
            return None

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]

        value_ref["value"] = 0.0
        await node.on_exec("e0", "sendPositionCmd")
        value_ref["value"] = 0.5
        await node.on_exec("e1", "sendPositionCmd")
        value_ref["value"] = 1.0
        await node.on_exec("e2", "sendPositionCmd")

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], {"command": "Position", "value": "0", "apiVer": 1, "toy": "toy-1"})
        self.assertEqual(calls[1], {"command": "Position", "value": "50", "apiVer": 1, "toy": "toy-1"})
        self.assertEqual(calls[2], {"command": "Position", "value": "100", "apiVer": 1, "toy": "toy-1"})

    async def test_exec_without_position_does_not_send(self) -> None:
        _bus, node = await self._build_node(state_values={"enabled": True, "minSendIntervalMs": 0})
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append(dict(payload))
            return _HttpResult(status_code=200, headers={}, body={"code": 200}, raw_body="", error_message="")

        async def _pull(port: str, *, ctx_id: str | int | None = None) -> Any:
            del port, ctx_id
            return None

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]
        await node.on_exec("e1", "sendPositionCmd")

        self.assertEqual(calls, [])

    async def test_apply_builds_function_action_from_state(self) -> None:
        _bus, node = await self._build_node(
            state_values={
                "enabled": True,
                "vibrate": 0.5,
                "rotate": 1.0,
                "timeSec": 3,
                "stopPrevious": False,
                "loopRunningSec": 2,
                "loopPauseSec": 1,
                "toy": "toy-123",
            }
        )
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append(dict(payload))
            return _HttpResult(status_code=200, headers={}, body={"code": 200, "type": "ok"}, raw_body="", error_message="")

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        await node.on_exec("a1", "sendFunctionCmd")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["command"], "Function")
        self.assertEqual(calls[0]["action"], "Vibrate:10,Rotate:20")
        self.assertEqual(calls[0]["timeSec"], 3.0)
        self.assertEqual(calls[0]["stopPrevious"], 0)
        self.assertEqual(calls[0]["loopRunningSec"], 2.0)
        self.assertEqual(calls[0]["loopPauseSec"], 1.0)
        self.assertEqual(calls[0]["toy"], "toy-123")
        self.assertEqual(calls[0]["apiVer"], 1)

    async def test_apply_omits_optional_fields_when_empty(self) -> None:
        _bus, node = await self._build_node(
            state_values={
                "enabled": True,
                "vibrate": 0.1,
                "timeSec": 0,
                "stopPrevious": True,
                "toy": "",
                "defaultToy": "",
            }
        )
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append(dict(payload))
            return _HttpResult(status_code=200, headers={}, body={"code": 200}, raw_body="", error_message="")

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        await node.on_exec("a1", "sendFunctionCmd")

        self.assertEqual(len(calls), 1)
        self.assertNotIn("loopRunningSec", calls[0])
        self.assertNotIn("loopPauseSec", calls[0])
        self.assertNotIn("toy", calls[0])

    async def test_apply_stop_has_priority(self) -> None:
        _bus, node = await self._build_node(
            state_values={
                "enabled": True,
                "stop": True,
                "vibrate": 1.0,
                "timeSec": 9,
                "defaultToy": "toy-default",
            }
        )
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append(dict(payload))
            return _HttpResult(status_code=200, headers={}, body={"code": 200}, raw_body="", error_message="")

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        await node.on_exec("a1", "sendFunctionCmd")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["command"], "Function")
        self.assertEqual(calls[0]["action"], "Stop")
        self.assertEqual(calls[0]["timeSec"], 0)
        self.assertEqual(calls[0]["toy"], "toy-default")

    async def test_apply_stroke_pair_validation_error(self) -> None:
        _bus, node = await self._build_node(state_values={"enabled": True, "strokeMin": 0.2, "timeSec": 0})
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append(dict(payload))
            return _HttpResult(status_code=200, headers={}, body={"code": 200}, raw_body="", error_message="")

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        await node.on_exec("a1", "sendFunctionCmd")

        self.assertEqual(calls, [])
        self.assertIn("strokeMin and strokeMax", str(node._last_error_message))

    async def test_position_min_send_interval_throttles(self) -> None:
        _bus, node = await self._build_node(state_values={"enabled": True, "minSendIntervalMs": 1000})
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append(dict(payload))
            return _HttpResult(status_code=200, headers={}, body={"code": 200}, raw_body="", error_message="")

        async def _pull(port: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            if port == "position":
                return 0.5
            return None

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]

        await node.on_exec("e1", "sendPositionCmd")
        await node.on_exec("e2", "sendPositionCmd")

        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(int(node._dropped_commands), 1)

    async def test_spec_ports_match_new_contract(self) -> None:
        spec = LovenseOutRuntimeNode.SPEC
        data_in_names = [p.name for p in (spec.dataInPorts or [])]
        data_out_names = [p.name for p in (spec.dataOutPorts or [])]
        self.assertEqual(data_in_names, ["position"])
        self.assertEqual(data_out_names, [])
        self.assertEqual([port.name for port in (spec.execInPorts or [])], ["sendPositionCmd", "sendFunctionCmd"])
        toy_state = None
        for state_spec in list(spec.stateFields or []):
            if state_spec.name == "toy":
                toy_state = state_spec
                break
        self.assertIsNotNone(toy_state)
        assert toy_state is not None
        self.assertEqual(toy_state.control.kind.value, "select")
        self.assertEqual(toy_state.control.optionsFromState, "availableToys")

    async def test_api_error_updates_last_error(self) -> None:
        _bus, node = await self._build_node(state_values={"enabled": True, "timeSec": 0, "vibrate": 0.2})

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg, payload
            return _HttpResult(
                status_code=200,
                headers={},
                body={"code": 404, "type": "error", "message": "Invalid Parameter"},
                raw_body='{"code":404,"type":"error","message":"Invalid Parameter"}',
                error_message="",
            )

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        await node.on_exec("a1", "sendFunctionCmd")

        self.assertIn("404", str(node._last_error_message))

    async def test_validate_state_rejects_invalid_values(self) -> None:
        _bus, node = await self._build_node(state_values={"enabled": True})
        with self.assertRaises(ValueError):
            await node.validate_state("commandUrl", "ftp://example.com/command", ts_ms=1, meta={})
        with self.assertRaises(ValueError):
            await node.validate_state("requestTimeoutMs", 10, ts_ms=2, meta={})
        with self.assertRaises(ValueError):
            await node.validate_state("minSendIntervalMs", -1, ts_ms=3, meta={})
        with self.assertRaises(ValueError):
            await node.validate_state("vibrate", 1.5, ts_ms=4, meta={})
        with self.assertRaises(ValueError):
            await node.validate_state("timeSec", -1, ts_ms=5, meta={})

    async def test_legacy_exec_port_names_are_rejected(self) -> None:
        _bus, node = await self._build_node(state_values={"enabled": True, "minSendIntervalMs": 0, "vibrate": 0.5})
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append(dict(payload))
            return _HttpResult(status_code=200, headers={}, body={"code": 200}, raw_body="", error_message="")

        async def _pull(port: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            if port == "position":
                return 0.4
            return None

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]

        await node.on_exec("e1", "exec")
        await node.on_exec("a1", "apply")

        self.assertEqual(calls, [])
        self.assertIn("unsupported exec in port", str(node._last_error_message))

    async def test_active_lifecycle_auto_refreshes_available_toys_once(self) -> None:
        bus, node = await self._build_node(state_values={"enabled": True})
        calls: list[dict[str, Any]] = []

        async def _fake_http(*, cfg: Any, payload: dict[str, Any]) -> _HttpResult:
            del cfg
            calls.append(dict(payload))
            return _HttpResult(
                status_code=200,
                headers={},
                body={
                    "code": 200,
                    "type": "OK",
                    "data": {
                        "toys": '{"f082c00246fa":{"id":"f082c00246fa","name":"nora"},"ab12":{"id":"ab12","name":"domi"}}'
                    },
                },
                raw_body="",
                error_message="",
            )

        node._http_post_json = _fake_http  # type: ignore[method-assign]
        await node.on_lifecycle(False, {})
        await node.on_lifecycle(True, {})
        await node.on_lifecycle(True, {})

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["command"], "GetToys")
        self.assertEqual(calls[0]["apiVer"], 1)
        available_toys = (await bus.get_state("lovense1", "availableToys")).value
        self.assertEqual(available_toys, ["f082c00246fa", "ab12"])


if __name__ == "__main__":
    unittest.main()
