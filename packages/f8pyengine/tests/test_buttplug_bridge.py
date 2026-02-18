import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

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
from f8pyengine.operators import buttplug_bridge as bridge_mod  # noqa: E402
from f8pyengine.operators.buttplug_bridge import (  # noqa: E402
    ButtplugBridgeRuntimeNode,
    register_operator,
)


@dataclass(frozen=True)
class _FakeOutputDef:
    value: tuple[int, int]
    duration: tuple[int, int] | None = None


@dataclass(frozen=True)
class _FakeInputDef:
    value: list[tuple[int, int]]
    command: list[str]


@dataclass(frozen=True)
class _FakeCommand:
    output_type: str
    value: float
    duration: int | None = None


class _FakeOutputType:
    VIBRATE = "Vibrate"
    ROTATE = "Rotate"
    OSCILLATE = "Oscillate"
    POSITION = "Position"
    POSITION_WITH_DURATION = "HwPositionWithDuration"


class _FakeFeature:
    def __init__(
        self,
        *,
        index: int,
        outputs: dict[str, _FakeOutputDef] | None = None,
        inputs: dict[str, _FakeInputDef] | None = None,
        description: str = "",
    ) -> None:
        self._index = int(index)
        self._outputs = dict(outputs or {})
        self._inputs = dict(inputs or {})
        self._description = str(description)
        self.commands: list[_FakeCommand] = []

    @property
    def index(self) -> int:
        return self._index

    @property
    def description(self) -> str:
        return self._description

    @property
    def outputs(self) -> dict[str, _FakeOutputDef]:
        return self._outputs

    @property
    def inputs(self) -> dict[str, _FakeInputDef]:
        return self._inputs

    def has_output(self, output_type: str) -> bool:
        return str(output_type) in self._outputs

    async def run_output(self, command: _FakeCommand) -> None:
        self.commands.append(command)


class _FakeDevice:
    def __init__(self, *, index: int, name: str, features: dict[int, _FakeFeature], display_name: str = "") -> None:
        self._index = int(index)
        self._name = str(name)
        self._features = dict(features)
        self._display_name = str(display_name)
        self._message_timing_gap = 0
        self.stop_calls = 0

    @property
    def index(self) -> int:
        return self._index

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def message_timing_gap(self) -> int:
        return self._message_timing_gap

    @property
    def features(self) -> dict[int, _FakeFeature]:
        return self._features

    def has_output(self, output_type: str) -> bool:
        return any(feature.has_output(output_type) for feature in self._features.values())

    async def run_output(self, command: _FakeCommand) -> None:
        for feature in self._features.values():
            if feature.has_output(command.output_type):
                await feature.run_output(command)

    async def stop(self, inputs: bool = True, outputs: bool = True) -> None:
        del inputs, outputs
        self.stop_calls += 1


class _FakeClient:
    def __init__(self, *, devices: dict[int, _FakeDevice], fail_connect: bool = False) -> None:
        self._devices = dict(devices)
        self._fail_connect = bool(fail_connect)
        self._connected = False
        self._scanning = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.start_scanning_calls = 0
        self.stop_scanning_calls = 0

        self.on_device_added = None
        self.on_device_removed = None
        self.on_scanning_finished = None
        self.on_server_disconnect = None
        self.on_error = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def scanning(self) -> bool:
        return self._scanning

    @property
    def devices(self) -> dict[int, _FakeDevice]:
        return self._devices

    async def connect(self, url: str) -> None:
        del url
        self.connect_calls += 1
        if self._fail_connect:
            raise RuntimeError("connect failed")
        self._connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False

    async def start_scanning(self) -> None:
        self.start_scanning_calls += 1
        self._scanning = True

    async def stop_scanning(self) -> None:
        self.stop_scanning_calls += 1
        self._scanning = False
        cb = self.on_scanning_finished
        if cb is not None:
            await cb()


class _SlowConnectClient(_FakeClient):
    async def connect(self, url: str) -> None:
        await bridge_mod.asyncio.sleep(0.05)
        await super().connect(url)


class ButtplugBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any]) -> tuple[Any, ButtplugBridgeRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        op = F8RuntimeNode(
            nodeId="bp1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ButtplugBridgeRuntimeNode.SPEC.operatorClass,
            stateFields=list(ButtplugBridgeRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values),
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("bp1")
        self.assertIsInstance(node, ButtplugBridgeRuntimeNode)
        assert isinstance(node, ButtplugBridgeRuntimeNode)
        return bus, node

    async def test_connect_and_publish_device_infos_with_ranges(self) -> None:
        feature = _FakeFeature(
            index=0,
            outputs={"Vibrate": _FakeOutputDef((0, 20)), "HwPositionWithDuration": _FakeOutputDef((0, 20), (100, 10000))},
            inputs={"Battery": _FakeInputDef([(0, 100)], ["Read"])},
            description="Main Motor",
        )
        device = _FakeDevice(index=3, name="ToyA", features={0: feature}, display_name="Toy A")
        client = _FakeClient(devices={3: device})
        symbols = bridge_mod._ButtplugSymbols(  # type: ignore[attr-defined]
            buttplug_client_cls=object,
            device_output_command_cls=_FakeCommand,
            output_type_enum=_FakeOutputType,
        )

        with patch.object(ButtplugBridgeRuntimeNode, "_create_client", return_value=client), patch.object(
            ButtplugBridgeRuntimeNode, "_load_buttplug_symbols", return_value=symbols
        ):
            bus, node = await self._build_node(
                state_values={"enabled": True, "autoConnect": True, "autoScanOnConnect": False, "scanDurationMs": 100}
            )
            await node._tick_once()
            self.assertEqual(client.connect_calls, 1)
            self.assertEqual((await bus.get_state("bp1", "connected")).value, True)

            available = (await bus.get_state("bp1", "availableDevices")).value
            self.assertEqual(available, ["3|ToyA"])
            infos = (await bus.get_state("bp1", "deviceInfos")).value
            self.assertIsInstance(infos, list)
            self.assertEqual(infos[0]["outputs"]["Vibrate"][0]["stepRange"], [0, 20])
            self.assertEqual(infos[0]["outputs"]["HwPositionWithDuration"][0]["durationRange"], [100, 10000])
            await node.close()

    async def test_selected_device_fallback_and_vibrate_feature_targeting(self) -> None:
        f0 = _FakeFeature(index=0, outputs={"Vibrate": _FakeOutputDef((0, 20))}, description="A")
        f1 = _FakeFeature(index=1, outputs={"Vibrate": _FakeOutputDef((0, 20))}, description="B")
        device = _FakeDevice(index=5, name="ToyB", features={0: f0, 1: f1})
        client = _FakeClient(devices={5: device})
        symbols = bridge_mod._ButtplugSymbols(  # type: ignore[attr-defined]
            buttplug_client_cls=object,
            device_output_command_cls=_FakeCommand,
            output_type_enum=_FakeOutputType,
        )

        with patch.object(ButtplugBridgeRuntimeNode, "_load_buttplug_symbols", return_value=symbols):
            bus, node = await self._build_node(
                state_values={"enabled": True, "autoConnect": False, "selectedDevice": "999|missing", "vibrateFeatureIndex": -1}
            )
            node._client = client
            node._bind_client_callbacks(client)
            client._connected = True
            node._client_url = "ws://127.0.0.1:12345"

            async def _pull_all(port: str, *, ctx_id: str | int | None = None) -> Any:
                del ctx_id
                if port == "vibrate":
                    return 0.5
                return None

            node.pull = _pull_all  # type: ignore[method-assign]
            await node.on_exec("e1")
            selected_info = (await bus.get_state("bp1", "selectedDeviceInfo")).value
            self.assertEqual(selected_info["index"], 5)
            self.assertEqual(len(f0.commands), 1)
            self.assertEqual(len(f1.commands), 1)

            await bus.publish_state_external("bp1", "vibrateFeatureIndex", 1, source="test")
            await node.on_exec("e2")
            self.assertEqual(len(f0.commands), 1)
            self.assertEqual(len(f1.commands), 2)
            await node.close()

    async def test_position_is_clamped_to_safe_range(self) -> None:
        feature = _FakeFeature(index=0, outputs={"HwPositionWithDuration": _FakeOutputDef((0, 20), (100, 10000))})
        device = _FakeDevice(index=7, name="ToyPos", features={0: feature})
        client = _FakeClient(devices={7: device})
        symbols = bridge_mod._ButtplugSymbols(  # type: ignore[attr-defined]
            buttplug_client_cls=object,
            device_output_command_cls=_FakeCommand,
            output_type_enum=_FakeOutputType,
        )

        with patch.object(ButtplugBridgeRuntimeNode, "_load_buttplug_symbols", return_value=symbols):
            _bus, node = await self._build_node(
                state_values={
                    "enabled": True,
                    "autoConnect": False,
                    "selectedDevice": "7|ToyPos",
                    "defaultPositionDurationMs": 500,
                }
            )
            node._client = client
            node._bind_client_callbacks(client)
            client._connected = True
            node._client_url = "ws://127.0.0.1:12345"

            async def _pull_hi(port: str, *, ctx_id: str | int | None = None) -> Any:
                del ctx_id
                if port == "position":
                    return 1.0
                if port == "positionDurationMs":
                    return 800
                return None

            node.pull = _pull_hi  # type: ignore[method-assign]
            await node.on_exec("e_hi")
            self.assertEqual(len(feature.commands), 1)
            self.assertAlmostEqual(feature.commands[-1].value, 0.9999, places=6)
            self.assertEqual(feature.commands[-1].duration, 800)

            async def _pull_lo(port: str, *, ctx_id: str | int | None = None) -> Any:
                del ctx_id
                if port == "position":
                    return 0.0
                return None

            node.pull = _pull_lo  # type: ignore[method-assign]
            await node.on_exec("e_lo")
            self.assertEqual(len(feature.commands), 2)
            self.assertAlmostEqual(feature.commands[-1].value, 0.0001, places=6)
            await node.close()

    async def test_rescan_resets_state_and_calls_scan(self) -> None:
        device = _FakeDevice(index=1, name="ToyC", features={})
        client = _FakeClient(devices={1: device})
        bus, node = await self._build_node(
            state_values={"enabled": True, "autoConnect": False, "scanDurationMs": 100, "rescan": False}
        )
        node._client = client
        node._bind_client_callbacks(client)
        client._connected = True
        node._client_url = "ws://127.0.0.1:12345"

        await node.on_state("rescan", True)
        await node._tick_once()
        self.assertEqual(client.start_scanning_calls, 1)
        self.assertEqual(client.stop_scanning_calls, 1)
        self.assertEqual((await bus.get_state("bp1", "rescan")).value, False)
        await node.close()

    async def test_reconnect_is_throttled_after_failure(self) -> None:
        client = _FakeClient(devices={}, fail_connect=True)
        with patch.object(ButtplugBridgeRuntimeNode, "_create_client", return_value=client):
            _bus, node = await self._build_node(
                state_values={"enabled": True, "autoConnect": True, "autoScanOnConnect": False, "reconnectIntervalMs": 60000}
            )
            await node._tick_once()
            await node._tick_once()
            self.assertEqual(client.connect_calls, 1)
            await node.close()

    async def test_stop_on_deactivate_and_close_disconnect(self) -> None:
        feature = _FakeFeature(index=0, outputs={"Vibrate": _FakeOutputDef((0, 20))})
        device = _FakeDevice(index=1, name="ToyD", features={0: feature})
        client = _FakeClient(devices={1: device})
        bus, node = await self._build_node(
            state_values={"enabled": True, "autoConnect": False, "selectedDevice": "1|ToyD", "stopOnDeactivate": True}
        )
        node._client = client
        node._bind_client_callbacks(client)
        client._connected = True
        node._client_url = "ws://127.0.0.1:12345"

        await node.on_lifecycle(False, {"case": "deactivate"})
        self.assertEqual(device.stop_calls, 1)
        await node.close()
        self.assertEqual(client.disconnect_calls, 1)
        self.assertEqual((await bus.get_state("bp1", "lastCommandTsMs")).value > 0, True)

    async def test_tick_once_is_mutex_guarded(self) -> None:
        client = _SlowConnectClient(devices={})
        with patch.object(ButtplugBridgeRuntimeNode, "_create_client", return_value=client):
            _bus, node = await self._build_node(
                state_values={"enabled": True, "autoConnect": True, "autoScanOnConnect": False, "reconnectIntervalMs": 0}
            )
            await bridge_mod.asyncio.gather(node._tick_once(), node._tick_once())
            self.assertEqual(client.connect_calls, 1)
            await node.close()


if __name__ == "__main__":
    unittest.main()
