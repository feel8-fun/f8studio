from __future__ import annotations

from typing import Any, Callable

import pytest

from f8pyengine.constants import SERVICE_CLASS
from f8pyengine.operators.relative_pose_axes import RelativePoseAxesRuntimeNode, register_operator as register_pose_axes
from f8pyengine.operators.skeleton_selector import SkeletonSelectorRuntimeNode, register_operator as register_selector
from f8pyengine.operators.stream_watchdog import StreamWatchdogRuntimeNode, register_operator as register_watchdog
from f8pysdk.host import ServiceHost, ServiceHostConfig
from f8pysdk.registry import Registry, create_runtime_node_registry
from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode
from f8pysdk.testing import ServiceBusHarness, buffer_input
from f8pysdk.time_utils import now_ms

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _build_node(runtime_type: type, register: Callable[[Registry], Registry], state_values: dict[str, Any]):
    harness = ServiceBusHarness()
    bus = harness.create_bus("svcA")
    runtime_registry = create_runtime_node_registry()
    register(Registry.wrap(runtime_registry))
    ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=runtime_registry)
    spec = runtime_type.SPEC
    runtime_node = F8RuntimeNode(
        nodeId="node1",
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=spec.operatorClass,
        stateFields=list(spec.stateFields or []),
        stateValues=state_values,
        dataInPorts=list(spec.dataInPorts or []),
        dataOutPorts=list(spec.dataOutPorts or []),
        execInPorts=list(spec.execInPorts or []),
        execOutPorts=list(spec.execOutPorts or []),
    )
    await bus.set_rungraph(F8RuntimeGraph(graphId="g1", revision="r1", nodes=[runtime_node], edges=[]))
    node = bus.get_node("node1")
    assert isinstance(node, runtime_type)
    return bus, node


def _skeleton(*, profile_id: str, role: str, role_index: int, model_name: str) -> dict[str, Any]:
    stable_key = f"{profile_id}:{role.lower()}:{role_index}"
    return {
        "type": "skeleton_binary",
        "modelName": model_name,
        "stableKey": stable_key,
        "receivedAtMs": int(now_ms()),
        "bones": [],
        "trailer": {
            "extVersion": 2,
            "profileId": profile_id,
            "role": role,
            "roleIndex": role_index,
            "stableKey": stable_key,
        },
    }


async def test_skeleton_selector_uses_stable_role_identity() -> None:
    bus, node = await _build_node(
        SkeletonSelectorRuntimeNode,
        register_selector,
        {"profileId": "hs2", "role": "female", "roleIndex": 0},
    )
    skeletons = [
        _skeleton(profile_id="hs2", role="Male", role_index=0, model_name="11|TransientMale"),
        _skeleton(profile_id="hs2", role="Female", role_index=0, model_name="22|TransientFemale"),
    ]
    buffer_input(bus, "node1", "skeletons", skeletons, ts_ms=1, edge=None, ctx_id=1)

    selected = await node.compute_output("skeleton", ctx_id=1)
    status = await node.compute_output("status", ctx_id=1)

    assert selected["modelName"] == "22|TransientFemale"
    assert status == {
        "valid": True,
        "stableKey": "hs2:female:0",
        "profileId": "hs2",
        "role": "female",
        "roleIndex": 0,
        "reason": "stable_identity",
    }


async def test_skeleton_selector_does_not_guess_when_identity_is_missing() -> None:
    bus, node = await _build_node(
        SkeletonSelectorRuntimeNode,
        register_selector,
        {"profileId": "hs2", "role": "female", "roleIndex": 0},
    )
    buffer_input(bus, "node1", "skeletons", [{"modelName": "123|Legacy", "bones": []}], ts_ms=1, edge=None, ctx_id=1)

    assert await node.compute_output("skeleton", ctx_id=1) is None


async def test_relative_pose_axes_maps_reference_local_y_to_l0() -> None:
    bus, node = await _build_node(RelativePoseAxesRuntimeNode, register_pose_axes, {"primaryAxis": "local_y"})
    reference = {"name": "MalePenisBase", "pos": [1.0, 2.0, 3.0], "rot": [1.0, 0.0, 0.0, 0.0]}
    target = {"name": "Vagina", "pos": [1.25, 2.8, 2.9], "rot": [1.0, 0.0, 0.0, 0.0]}
    buffer_input(bus, "node1", "referenceBone", reference, ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target, ts_ms=1, edge=None, ctx_id=1)

    assert await node.compute_output("L0", ctx_id=1) == pytest.approx(0.8)
    assert await node.compute_output("L1", ctx_id=1) == pytest.approx(-0.1)
    assert await node.compute_output("L2", ctx_id=1) == pytest.approx(0.25)
    assert await node.compute_output("R0", ctx_id=1) == pytest.approx(0.0)


async def test_stream_watchdog_gates_stale_data() -> None:
    bus, node = await _build_node(StreamWatchdogRuntimeNode, register_watchdog, {"timeoutMs": 250})
    fresh = [{"receivedAtMs": int(now_ms()), "bones": []}]
    buffer_input(bus, "node1", "value", fresh, ts_ms=1, edge=None, ctx_id=1)

    assert await node.compute_output("valid", ctx_id=1) is True
    assert await node.on_exec(1, "check") == ["valid"]

    stale = [{"receivedAtMs": int(now_ms()) - 1000, "bones": []}]
    buffer_input(bus, "node1", "value", stale, ts_ms=2, edge=None, ctx_id=2)
    assert await node.compute_output("value", ctx_id=2) is None
    assert await node.on_exec(2, "check") == []
