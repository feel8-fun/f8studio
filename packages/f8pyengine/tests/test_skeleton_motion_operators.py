from __future__ import annotations

import math
from typing import Any, Callable

import pytest

from f8pyengine.constants import SERVICE_CLASS
from f8pyengine.operators.contact_pose_axes import (
    ContactPoseAxesRuntimeNode,
    register_operator as register_contact_pose_axes,
)
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


class _CountingContactPoseAxesRuntimeNode(ContactPoseAxesRuntimeNode):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pull_count = 0

    async def pull(self, port: str, *, ctx_id: str | int | None = None) -> Any:
        self.pull_count += 1
        return await super().pull(port, ctx_id=ctx_id)


def _register_counting_contact_pose_axes(registry: Registry) -> Registry:
    registry.register_operator(
        ContactPoseAxesRuntimeNode.SPEC,
        _CountingContactPoseAxesRuntimeNode,
        overwrite=True,
    )
    return registry


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


def _contact_reference_skeleton() -> dict[str, Any]:
    identity = [1.0, 0.0, 0.0, 0.0]
    return {
        "bones": [
            {"name": "Penis01", "pos": [0.0, 0.0, 0.0], "rot": identity},
            {"name": "Penis02", "pos": [0.0, 0.0, 0.25], "rot": identity},
            {"name": "Penis09", "pos": [0.0, 0.0, 1.0], "rot": identity},
            {"name": "M_Hips", "pos": [0.0, 0.0, 0.0], "rot": identity},
        ]
    }


def _contact_state() -> dict[str, Any]:
    return {
        "supportRightAxis": "+local_x",
        "supportUpAxis": "+local_y",
        "targetUpAxis": "+local_z",
        "targetRightAxis": "+local_x",
        "l0MinMeters": 0.0,
        "l0MaxMeters": 1.0,
        "lateralRangeMeters": 1.0,
        "twistRangeDegrees": 90.0,
        "tiltRangeDegrees": 45.0,
        "radiusScale": 0.5,
    }


async def test_contact_pose_axes_maps_orthogonal_reference_frame() -> None:
    bus, node = await _build_node(ContactPoseAxesRuntimeNode, register_contact_pose_axes, _contact_state())
    target = {
        "name": "R_Hand",
        "pos": [-0.4, -0.2, 0.25],
        "rot": [1.0, 0.0, 0.0, 0.0],
    }
    buffer_input(bus, "node1", "referenceSkeleton", _contact_reference_skeleton(), ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target, ts_ms=1, edge=None, ctx_id=1)

    assert await node.compute_output("L0", ctx_id=1) == pytest.approx(0.25)
    assert await node.compute_output("L1", ctx_id=1) == pytest.approx(0.6)
    assert await node.compute_output("L2", ctx_id=1) == pytest.approx(0.3)
    assert await node.compute_output("R0", ctx_id=1) == pytest.approx(0.5)
    assert await node.compute_output("R1", ctx_id=1) == pytest.approx(0.5)
    assert await node.compute_output("R2", ctx_id=1) == pytest.approx(0.5)

    frame = await node.compute_output("frame", ctx_id=1)
    assert frame["axes"]["L0"] == pytest.approx(0.25)
    assert frame["axes"]["L1"] == pytest.approx(0.6)
    assert frame["status"]["valid"] is True


async def test_contact_pose_axes_uses_target_supplied_hand_basis() -> None:
    bus, node = await _build_node(ContactPoseAxesRuntimeNode, register_contact_pose_axes, {})
    target = {
        "name": "R_Hand",
        "pos": [0.0, 0.0, 0.15],
        "rot": [1.0, 0.0, 0.0, 0.0],
        "basis": {"up": "+local_z", "right": "-local_y"},
    }
    buffer_input(bus, "node1", "referenceSkeleton", _contact_reference_skeleton(), ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target, ts_ms=1, edge=None, ctx_id=1)

    status = await node.compute_output("status", ctx_id=1)
    assert status["pitchDegrees"] == pytest.approx(0.0)
    assert status["rollDegrees"] == pytest.approx(0.0)
    assert status["R2"] == pytest.approx(0.5)


async def test_contact_pose_axes_uses_bilateral_center_without_false_tilt() -> None:
    state = {**_contact_state(), "l0MinMeters": 0.8, "l0MaxMeters": 1.0}
    bus, node = await _build_node(ContactPoseAxesRuntimeNode, register_contact_pose_axes, state)
    target = {
        "name": "R_Foot+L_Foot",
        "pos": [0.0, 0.0, 0.3],
        # Deliberately tilted: bilateral mode must not use either ankle as the
        # orientation of the combined contact sleeve.
        "rot": [0.70710678, 0.70710678, 0.0, 0.0],
        "targetMode": "bilateral_reference_axis",
        "pairPositions": [[-0.2, 0.0, 0.3], [0.2, 0.0, 0.3]],
    }
    buffer_input(bus, "node1", "referenceSkeleton", _contact_reference_skeleton(), ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target, ts_ms=1, edge=None, ctx_id=1)

    frame = await node.compute_output("frame", ctx_id=1)

    assert frame["axes"]["L0"] == pytest.approx(0.3)
    assert frame["axes"]["R1"] == pytest.approx(0.5)
    assert frame["axes"]["R2"] == pytest.approx(0.5)
    assert frame["status"]["rollDegrees"] == pytest.approx(0.0)
    assert frame["status"]["pitchDegrees"] == pytest.approx(0.0)
    assert frame["status"]["targetMode"] == "bilateral_reference_axis"


async def test_contact_pose_axes_uses_signed_twist_around_reference_axis() -> None:
    bus, node = await _build_node(ContactPoseAxesRuntimeNode, register_contact_pose_axes, _contact_state())
    def target_at(degrees: float) -> dict[str, Any]:
        half_angle = math.radians(degrees) / 2.0
        return {
            "name": "R_Hand",
            "pos": [0.0, 0.0, 0.5],
            "rot": [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
        }

    buffer_input(bus, "node1", "referenceSkeleton", _contact_reference_skeleton(), ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target_at(30.0), ts_ms=1, edge=None, ctx_id=1)

    initial = await node.compute_output("status", ctx_id=1)
    assert initial["R0"] == pytest.approx(0.5)
    assert initial["rawTwistDegrees"] == pytest.approx(30.0)
    assert initial["twistBaselineDegrees"] == pytest.approx(30.0)
    assert initial["twistDegrees"] == pytest.approx(0.0)

    buffer_input(bus, "node1", "referenceSkeleton", _contact_reference_skeleton(), ts_ms=2, edge=None, ctx_id=2)
    buffer_input(bus, "node1", "targetBone", target_at(75.0), ts_ms=2, edge=None, ctx_id=2)

    assert await node.compute_output("R0", ctx_id=2) == pytest.approx(0.75)
    status = await node.compute_output("status", ctx_id=2)
    assert status["twistDegrees"] == pytest.approx(45.0)


async def test_contact_pose_axes_unwraps_rotation_across_signed_angle_seam() -> None:
    bus, node = await _build_node(ContactPoseAxesRuntimeNode, register_contact_pose_axes, _contact_state())
    reference = _contact_reference_skeleton()

    def target_at(degrees: float) -> dict[str, Any]:
        half_angle = math.radians(degrees) / 2.0
        return {
            "name": "R_Hand",
            "pos": [0.0, 0.0, 0.5],
            "rot": [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
        }

    buffer_input(bus, "node1", "referenceSkeleton", reference, ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target_at(179.0), ts_ms=1, edge=None, ctx_id=1)
    first = await node.compute_output("status", ctx_id=1)

    buffer_input(bus, "node1", "referenceSkeleton", reference, ts_ms=2, edge=None, ctx_id=2)
    buffer_input(bus, "node1", "targetBone", target_at(-179.0), ts_ms=2, edge=None, ctx_id=2)
    second = await node.compute_output("status", ctx_id=2)

    assert first["rawTwistDegrees"] == pytest.approx(179.0)
    assert second["rawTwistDegrees"] == pytest.approx(181.0)
    assert first["twistDegrees"] == pytest.approx(0.0)
    assert second["twistDegrees"] == pytest.approx(2.0)
    assert first["R0"] == pytest.approx(0.5)
    assert second["R0"] == pytest.approx(0.5 + 2.0 / 180.0)


async def test_contact_pose_axes_reacquires_twist_baseline_when_binding_changes() -> None:
    bus, node = await _build_node(ContactPoseAxesRuntimeNode, register_contact_pose_axes, _contact_state())

    def reference_for(pose_id: str) -> dict[str, Any]:
        reference = _contact_reference_skeleton()
        reference["stableKey"] = "fallen-doll:male:0"
        reference["trailer"] = {"poseId": pose_id, "bindingGeneration": 1}
        return reference

    def target_at(degrees: float) -> dict[str, Any]:
        half_angle = math.radians(degrees) / 2.0
        return {
            "name": "M_Gen",
            "pos": [0.0, 0.0, 0.5],
            "rot": [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
        }

    buffer_input(bus, "node1", "referenceSkeleton", reference_for("PoseA"), ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target_at(30.0), ts_ms=1, edge=None, ctx_id=1)
    first = await node.compute_output("status", ctx_id=1)

    buffer_input(bus, "node1", "referenceSkeleton", reference_for("PoseA"), ts_ms=2, edge=None, ctx_id=2)
    buffer_input(bus, "node1", "targetBone", target_at(60.0), ts_ms=2, edge=None, ctx_id=2)
    moved = await node.compute_output("status", ctx_id=2)

    buffer_input(bus, "node1", "referenceSkeleton", reference_for("PoseB"), ts_ms=3, edge=None, ctx_id=3)
    buffer_input(bus, "node1", "targetBone", target_at(80.0), ts_ms=3, edge=None, ctx_id=3)
    rebound = await node.compute_output("status", ctx_id=3)

    assert first["twistDegrees"] == pytest.approx(0.0)
    assert moved["twistDegrees"] == pytest.approx(30.0)
    assert moved["twistBaselineDegrees"] == pytest.approx(30.0)
    assert rebound["twistDegrees"] == pytest.approx(0.0)
    assert rebound["twistBaselineDegrees"] == pytest.approx(80.0)


async def test_contact_pose_axes_calculates_once_per_context() -> None:
    bus, node = await _build_node(
        _CountingContactPoseAxesRuntimeNode,
        _register_counting_contact_pose_axes,
        _contact_state(),
    )
    target = {"name": "R_Hand", "pos": [0.0, 0.0, 0.5], "rot": [1.0, 0.0, 0.0, 0.0]}
    buffer_input(bus, "node1", "referenceSkeleton", _contact_reference_skeleton(), ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target, ts_ms=1, edge=None, ctx_id=1)

    for port in ("L0", "L1", "L2", "R0", "R1", "R2", "status"):
        assert await node.compute_output(port, ctx_id=1) is not None
    assert node.pull_count == 2

    buffer_input(bus, "node1", "referenceSkeleton", _contact_reference_skeleton(), ts_ms=2, edge=None, ctx_id=2)
    buffer_input(bus, "node1", "targetBone", target, ts_ms=2, edge=None, ctx_id=2)
    assert await node.compute_output("L0", ctx_id=2) == pytest.approx(0.5)
    assert node.pull_count == 4


async def test_contact_pose_axes_state_change_invalidates_context_cache() -> None:
    bus, node = await _build_node(ContactPoseAxesRuntimeNode, register_contact_pose_axes, _contact_state())
    target = {"name": "R_Hand", "pos": [0.0, 0.0, 0.25], "rot": [1.0, 0.0, 0.0, 0.0]}
    buffer_input(bus, "node1", "referenceSkeleton", _contact_reference_skeleton(), ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target, ts_ms=1, edge=None, ctx_id=1)

    assert await node.compute_output("L0", ctx_id=1) == pytest.approx(0.25)
    await node.on_state("invertL0", True)
    assert await node.compute_output("L0", ctx_id=1) == pytest.approx(0.75)


async def test_contact_pose_axes_requires_extended_tip_and_support_bones() -> None:
    bus, node = await _build_node(ContactPoseAxesRuntimeNode, register_contact_pose_axes, _contact_state())
    incomplete = _contact_reference_skeleton()
    incomplete["bones"] = [bone for bone in incomplete["bones"] if bone["name"] != "Penis09"]
    target = {"name": "R_Hand", "pos": [0.0, 0.0, 0.5], "rot": [1.0, 0.0, 0.0, 0.0]}
    buffer_input(bus, "node1", "referenceSkeleton", incomplete, ts_ms=1, edge=None, ctx_id=1)
    buffer_input(bus, "node1", "targetBone", target, ts_ms=1, edge=None, ctx_id=1)

    assert await node.compute_output("L0", ctx_id=1) is None
    assert await node.compute_output("status", ctx_id=1) == {
        "valid": False,
        "contactValid": False,
        "reason": "missing_or_invalid_input",
    }


async def test_contact_pose_axes_reports_inside_and_outside_contact_radius() -> None:
    bus, node = await _build_node(ContactPoseAxesRuntimeNode, register_contact_pose_axes, _contact_state())
    buffer_input(bus, "node1", "referenceSkeleton", _contact_reference_skeleton(), ts_ms=1, edge=None, ctx_id=1)
    inside = {"name": "R_Hand", "pos": [0.49, 0.0, 0.5], "rot": [1.0, 0.0, 0.0, 0.0]}
    buffer_input(bus, "node1", "targetBone", inside, ts_ms=1, edge=None, ctx_id=1)
    assert (await node.compute_output("status", ctx_id=1))["contactValid"] is True

    outside = {"name": "R_Hand", "pos": [0.51, 0.0, 0.5], "rot": [1.0, 0.0, 0.0, 0.0]}
    buffer_input(bus, "node1", "targetBone", outside, ts_ms=2, edge=None, ctx_id=2)
    status = await node.compute_output("status", ctx_id=2)
    assert status["contactValid"] is False
    assert status["reason"] == "outside_contact_radius"


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
