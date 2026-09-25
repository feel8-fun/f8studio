from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    boolean_schema,
    complex_object_schema,
    number_schema,
    string_schema,
)

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.relative_pose_axes"
_AXES = ("L0", "L1", "L2", "R0", "R1", "R2")


def _bone_schema():
    return complex_object_schema(
        properties={
            "pos": number_array_schema(),
            "rot": number_array_schema(),
        }
    )


def number_array_schema():
    return array_schema(items=number_schema())


def _status_schema():
    return complex_object_schema(
        properties={
            "valid": boolean_schema(),
            "reason": string_schema(),
            "primaryAxis": string_schema(),
            **{axis: number_schema() for axis in _AXES},
        }
    )


@dataclass(frozen=True, slots=True)
class _Pose:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


class RelativePoseAxesRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[port.name for port in (node.dataInPorts or [])],
            data_out_ports=[port.name for port in (node.dataOutPorts or [])],
            state_fields=[state.name for state in (node.stateFields or [])],
        )
        state = dict(initial_state or {})
        self._primary_axis = _normalize_primary_axis(state.get("primaryAxis"))
        self._invert_primary = _boolean_or_default(state.get("invertPrimary"), False)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "primaryAxis":
            self._primary_axis = _normalize_primary_axis(value)
        elif name == "invertPrimary":
            self._invert_primary = _boolean_or_default(value, self._invert_primary)

    async def validate_state(
        self,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "primaryAxis":
            normalized = _normalize_primary_axis(value)
            if str(value or "").strip().lower() not in {"local_x", "local_y", "local_z", "distance"}:
                raise ValueError("primaryAxis must be local_x, local_y, local_z, or distance")
            return normalized
        if name == "invertPrimary":
            return _boolean_or_default(value, False)
        return value

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_name = str(port or "").strip()
        if port_name not in {*_AXES, "status"}:
            return None
        reference_raw = await self.pull("referenceBone", ctx_id=ctx_id)
        target_raw = await self.pull("targetBone", ctx_id=ctx_id)
        reference = _parse_pose(reference_raw)
        target = _parse_pose(target_raw)
        if reference is None or target is None:
            if port_name == "status":
                return {"valid": False, "reason": "missing_or_invalid_bone"}
            return None
        axes = _relative_axes(reference, target, self._primary_axis, self._invert_primary)
        if port_name == "status":
            return {
                "valid": True,
                "reason": "ok",
                "primaryAxis": self._primary_axis,
                **axes,
            }
        return axes[port_name]


def _parse_pose(value: Any) -> _Pose | None:
    if not isinstance(value, dict):
        return None
    position_raw = value.get("pos")
    rotation_raw = value.get("rot")
    if not isinstance(position_raw, list) or len(position_raw) != 3:
        return None
    if not isinstance(rotation_raw, list) or len(rotation_raw) != 4:
        return None
    position_values = tuple(_finite_float(item) for item in position_raw)
    rotation_values = tuple(_finite_float(item) for item in rotation_raw)
    if any(item is None for item in position_values) or any(item is None for item in rotation_values):
        return None
    position = (float(position_values[0]), float(position_values[1]), float(position_values[2]))
    rotation = _normalize_quaternion(
        (float(rotation_values[0]), float(rotation_values[1]), float(rotation_values[2]), float(rotation_values[3]))
    )
    if rotation is None:
        return None
    return _Pose(position=position, rotation=rotation)


def _relative_axes(reference: _Pose, target: _Pose, primary_axis: str, invert_primary: bool) -> dict[str, float]:
    world_delta = (
        target.position[0] - reference.position[0],
        target.position[1] - reference.position[1],
        target.position[2] - reference.position[2],
    )
    local_delta = _rotate_vector(_quaternion_inverse(reference.rotation), world_delta)
    if primary_axis == "local_x":
        l0, l1, l2 = local_delta[0], local_delta[1], local_delta[2]
        rotation_order = (0, 1, 2)
    elif primary_axis == "local_z":
        l0, l1, l2 = local_delta[2], local_delta[1], local_delta[0]
        rotation_order = (2, 1, 0)
    elif primary_axis == "distance":
        l0 = math.sqrt(sum(component * component for component in local_delta))
        l1, l2 = local_delta[1], local_delta[0]
        rotation_order = (1, 2, 0)
    else:
        l0, l1, l2 = local_delta[1], local_delta[2], local_delta[0]
        rotation_order = (1, 2, 0)
    if invert_primary:
        l0 = -l0

    relative_rotation = _quaternion_multiply(_quaternion_inverse(reference.rotation), target.rotation)
    euler_xyz = _quaternion_to_euler_degrees(relative_rotation)
    return {
        "L0": float(l0),
        "L1": float(l1),
        "L2": float(l2),
        "R0": float(euler_xyz[rotation_order[0]]),
        "R1": float(euler_xyz[rotation_order[1]]),
        "R2": float(euler_xyz[rotation_order[2]]),
    }


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _normalize_quaternion(value: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    magnitude = math.sqrt(sum(component * component for component in value))
    if magnitude <= 1e-12:
        return None
    return (
        value[0] / magnitude,
        value[1] / magnitude,
        value[2] / magnitude,
        value[3] / magnitude,
    )


def _quaternion_inverse(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (value[0], -value[1], -value[2], -value[3])


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    vector_quaternion = (0.0, vector[0], vector[1], vector[2])
    rotated = _quaternion_multiply(
        _quaternion_multiply(quaternion, vector_quaternion),
        _quaternion_inverse(quaternion),
    )
    return (rotated[1], rotated[2], rotated[3])


def _quaternion_to_euler_degrees(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _normalize_primary_axis(value: object) -> str:
    text = str(value or "local_y").strip().lower()
    return text if text in {"local_x", "local_y", "local_z", "distance"} else "local_y"


def _boolean_or_default(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        if value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        if value.strip().lower() in {"0", "false", "no", "off"}:
            return False
    return default


RelativePoseAxesRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.1.0",
    label="Relative Pose Axes",
    description="Convert a target bone pose into reference-local L0/L1/L2 and R0/R1/R2 signals.",
    tags=["skeleton", "relative", "pose", "axis", "osr", "tcode"],
    dataInPorts=[
        F8DataPortSpec(name="referenceBone", description="Reference bone with pos and rot.", valueSchema=_bone_schema()),
        F8DataPortSpec(name="targetBone", description="Target bone with pos and rot.", valueSchema=_bone_schema()),
    ],
    dataOutPorts=[
        *[
            F8DataPortSpec(name=axis, description=f"Raw relative {axis} signal.", valueSchema=number_schema())
            for axis in _AXES
        ],
        F8DataPortSpec(name="status", description="Per-sample pose calculation status.", valueSchema=_status_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="primaryAxis",
            label="Primary Axis",
            description="Reference-local axis used for L0.",
            valueSchema=string_schema(default="local_y", enum=["local_x", "local_y", "local_z", "distance"]),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="invertPrimary",
            label="Invert L0",
            description="Invert the raw L0 direction before normalization.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(RelativePoseAxesRuntimeNode.SPEC, RelativePoseAxesRuntimeNode, overwrite=True)
    return registry


__all__ = ["RelativePoseAxesRuntimeNode", "register_operator"]
