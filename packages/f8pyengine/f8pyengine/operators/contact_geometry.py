from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


AXIS_NAMES = ("L0", "L1", "L2", "R0", "R1", "R2")
LOCAL_AXIS_NAMES = ("+local_x", "-local_x", "+local_y", "-local_y", "+local_z", "-local_z")

_EPSILON = 1e-8

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ContactGeometryConfig:
    origin_bone: str
    direction_bone: str
    tip_bone: str
    support_bone: str
    support_right_axis: str
    support_up_axis: str
    target_up_axis: str
    target_right_axis: str
    l0_min: float
    l0_max: float
    lateral_range: float
    twist_range: float
    tilt_range: float
    radius_scale: float
    invert_l0: bool


@dataclass(frozen=True, slots=True)
class ContactResult:
    axes: dict[str, float]
    status: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Pose:
    name: str
    position: Vector3
    rotation: Quaternion


def calculate_contact_axes(
    reference_raw: Any,
    target_raw: Any,
    config: ContactGeometryConfig,
) -> ContactResult | None:
    bones = _parse_skeleton(reference_raw)
    origin = bones.get(config.origin_bone)
    direction = bones.get(config.direction_bone)
    tip = bones.get(config.tip_bone)
    support = bones.get(config.support_bone)
    target = _parse_pose(target_raw)
    if origin is None or direction is None or tip is None or support is None or target is None:
        return None

    axis = _normalize(_subtract(direction.position, origin.position))
    if axis is None:
        return None
    reference_length = _magnitude(_subtract(tip.position, origin.position))
    if reference_length <= _EPSILON:
        return None

    support_right = _axis_from_rotation(support.rotation, config.support_right_axis)
    support_up = _axis_from_rotation(support.rotation, config.support_up_axis)
    reference_right = _normalize(_project_on_plane(support_right, axis))
    if reference_right is None:
        reference_right = _normalize(_project_on_plane(support_up, axis))
    if reference_right is None:
        return None
    reference_forward = _normalize(_cross(reference_right, axis))
    if reference_forward is None:
        return None

    pair_positions = _parse_pair_positions(target_raw)
    if pair_positions is not None:
        # A bilateral hand/foot contact acts like a sleeve around the
        # Reference.  Either ankle's anatomical rotation would create a large
        # false tilt, so use the Reference axis as target up and the line
        # between both contacts for twist around that axis.
        target_up = axis
        target_right = _normalize(
            _project_on_plane(_subtract(pair_positions[0], pair_positions[1]), axis)
        )
        if target_right is None:
            target_right = reference_right
    else:
        target_up = _normalize(_axis_from_rotation(target.rotation, config.target_up_axis))
        target_right = _normalize(_axis_from_rotation(target.rotation, config.target_right_axis))
    if target_up is None or target_right is None:
        return None

    delta = _subtract(target.position, origin.position)
    axial = _dot(delta, axis)
    closest_axial = _clamp(axial, 0.0, reference_length)
    closest_point = _add(origin.position, _scale(axis, closest_axial))
    radial_vector = _subtract(target.position, closest_point)
    radial_distance = _magnitude(radial_vector)
    lateral_forward = _dot(radial_vector, reference_forward)
    lateral_right = _dot(radial_vector, reference_right)
    reference_radius = reference_length * config.radius_scale
    contact_valid = (
        radial_distance <= reference_radius
        and axial >= -reference_radius
        and axial <= reference_length + reference_radius
    )

    corrected_target_right = _project_on_plane(target_right, axis)
    if _magnitude(corrected_target_right) <= _EPSILON:
        return None
    twist = _signed_angle_degrees(reference_right, corrected_target_right, axis)
    target_up_on_forward_plane = _project_on_plane(target_up, reference_forward)
    target_up_on_right_plane = _project_on_plane(target_up, reference_right)
    roll = -_signed_angle_degrees(axis, target_up_on_forward_plane, reference_forward)
    pitch = _signed_angle_degrees(axis, target_up_on_right_plane, reference_right)

    # Bone pivots for paired feet/hands sit outside the shaft contact surface.
    # Their meaningful stroke coordinate is depth along the live Reference,
    # independent of the single-contact meter calibration.
    l0 = (
        _range01(axial, 0.0, reference_length)
        if pair_positions is not None
        else _range01(axial, config.l0_min, config.l0_max)
    )
    if config.invert_l0:
        l0 = 1.0 - l0
    axes = {
        "L0": l0,
        "L1": _symmetric01(lateral_forward, config.lateral_range),
        "L2": _symmetric01(lateral_right, config.lateral_range),
        "R0": _symmetric01(twist, config.twist_range),
        "R1": _symmetric01(roll, config.tilt_range),
        "R2": _symmetric01(pitch, config.tilt_range),
    }
    return ContactResult(
        axes=axes,
        status={
            "valid": True,
            "contactValid": contact_valid,
            "reason": "ok" if contact_valid else "outside_contact_radius",
            "referenceLength": reference_length,
            "referenceRadius": reference_radius,
            "axialMeters": axial,
            "radialMeters": radial_distance,
            "lateralForwardMeters": lateral_forward,
            "lateralRightMeters": lateral_right,
            "twistDegrees": twist,
            "rollDegrees": roll,
            "pitchDegrees": pitch,
            "targetMode": "bilateral_reference_axis" if pair_positions is not None else "single_bone",
            **axes,
        },
    )


def _parse_skeleton(value: Any) -> dict[str, _Pose]:
    if not isinstance(value, dict) or not isinstance(value.get("bones"), list):
        return {}
    result: dict[str, _Pose] = {}
    for raw_bone in value["bones"]:
        pose = _parse_pose(raw_bone)
        if pose is not None:
            result[pose.name] = pose
    return result


def _parse_pose(value: Any) -> _Pose | None:
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or "").strip()
    position_raw = value.get("pos")
    rotation_raw = value.get("rot")
    if not name or not isinstance(position_raw, list) or len(position_raw) != 3:
        return None
    if not isinstance(rotation_raw, list) or len(rotation_raw) != 4:
        return None

    px = _finite_float(position_raw[0])
    py = _finite_float(position_raw[1])
    pz = _finite_float(position_raw[2])
    rw = _finite_float(rotation_raw[0])
    rx = _finite_float(rotation_raw[1])
    ry = _finite_float(rotation_raw[2])
    rz = _finite_float(rotation_raw[3])
    if px is None or py is None or pz is None or rw is None or rx is None or ry is None or rz is None:
        return None

    normalized_rotation = _normalize_quaternion((rw, rx, ry, rz))
    if normalized_rotation is None:
        return None
    return _Pose(name=name, position=(px, py, pz), rotation=normalized_rotation)


def _parse_pair_positions(value: Any) -> tuple[Vector3, Vector3] | None:
    if not isinstance(value, dict) or value.get("targetMode") != "bilateral_reference_axis":
        return None
    raw_positions = value.get("pairPositions")
    if not isinstance(raw_positions, list) or len(raw_positions) != 2:
        return None
    parsed: list[Vector3] = []
    for raw in raw_positions:
        if not isinstance(raw, list) or len(raw) != 3:
            return None
        values = tuple(_finite_float(item) for item in raw)
        if any(item is None for item in values):
            return None
        parsed.append((float(values[0]), float(values[1]), float(values[2])))
    return parsed[0], parsed[1]


def _axis_from_rotation(rotation: Quaternion, axis_name: str) -> Vector3:
    sign = -1.0 if axis_name.startswith("-") else 1.0
    if axis_name.endswith("local_y"):
        local = (0.0, sign, 0.0)
    elif axis_name.endswith("local_z"):
        local = (0.0, 0.0, sign)
    else:
        local = (sign, 0.0, 0.0)
    return _rotate_vector(rotation, local)


def _signed_angle_degrees(start: Vector3, end: Vector3, axis: Vector3) -> float:
    start_normalized = _normalize(start)
    end_normalized = _normalize(end)
    axis_normalized = _normalize(axis)
    if start_normalized is None or end_normalized is None or axis_normalized is None:
        return 0.0
    sine = _dot(axis_normalized, _cross(start_normalized, end_normalized))
    cosine = _clamp(_dot(start_normalized, end_normalized), -1.0, 1.0)
    return math.degrees(math.atan2(sine, cosine))


def _rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    vector_quaternion = (0.0, vector[0], vector[1], vector[2])
    rotated = _quaternion_multiply(
        _quaternion_multiply(quaternion, vector_quaternion),
        _quaternion_inverse(quaternion),
    )
    return (rotated[1], rotated[2], rotated[3])


def _normalize_quaternion(value: Quaternion) -> Quaternion | None:
    magnitude = math.sqrt(sum(component * component for component in value))
    if magnitude <= _EPSILON:
        return None
    return (
        value[0] / magnitude,
        value[1] / magnitude,
        value[2] / magnitude,
        value[3] / magnitude,
    )


def _quaternion_inverse(value: Quaternion) -> Quaternion:
    return (value[0], -value[1], -value[2], -value[3])


def _quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _scale(value: Vector3, scalar: float) -> Vector3:
    return (value[0] * scalar, value[1] * scalar, value[2] * scalar)


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _magnitude(value: Vector3) -> float:
    return math.sqrt(_dot(value, value))


def _normalize(value: Vector3) -> Vector3 | None:
    magnitude = _magnitude(value)
    return None if magnitude <= _EPSILON else _scale(value, 1.0 / magnitude)


def _project_on_plane(value: Vector3, normal: Vector3) -> Vector3:
    normalized_normal = _normalize(normal)
    if normalized_normal is None:
        return value
    return _subtract(value, _scale(normalized_normal, _dot(value, normalized_normal)))


def _range01(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum + _EPSILON:
        return 0.5
    return _clamp((value - minimum) / (maximum - minimum), 0.0, 1.0)


def _symmetric01(value: float, maximum: float) -> float:
    return _clamp(0.5 + value / (2.0 * maximum), 0.0, 1.0)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


__all__ = [
    "AXIS_NAMES",
    "LOCAL_AXIS_NAMES",
    "ContactGeometryConfig",
    "ContactResult",
    "calculate_contact_axes",
]
