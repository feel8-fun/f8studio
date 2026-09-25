from __future__ import annotations

from f8pysdk.codec import coerce_bool
from f8pysdk.codec import parse_number
import math
import time
from dataclasses import dataclass
from typing import Any

from f8pysdk.specs import (
    F8DataPortSpec,
    F8ComplexObjectTypeSchema,
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
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from .envelope import DoubleExponentialMovingAverage, ExponentialMovingAverage
from .smooth_filter import (
    FILTER_CHOICES,
    FILTER_DEMA,
    FILTER_EMA,
    FILTER_NONE,
    FILTER_ONE_EURO,
    OneEuroFilter,
)

OPERATOR_CLASS = "f8.bone_filter"
_EPS = 1e-9

@dataclass(frozen=True)
class _Bone:
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]

def _normalize_filter(value: Any) -> str:
    normalized = str(value or FILTER_NONE).strip().upper()
    if normalized in FILTER_CHOICES:
        return normalized
    return FILTER_NONE

def _clamp_alpha(value: Any, default: float) -> float:
    numeric = parse_number(value)
    if numeric is None:
        return float(default)
    return max(0.0, min(1.0, float(numeric)))

def _quat_dot(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])

def _quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(_quat_dot(q, q))
    if norm <= _EPS:
        return (1.0, 0.0, 0.0, 0.0)
    inv = 1.0 / norm
    return (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)

def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])

def _quat_inverse(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return _quat_conjugate(_quat_normalize(q))

def _quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )

def _quat_rotate_vec(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    qn = _quat_normalize(q)
    p = (0.0, v[0], v[1], v[2])
    rotated = _quat_mul(_quat_mul(qn, p), _quat_conjugate(qn))
    return (float(rotated[1]), float(rotated[2]), float(rotated[3]))

def _vector_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def _vector_norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

def _quat_relative_angle_deg(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dot = abs(_quat_dot(_quat_normalize(a), _quat_normalize(b)))
    clamped = min(1.0, max(-1.0, dot))
    rad = 2.0 * math.acos(clamped)
    return float(rad * 180.0 / math.pi)

def _bone_schema() -> F8ComplexObjectTypeSchema:
    return complex_object_schema(
        properties={
            "pos": array_schema(items=number_schema()),
            "rot": array_schema(items=number_schema()),
        }
    )

class BoneFilterRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

        self._filter_type = _normalize_filter(self._initial_state.get("filter_type") or FILTER_EMA)
        self._ema_alpha = _clamp_alpha(self._initial_state.get("ema_alpha"), 0.4)
        self._dema_alpha = _clamp_alpha(self._initial_state.get("dema_alpha"), 0.4)
        self._one_euro_min_cutoff = max(1e-6, float(parse_number(self._initial_state.get("one_euro_min_cutoff")) or 1.5))
        self._one_euro_beta = max(0.0, float(parse_number(self._initial_state.get("one_euro_beta")) or 0.0))
        self._one_euro_derivative_cutoff = max(
            1e-6, float(parse_number(self._initial_state.get("one_euro_derivative_cutoff")) or 1.0)
        )
        self._one_euro_default_freq = max(
            1e-3, float(parse_number(self._initial_state.get("one_euro_default_freq")) or 90.0)
        )

        self._jump_enabled = coerce_bool(self._initial_state.get("jumpEnabled"), default=True)
        self._jump_pos_threshold = max(0.0, float(parse_number(self._initial_state.get("jumpPosThreshold")) or 0.25))
        self._jump_rot_deg_threshold = max(
            0.0, float(parse_number(self._initial_state.get("jumpRotDegThreshold")) or 35.0)
        )
        self._jump_consecutive_frames = max(1, int(parse_number(self._initial_state.get("jumpConsecutiveFrames")) or 3))
        self._jump_cooldown_frames = max(0, int(parse_number(self._initial_state.get("jumpCooldownFrames")) or 8))

        self._filters: list[Any] = []
        self._filtered_bone: _Bone | None = None
        self._last_raw_bone: _Bone | None = None
        self._last_outputs: dict[str, dict[str, list[float]] | None] = {"filtered": None, "relative": None}
        self._last_ctx_id: str | int | None = None
        self._dirty = True

        self._far_count = 0
        self._jump_count = 0
        self._last_jump_ts_ms: int | None = None
        self._cooldown_remaining = 0

    def _reset_bank(self) -> None:
        self._filters = []
        self._dirty = True

    def _ensure_filter_bank(self) -> None:
        if self._filter_type == FILTER_NONE:
            self._filters = []
            return
        if len(self._filters) == 7:
            return
        if self._filter_type == FILTER_EMA:
            self._filters = [ExponentialMovingAverage(alpha=self._ema_alpha) for _ in range(7)]
            return
        if self._filter_type == FILTER_DEMA:
            self._filters = [DoubleExponentialMovingAverage(alpha=self._dema_alpha) for _ in range(7)]
            return
        if self._filter_type == FILTER_ONE_EURO:
            self._filters = [
                OneEuroFilter(
                    min_cutoff=self._one_euro_min_cutoff,
                    beta=self._one_euro_beta,
                    derivative_cutoff=self._one_euro_derivative_cutoff,
                    default_frequency=self._one_euro_default_freq,
                )
                for _ in range(7)
            ]
            return
        self._filters = []

    def _parse_bone(self, value: Any) -> _Bone | None:
        if not isinstance(value, dict):
            return None
        pos_raw = value.get("pos")
        rot_raw = value.get("rot")
        if not isinstance(pos_raw, (list, tuple)) or not isinstance(rot_raw, (list, tuple)):
            return None
        if len(pos_raw) != 3 or len(rot_raw) != 4:
            return None

        x = parse_number(pos_raw[0])
        y = parse_number(pos_raw[1])
        z = parse_number(pos_raw[2])
        w = parse_number(rot_raw[0])
        qx = parse_number(rot_raw[1])
        qy = parse_number(rot_raw[2])
        qz = parse_number(rot_raw[3])
        if x is None or y is None or z is None or w is None or qx is None or qy is None or qz is None:
            return None

        rot = _quat_normalize((float(w), float(qx), float(qy), float(qz)))
        if _quat_dot(rot, rot) <= _EPS:
            return None
        return _Bone(pos=(float(x), float(y), float(z)), rot=rot)

    @staticmethod
    def _to_output(bone: _Bone | None) -> dict[str, list[float]] | None:
        if bone is None:
            return None
        return {
            "pos": [float(bone.pos[0]), float(bone.pos[1]), float(bone.pos[2])],
            "rot": [float(bone.rot[0]), float(bone.rot[1]), float(bone.rot[2]), float(bone.rot[3])],
        }

    def _filter_bone(self, raw: _Bone, timestamp: float) -> _Bone:
        if self._filter_type == FILTER_NONE:
            return raw

        self._ensure_filter_bank()
        values = [raw.pos[0], raw.pos[1], raw.pos[2], raw.rot[0], raw.rot[1], raw.rot[2], raw.rot[3]]
        out: list[float] = []
        if self._filter_type in {FILTER_EMA, FILTER_DEMA}:
            alpha_value = self._ema_alpha if self._filter_type == FILTER_EMA else self._dema_alpha
            for index, value in enumerate(values):
                filt = self._filters[index]
                out.append(float(filt.update(value, alpha=alpha_value)))
        else:
            for index, value in enumerate(values):
                filt = self._filters[index]
                out.append(float(filt.update(value, timestamp)))
        rot = _quat_normalize((out[3], out[4], out[5], out[6]))
        return _Bone(pos=(out[0], out[1], out[2]), rot=rot)

    def _set_bank_from_sample(self, bone: _Bone) -> None:
        self._reset_bank()
        if self._filter_type == FILTER_NONE:
            return
        self._ensure_filter_bank()
        values = [bone.pos[0], bone.pos[1], bone.pos[2], bone.rot[0], bone.rot[1], bone.rot[2], bone.rot[3]]
        timestamp = time.monotonic()
        for index, value in enumerate(values):
            filt = self._filters[index]
            if self._filter_type in {FILTER_EMA, FILTER_DEMA}:
                alpha_value = self._ema_alpha if self._filter_type == FILTER_EMA else self._dema_alpha
                filt.update(value, alpha=alpha_value)
            else:
                filt.update(value, timestamp)

    def _compute_relative(self, filtered: _Bone, raw: _Bone) -> _Bone:
        qf_inv = _quat_inverse(filtered.rot)
        rel_rot = _quat_normalize(_quat_mul(qf_inv, raw.rot))
        world_delta = _vector_sub(raw.pos, filtered.pos)
        rel_pos = _quat_rotate_vec(qf_inv, world_delta)
        return _Bone(pos=rel_pos, rot=rel_rot)

    def _should_trigger_jump(self, raw: _Bone) -> bool:
        if not self._jump_enabled:
            self._far_count = 0
            return False
        if self._filtered_bone is None:
            self._far_count = 0
            return False
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._far_count = 0
            return False

        pos_diff = _vector_norm(_vector_sub(raw.pos, self._filtered_bone.pos))
        rot_diff_deg = _quat_relative_angle_deg(raw.rot, self._filtered_bone.rot)
        far = pos_diff >= self._jump_pos_threshold or rot_diff_deg >= self._jump_rot_deg_threshold

        if far:
            self._far_count += 1
        else:
            self._far_count = 0

        return self._far_count >= self._jump_consecutive_frames

    def _trigger_jump_reset(self, raw: _Bone) -> None:
        self._filtered_bone = raw
        self._set_bank_from_sample(raw)
        self._far_count = 0
        self._jump_count += 1
        self._last_jump_ts_ms = int(time.time() * 1000)
        self._cooldown_remaining = self._jump_cooldown_frames

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "")
        if name == "filter_type":
            self._filter_type = _normalize_filter(value)
            self._reset_bank()
            return
        if name == "ema_alpha":
            self._ema_alpha = _clamp_alpha(value, 0.4)
            self._reset_bank()
            return
        if name == "dema_alpha":
            self._dema_alpha = _clamp_alpha(value, 0.4)
            self._reset_bank()
            return
        if name == "one_euro_min_cutoff":
            numeric = parse_number(value)
            if numeric is not None:
                self._one_euro_min_cutoff = max(1e-6, float(numeric))
            self._reset_bank()
            return
        if name == "one_euro_beta":
            numeric = parse_number(value)
            if numeric is not None:
                self._one_euro_beta = max(0.0, float(numeric))
            self._reset_bank()
            return
        if name == "one_euro_derivative_cutoff":
            numeric = parse_number(value)
            if numeric is not None:
                self._one_euro_derivative_cutoff = max(1e-6, float(numeric))
            self._reset_bank()
            return
        if name == "one_euro_default_freq":
            numeric = parse_number(value)
            if numeric is not None:
                self._one_euro_default_freq = max(1e-3, float(numeric))
            self._reset_bank()
            return
        if name == "jumpEnabled":
            self._jump_enabled = coerce_bool(value, default=self._jump_enabled)
            self._far_count = 0
            return
        if name == "jumpPosThreshold":
            numeric = parse_number(value)
            if numeric is not None:
                self._jump_pos_threshold = max(0.0, float(numeric))
            return
        if name == "jumpRotDegThreshold":
            numeric = parse_number(value)
            if numeric is not None:
                self._jump_rot_deg_threshold = max(0.0, float(numeric))
            return
        if name == "jumpConsecutiveFrames":
            numeric = parse_number(value)
            if numeric is not None:
                self._jump_consecutive_frames = max(1, int(round(float(numeric))))
            return
        if name == "jumpCooldownFrames":
            numeric = parse_number(value)
            if numeric is not None:
                self._jump_cooldown_frames = max(0, int(round(float(numeric))))

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_name = str(port or "")
        if port_name not in self._last_outputs:
            return None

        raw_input = await self.pull("bone", ctx_id=ctx_id)
        raw_bone = self._parse_bone(raw_input)
        if raw_bone is None:
            return self._last_outputs.get(port_name)

        if self._filtered_bone is not None:
            dot = _quat_dot(raw_bone.rot, self._filtered_bone.rot)
            if dot < 0.0:
                raw_bone = _Bone(
                    pos=raw_bone.pos,
                    rot=(-raw_bone.rot[0], -raw_bone.rot[1], -raw_bone.rot[2], -raw_bone.rot[3]),
                )

        if not self._dirty and self._last_raw_bone == raw_bone:
            if ctx_id is None or ctx_id == self._last_ctx_id:
                return self._last_outputs.get(port_name)

        if self._filtered_bone is None:
            self._filtered_bone = raw_bone
            self._set_bank_from_sample(raw_bone)
        else:
            if self._should_trigger_jump(raw_bone):
                self._trigger_jump_reset(raw_bone)
            else:
                self._filtered_bone = self._filter_bone(raw_bone, time.monotonic())

        assert self._filtered_bone is not None
        relative = self._compute_relative(self._filtered_bone, raw_bone)

        self._last_outputs = {
            "filtered": self._to_output(self._filtered_bone),
            "relative": self._to_output(relative),
        }
        self._last_raw_bone = raw_bone
        self._last_ctx_id = ctx_id
        self._dirty = False
        return self._last_outputs.get(port_name)

BoneFilterRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Bone Filter",
    description="Smooths a single bone pose and outputs filtered + local relative pose.",
    tags=["skeleton", "bone", "filter", "pose", "mocap"],
    dataInPorts=[
        F8DataPortSpec(
            name="bone",
            description="Input bone pose with pos[3] and rot[4] quaternion.",
            valueSchema=_bone_schema(),
        )
    ],
    dataOutPorts=[
        F8DataPortSpec(name="filtered", description="Filtered bone pose.", valueSchema=_bone_schema()),
        F8DataPortSpec(name="relative", description="Relative pose in filtered local space.", valueSchema=_bone_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="filter_type",
            label="Filter",
            description="Filter type.",
            valueSchema=string_schema(default="EMA", enum=list(FILTER_CHOICES)),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="ema_alpha",
            label="EMA Alpha",
            description="EMA smoothing factor (0..1).",
            valueSchema=number_schema(default=0.4, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="dema_alpha",
            label="DEMA Alpha",
            description="DEMA smoothing factor (0..1).",
            valueSchema=number_schema(default=0.4, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="one_euro_min_cutoff",
            label="One Euro Min Cutoff",
            description="Minimum cutoff frequency.",
            valueSchema=number_schema(default=1.5, minimum=0.01, maximum=10.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="one_euro_beta",
            label="One Euro Beta",
            description="Speed coefficient for dynamic cutoff.",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=5.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="one_euro_derivative_cutoff",
            label="One Euro Deriv Cutoff",
            description="Cutoff frequency for derivative filter.",
            valueSchema=number_schema(default=1.0, minimum=0.01, maximum=10.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="one_euro_default_freq",
            label="One Euro Default Freq",
            description="Default sampling frequency (Hz).",
            valueSchema=number_schema(default=90.0, minimum=1.0, maximum=240.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpEnabled",
            label="Jump Enabled",
            description="Enable jump detection and hard reset.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpPosThreshold",
            label="Jump Pos Threshold",
            description="Position distance threshold.",
            valueSchema=number_schema(default=0.25, minimum=0.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpRotDegThreshold",
            label="Jump Rot Deg Threshold",
            description="Rotation distance threshold in degrees.",
            valueSchema=number_schema(default=35.0, minimum=0.0, maximum=180.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpConsecutiveFrames",
            label="Jump Frames",
            description="Consecutive far frames required before reset.",
            valueSchema=number_schema(default=3, minimum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="jumpCooldownFrames",
            label="Jump Cooldown",
            description="Cooldown frames after reset.",
            valueSchema=number_schema(default=8, minimum=0.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)

def register_operator(registry: Registry) -> Registry:
    registry.register_operator(BoneFilterRuntimeNode.SPEC, BoneFilterRuntimeNode, overwrite=True)
    return registry
