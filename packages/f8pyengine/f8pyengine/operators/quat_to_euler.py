from __future__ import annotations

from f8pysdk.codec import parse_number
from f8pysdk.codec import coerce_bool
import math
from typing import Any

from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    boolean_schema,
    number_schema,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.quat_to_euler"
_ORDERS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")
_EPS = 1e-9

def _coerce_quat(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) != 4:
        return None
    w = parse_number(value[0])
    x = parse_number(value[1])
    y = parse_number(value[2])
    z = parse_number(value[3])
    if w is None or x is None or y is None or z is None:
        return None
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= _EPS:
        return None
    inv = 1.0 / norm
    return (w * inv, x * inv, y * inv, z * inv)

def _normalize_order(value: Any, *, default: str = "ZYX") -> str:
    text = str(value or "").strip().upper()
    if text in _ORDERS:
        return text
    return default

def _clamp_unit(value: float) -> float:
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return float(value)

def _quat_to_matrix(q: tuple[float, float, float, float]) -> tuple[float, float, float, float, float, float, float, float, float]:
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    m11 = 1.0 - 2.0 * (yy + zz)
    m12 = 2.0 * (xy - wz)
    m13 = 2.0 * (xz + wy)
    m21 = 2.0 * (xy + wz)
    m22 = 1.0 - 2.0 * (xx + zz)
    m23 = 2.0 * (yz - wx)
    m31 = 2.0 * (xz - wy)
    m32 = 2.0 * (yz + wx)
    m33 = 1.0 - 2.0 * (xx + yy)
    return (m11, m12, m13, m21, m22, m23, m31, m32, m33)

def _quat_to_euler(q: tuple[float, float, float, float], order: str) -> tuple[float, float, float]:
    m11, m12, m13, m21, m22, m23, m31, m32, m33 = _quat_to_matrix(q)
    if order == "XYZ":
        y = math.asin(_clamp_unit(m13))
        if abs(m13) < 0.9999999:
            x = math.atan2(-m23, m33)
            z = math.atan2(-m12, m11)
        else:
            x = math.atan2(m32, m22)
            z = 0.0
        return (x, y, z)
    if order == "XZY":
        z = math.asin(-_clamp_unit(m12))
        if abs(m12) < 0.9999999:
            x = math.atan2(m32, m22)
            y = math.atan2(m13, m11)
        else:
            x = math.atan2(-m23, m33)
            y = 0.0
        return (x, y, z)
    if order == "YXZ":
        x = math.asin(-_clamp_unit(m23))
        if abs(m23) < 0.9999999:
            y = math.atan2(m13, m33)
            z = math.atan2(m21, m22)
        else:
            y = math.atan2(-m31, m11)
            z = 0.0
        return (x, y, z)
    if order == "YZX":
        z = math.asin(_clamp_unit(m21))
        if abs(m21) < 0.9999999:
            x = math.atan2(-m23, m22)
            y = math.atan2(-m31, m11)
        else:
            x = 0.0
            y = math.atan2(m13, m33)
        return (x, y, z)
    if order == "ZXY":
        x = math.asin(_clamp_unit(m32))
        if abs(m32) < 0.9999999:
            y = math.atan2(-m31, m33)
            z = math.atan2(-m12, m22)
        else:
            y = 0.0
            z = math.atan2(m21, m11)
        return (x, y, z)
    y = math.asin(-_clamp_unit(m31))
    if abs(m31) < 0.9999999:
        x = math.atan2(m32, m33)
        z = math.atan2(m21, m11)
    else:
        x = 0.0
        z = math.atan2(-m12, m22)
    return (x, y, z)

class QuatToEulerRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._order = _normalize_order(self._initial_state.get("order"), default="ZYX")
        self._degrees = coerce_bool(self._initial_state.get("degrees"), default=True)
        self._last_input: tuple[float, float, float, float] | None = None
        self._last_output: list[float] | None = None
        self._last_ctx_id: str | int | None = None
        self._dirty = True

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "")
        if name == "order":
            self._order = _normalize_order(value, default=self._order)
            self._dirty = True
            return
        if name == "degrees":
            self._degrees = coerce_bool(value, default=self._degrees)
            self._dirty = True

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port or "") != "euler":
            return None
        quat = _coerce_quat(await self.pull("quat", ctx_id=ctx_id))
        if quat is None:
            return self._last_output
        if not self._dirty and quat == self._last_input:
            if ctx_id is None or ctx_id == self._last_ctx_id:
                return self._last_output

        x, y, z = _quat_to_euler(quat, self._order)
        if self._degrees:
            factor = 180.0 / math.pi
            output = [x * factor, y * factor, z * factor]
        else:
            output = [x, y, z]
        self._last_input = quat
        self._last_output = output
        self._last_ctx_id = ctx_id
        self._dirty = False
        return output

QuatToEulerRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Quat To Euler",
    description="Converts quaternion [w,x,y,z] to Euler angles with configurable order.",
    tags=["math", "rotation", "quaternion", "euler", "transform"],
    dataInPorts=[
        F8DataPortSpec(
            name="quat",
            description="Input quaternion [w,x,y,z].",
            valueSchema=array_schema(items=number_schema()),
            definitionProtected=False,
        )
    ],
    dataOutPorts=[
        F8DataPortSpec(
            name="euler",
            description="Euler angles [x,y,z] in selected order.",
            valueSchema=array_schema(items=number_schema()),
        )
    ],
    stateFields=[
        F8StateSpec(
            name="order",
            label="Order",
            description="Euler rotation order.",
            valueSchema=string_schema(default="ZYX", enum=list(_ORDERS)),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="degrees",
            label="Degrees",
            description="Output in degrees when true, radians when false.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
    ],
)

def register_operator(registry: Registry) -> Registry:
    registry.register_operator(QuatToEulerRuntimeNode.SPEC, QuatToEulerRuntimeNode, overwrite=True)
    return registry
