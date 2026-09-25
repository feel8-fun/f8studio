from __future__ import annotations

import math
from typing import Any

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8DataPortSpec,
    F8ComplexObjectTypeSchema,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    complex_object_schema,
    number_schema,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.bone_selector"


def _bone_schema() -> F8ComplexObjectTypeSchema:
    return complex_object_schema(
        properties={
            "name": string_schema(),
            "pos": array_schema(items=number_schema()),
            "rot": array_schema(items=number_schema()),
        }
    )


def _skeleton_schema() -> F8ComplexObjectTypeSchema:
    return complex_object_schema(
        properties={
            "bones": array_schema(items=_bone_schema()),
        }
    )


def _coerce_finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return float(numeric)


class BoneSelectorRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._target = str(self._initial_state.get("target", "") or "")
        self._available_bones: list[str] = []
        self._available_bones_synced = False
        self._target_synced = False

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if str(field or "").strip() != "target":
            return
        self._target = str(value or "").strip()
        self._target_synced = True

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port or "").strip() != "bone":
            return None

        skeleton_value = await self.pull("skeleton", ctx_id=ctx_id)
        bone_names, bones_by_name = self._parse_skeleton(skeleton_value)
        await self._sync_state(bone_names)

        target_name = str(self._target or "").strip()
        selected = bones_by_name.get(target_name)
        if selected is None:
            return None
        return {
            "name": str(selected["name"]),
            "pos": [float(selected["pos"][0]), float(selected["pos"][1]), float(selected["pos"][2])],
            "rot": [
                float(selected["rot"][0]),
                float(selected["rot"][1]),
                float(selected["rot"][2]),
                float(selected["rot"][3]),
            ],
        }

    async def _sync_state(self, bone_names: list[str]) -> None:
        if (not self._available_bones_synced) or bone_names != self._available_bones:
            self._available_bones = list(bone_names)
            await self.set_state("availableBones", list(self._available_bones))
            self._available_bones_synced = True

        next_target = self._normalize_target(bone_names, self._target)
        if (not self._target_synced) or next_target != self._target:
            self._target = next_target
            await self.set_state("target", self._target)
            self._target_synced = True

    @staticmethod
    def _normalize_target(bone_names: list[str], current_target: str) -> str:
        if not bone_names:
            return ""
        if current_target in bone_names:
            return current_target
        return bone_names[0]

    def _parse_skeleton(self, skeleton: Any) -> tuple[list[str], dict[str, dict[str, Any]]]:
        if not isinstance(skeleton, dict):
            return ([], {})
        bones_raw = skeleton.get("bones")
        if not isinstance(bones_raw, list):
            return ([], {})

        bone_names: list[str] = []
        bones_by_name: dict[str, dict[str, Any]] = {}
        for item in bones_raw:
            parsed = self._parse_bone(item)
            if parsed is None:
                continue
            name = str(parsed["name"])
            if name in bones_by_name:
                continue
            bones_by_name[name] = parsed
            bone_names.append(name)
        return (bone_names, bones_by_name)

    def _parse_bone(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        name_raw = value.get("name")
        pos_raw = value.get("pos")
        rot_raw = value.get("rot")

        name = str(name_raw or "").strip()
        if not name:
            return None
        if not isinstance(pos_raw, list) or len(pos_raw) != 3:
            return None
        if not isinstance(rot_raw, list) or len(rot_raw) != 4:
            return None

        x = _coerce_finite_float(pos_raw[0])
        y = _coerce_finite_float(pos_raw[1])
        z = _coerce_finite_float(pos_raw[2])
        w = _coerce_finite_float(rot_raw[0])
        qx = _coerce_finite_float(rot_raw[1])
        qy = _coerce_finite_float(rot_raw[2])
        qz = _coerce_finite_float(rot_raw[3])
        if x is None or y is None or z is None or w is None or qx is None or qy is None or qz is None:
            return None

        return {
            "name": name,
            "pos": [x, y, z],
            "rot": [w, qx, qy, qz],
        }


BoneSelectorRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Bone Selector",
    description="Selects one bone from a skeleton by `target` and outputs `{name,pos,rot}`.",
    tags=["skeleton", "bone", "select", "mocap"],
    execInPorts=[],
    execOutPorts=[],
    dataInPorts=[
        F8DataPortSpec(
            name="skeleton",
            description="Single skeleton payload (e.g. skeleton_decoder.selectedSkeleton or vmc_decoder.selectedSkeleton).",
            valueSchema=_skeleton_schema(),
        )
    ],
    dataOutPorts=[
        F8DataPortSpec(
            name="bone",
            description="Selected bone payload `{name,pos,rot}` or None.",
            valueSchema=_bone_schema(),
        )
    ],
    stateFields=[
        F8StateSpec(
            name="target",
            label="Target Bone",
            description="Bone name to select.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select, optionsFromState="availableBones"),
            showOnNode=True,
        ),
        F8StateSpec(
            name="availableBones",
            label="Available Bones",
            description="Read-only list of available bone names from current skeleton input.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(BoneSelectorRuntimeNode.SPEC, BoneSelectorRuntimeNode, overwrite=True)
    return registry
