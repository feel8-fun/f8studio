from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, cast

import msgspec

from f8pysdk.codec import coerce_flag, coerce_float, coerce_int
from f8pysdk.f8_naming import ensure_token
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    editable_collection_edit_policy,
    integer_schema,
    number_schema,
    string_schema,
)

from ..identifiers import SERVICE_CLASS
from ..visualization import skeleton_edges_for_nodes
from ._viz_base import StudioVizRuntimeNodeBase, viz_sampling_state_fields
from .categories import PALETTE_CATEGORY_VIZ

OPERATOR_CLASS = "f8.viz.three_d"
RENDERER_CLASS = "viz_three_d"
_LARGE_NODE_THRESHOLD = 192
_BOX_SUPPRESSION_THRESHOLD = 384

logger = logging.getLogger(__name__)


class BoneInput(msgspec.Struct):
    name: str = ""
    pos: list[float] = msgspec.field(default_factory=list)
    rot: list[float] | None = None


class SkeletonInput(msgspec.Struct, rename="camel"):
    model_name: str = ""
    name: str = ""
    character: str = ""
    actor: str = ""
    skeleton_protocol: str = "none"
    bones: list[object] = msgspec.field(default_factory=list)


@dataclass(frozen=True)
class SceneNode:
    index: int
    name: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class ScenePerson:
    name: str
    bbox: tuple[float, float, float, float, float, float] | None
    skeleton_protocol: str
    skeleton_edges: list[tuple[int, int]] | None
    nodes: list[SceneNode]


class VizThreeDRuntimeNode(StudioVizRuntimeNodeBase):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[port.name for port in (node.dataInPorts or [])],
            data_out_ports=[],
            state_fields=[field.name for field in (node.stateFields or [])],
            initial_state=initial_state,
        )
        state = self._initial_state
        self._throttle_ms = coerce_int(state.get("throttleMs"), default=33, minimum=0, maximum=60_000)
        self._world_up = self._coerce_world_up(state.get("worldUp"), default="+y")
        self._show_person_boxes = coerce_flag(state.get("showPersonBoxes"), default=True)
        self._show_person_names = coerce_flag(state.get("showPersonNames"), default=False)
        self._show_bone_points = coerce_flag(state.get("showBonePoints"), default=True)
        self._show_skeleton_lines = coerce_flag(state.get("showSkeletonLines"), default=True)
        self._show_bone_axes = coerce_flag(state.get("showBoneAxes"), default=False)
        self._show_bone_names = coerce_flag(state.get("showBoneNames"), default=False)
        self._max_people = coerce_int(state.get("maxPeople"), default=64, minimum=1, maximum=4_096)
        self._max_bones = coerce_int(state.get("maxBonesPerPerson"), default=256, minimum=1, maximum=8_192)
        self._auto_zoom = coerce_flag(state.get("autoZoomOnNewPeople"), default=False)
        self._ui_fps_cap = coerce_int(state.get("uiFpsCap"), default=60, minimum=1, maximum=120)
        self._marker_scale = coerce_float(state.get("markerScale"), default=1.0, minimum=0.1, maximum=100_000.0)
        self._people_by_port: dict[str, list[ScenePerson]] = {}
        self._last_input_ts_ms = 0
        self._last_refresh_ms = 0
        self._dirty = False
        self._refresh_task: asyncio.Task[object] | None = None
        self._warned_inputs: set[str] = set()

    async def close(self) -> None:
        task = self._refresh_task
        self._refresh_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.presentation.emit(self.node_id, "viz.three_d.detach", {}, ts_ms=int(time.time() * 1_000))

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        port_name = port.strip()
        if not port_name:
            return
        people = self._parse_people(port_name, value)
        if people is None:
            signature = f"{port_name}:{type(value).__name__}"
            if signature not in self._warned_inputs:
                self._warned_inputs.add(signature)
                logger.warning("3D visualization ignored invalid input node_id=%s port=%s type=%s", self.node_id, port_name, type(value).__name__)
            return
        self._people_by_port[port_name] = people
        self._last_input_ts_ms = int(ts_ms) if ts_ms is not None else int(time.time() * 1_000)
        self._dirty = True
        await self._schedule_refresh(int(time.time() * 1_000))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        if field == "throttleMs":
            self._throttle_ms = coerce_int(value, default=self._throttle_ms, minimum=0, maximum=60_000)
        elif field == "worldUp":
            self._world_up = self._coerce_world_up(value, default=self._world_up)
            self.presentation.emit(self.node_id, "viz.three_d.world_up", {"worldUp": self._world_up}, ts_ms=ts_ms)
        elif field == "showPersonBoxes":
            self._show_person_boxes = coerce_flag(value, default=self._show_person_boxes)
        elif field == "showPersonNames":
            self._show_person_names = coerce_flag(value, default=self._show_person_names)
        elif field == "showBonePoints":
            self._show_bone_points = coerce_flag(value, default=self._show_bone_points)
        elif field == "showSkeletonLines":
            self._show_skeleton_lines = coerce_flag(value, default=self._show_skeleton_lines)
        elif field == "showBoneAxes":
            self._show_bone_axes = coerce_flag(value, default=self._show_bone_axes)
        elif field == "showBoneNames":
            self._show_bone_names = coerce_flag(value, default=self._show_bone_names)
        elif field == "maxPeople":
            self._max_people = coerce_int(value, default=self._max_people, minimum=1, maximum=4_096)
        elif field == "maxBonesPerPerson":
            self._max_bones = coerce_int(value, default=self._max_bones, minimum=1, maximum=8_192)
        elif field == "autoZoomOnNewPeople":
            self._auto_zoom = coerce_flag(value, default=self._auto_zoom)
        elif field == "uiFpsCap":
            self._ui_fps_cap = coerce_int(value, default=self._ui_fps_cap, minimum=1, maximum=120)
        elif field == "markerScale":
            self._marker_scale = coerce_float(value, default=self._marker_scale, minimum=0.1, maximum=100_000.0)
        else:
            return
        self._dirty = True
        await self._schedule_refresh(int(ts_ms) if ts_ms is not None else int(time.time() * 1_000))

    def _parse_people(self, port: str, value: object) -> list[ScenePerson] | None:
        raw_people: list[object] = cast(list[object], value) if isinstance(value, list) else [value]
        output: list[ScenePerson] = []
        for index, raw_person in enumerate(raw_people):
            person = self._parse_person(port, raw_person, index)
            if person is not None:
                output.append(person)
            if len(output) >= self._max_people:
                break
        return output or None

    def _parse_person(self, port: str, value: object, index: int) -> ScenePerson | None:
        try:
            skeleton = msgspec.convert(value, type=SkeletonInput, strict=False)
        except (msgspec.ValidationError, TypeError):
            skeleton = None
        if skeleton is None or not skeleton.bones:
            try:
                bone = msgspec.convert(value, type=BoneInput, strict=False)
            except (msgspec.ValidationError, TypeError):
                return None
            if len(bone.pos) < 3 or bone.rot is None or len(bone.rot) < 4:
                return None
            skeleton = SkeletonInput(model_name=port, skeleton_protocol="none", bones=[bone])

        nodes: list[SceneNode] = []
        for bone_index, raw_bone in enumerate(skeleton.bones):
            try:
                bone = raw_bone if isinstance(raw_bone, BoneInput) else msgspec.convert(raw_bone, type=BoneInput, strict=False)
            except (msgspec.ValidationError, TypeError):
                continue
            if len(bone.pos) < 3:
                continue
            rotation = None
            if bone.rot is not None and len(bone.rot) >= 4:
                rotation = (bone.rot[0], bone.rot[1], bone.rot[2], bone.rot[3])
            nodes.append(
                SceneNode(
                    index=bone_index,
                    name=bone.name or f"bone_{bone_index}",
                    pos=(bone.pos[0], bone.pos[1], bone.pos[2]),
                    rot=rotation,
                )
            )
            if len(nodes) >= self._max_bones:
                break
        if not nodes:
            return None
        base_name = skeleton.model_name or skeleton.name or skeleton.character or skeleton.actor or f"Person_{index + 1}"
        protocol = (skeleton.skeleton_protocol or "none").strip().lower()
        return ScenePerson(
            name=f"{port}:{base_name}",
            bbox=self._bbox(nodes),
            skeleton_protocol=protocol,
            skeleton_edges=skeleton_edges_for_nodes(protocol, [node.name for node in nodes]),
            nodes=nodes,
        )

    @staticmethod
    def _bbox(nodes: list[SceneNode]) -> tuple[float, float, float, float, float, float]:
        xs = [node.pos[0] for node in nodes]
        ys = [node.pos[1] for node in nodes]
        zs = [node.pos[2] for node in nodes]
        return min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)

    def _aggregate_people(self) -> list[ScenePerson]:
        preferred = [port for port in self.data_in_ports if port]
        remaining = [port for port in self._people_by_port if port not in set(preferred)]
        output: list[ScenePerson] = []
        for port in [*preferred, *remaining]:
            output.extend(self._people_by_port.get(port, []))
            if len(output) >= self._max_people:
                return output[: self._max_people]
        return output

    async def _schedule_refresh(self, now_ms: int) -> None:
        target_ms = self._last_refresh_ms + self._throttle_ms
        if self._throttle_ms <= 0 or self._last_refresh_ms <= 0 or now_ms >= target_ms:
            await self._flush(now_ms)
            return
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._flush_after(target_ms - now_ms), name=f"web-studio:three-d:{self.node_id}")

    async def _flush_after(self, delay_ms: int) -> None:
        await asyncio.sleep(max(0, delay_ms) / 1_000)
        await self._flush(int(time.time() * 1_000))

    async def _flush(self, now_ms: int) -> None:
        people = self._aggregate_people()
        encoded_people: list[dict[str, object]] = []
        total_nodes = 0
        for person in people:
            nodes = [
                {
                    "index": node.index,
                    "name": node.name,
                    "pos": list(node.pos),
                    "rot": list(node.rot) if node.rot is not None else None,
                }
                for node in person.nodes
            ]
            total_nodes += len(nodes)
            encoded_people.append(
                {
                    "name": person.name,
                    "bbox": list(person.bbox) if person.bbox is not None else None,
                    "skeletonProtocol": person.skeleton_protocol,
                    "skeletonEdges": [list(edge) for edge in person.skeleton_edges] if person.skeleton_edges is not None else None,
                    "nodes": nodes,
                }
            )
        large = total_nodes >= _LARGE_NODE_THRESHOLD
        label_budget = 256 if large else (32 if total_nodes >= 64 else None)
        self.presentation.emit(
            self.node_id,
            "viz.three_d.set",
            {
                "tsMs": now_ms or self._last_input_ts_ms,
                "worldUp": self._world_up,
                "uiFpsCap": self._ui_fps_cap,
                "renderFlags": {
                    "showPersonBoxes": self._show_person_boxes,
                    "showPersonNames": self._show_person_names,
                    "showBonePoints": self._show_bone_points,
                    "showSkeletonLines": self._show_skeleton_lines,
                    "showBoneAxes": self._show_bone_axes,
                    "showBoneNames": self._show_bone_names,
                    "autoZoomOnNewPeople": self._auto_zoom,
                    "markerScale": self._marker_scale,
                },
                "limits": {"maxPeople": self._max_people, "maxBonesPerPerson": self._max_bones},
                "performanceHints": {
                    "totalNodes": total_nodes,
                    "largeSkeletonMode": large,
                    "suppressBoneAxes": False,
                    "suppressBoneNames": False,
                    "suppressAxisTree": False,
                    "suppressPersonBoxes": total_nodes >= _BOX_SUPPRESSION_THRESHOLD,
                    "maxVisibleBoneLabels": label_budget,
                    "recommendedFpsCap": min(self._ui_fps_cap, 30) if large else self._ui_fps_cap,
                },
                "people": encoded_people,
            },
            ts_ms=now_ms,
        )
        self._last_refresh_ms = now_ms
        self._dirty = False

    @staticmethod
    def _coerce_world_up(value: object, *, default: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            return normalized
        if normalized in {"x", "y", "z"}:
            return f"+{normalized}"
        return default if default in {"+x", "-x", "+y", "-y", "+z", "-z"} else "+y"


VizThreeDRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=PALETTE_CATEGORY_VIZ,
    operatorClass=OPERATOR_CLASS,
    version="0.1.0",
    label="3D Viz",
    description="Publish normalized multi-person skeleton scenes for the Web Studio Three.js renderer.",
    tags=["viz", "3d", "skeleton", "web"],
    dataInPorts=[F8DataPortSpec(name="skeletons", description="Skeleton, skeleton list, or single bone.", valueSchema=any_schema())],
    dataOutPorts=[],
    editPolicy=F8SpecEditPolicy(dataInPorts=editable_collection_edit_policy()),
    rendererClass=RENDERER_CLASS,
    stateFields=[
        F8StateSpec(name="throttleMs", valueSchema=integer_schema(default=33, minimum=0, maximum=60_000), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="worldUp", valueSchema=string_schema(default="+y", enum=["+x", "-x", "+y", "-y", "+z", "-z"]), access=F8StateAccess.rw, required=True, showOnNode=True),
        F8StateSpec(name="showPersonBoxes", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="showPersonNames", valueSchema=boolean_schema(default=False), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="showBonePoints", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="showSkeletonLines", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="showBoneAxes", valueSchema=boolean_schema(default=False), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="showBoneNames", valueSchema=boolean_schema(default=False), access=F8StateAccess.rw, required=True, showOnNode=True),
        F8StateSpec(name="maxPeople", valueSchema=integer_schema(default=64, minimum=1, maximum=4_096), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="maxBonesPerPerson", valueSchema=integer_schema(default=256, minimum=1, maximum=8_192), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="autoZoomOnNewPeople", valueSchema=boolean_schema(default=False), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="uiFpsCap", valueSchema=integer_schema(default=60, minimum=1, maximum=120), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="markerScale", valueSchema=number_schema(default=1.0, minimum=0.1, maximum=100_000.0), access=F8StateAccess.rw, required=True),
        *viz_sampling_state_fields(),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(VizThreeDRuntimeNode.SPEC, VizThreeDRuntimeNode, overwrite=True)
    return registry


__all__ = ["VizThreeDRuntimeNode", "register_operator"]
