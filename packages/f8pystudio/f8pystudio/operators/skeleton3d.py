from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ..skeleton_protocols import skeleton_edges_for_protocol
from ..ui_bus import emit_ui_command
from ._viz_base import StudioVizRuntimeNodeBase, viz_sampling_state_fields

logger = logging.getLogger(__name__)

OPERATOR_CLASS = "f8.skeleton3d"
RENDERER_CLASS = "pystudio_skeleton3d"


@dataclass(frozen=True)
class _NodeViz:
    index: int
    name: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class _PersonViz:
    name: str
    bbox: tuple[float, float, float, float, float, float] | None
    skeleton_protocol: str
    skeleton_edges: list[tuple[int, int]] | None
    nodes: list[_NodeViz]


class PyStudioSkeleton3DRuntimeNode(StudioVizRuntimeNodeBase):
    """
    Studio-side runtime node for 3D skeleton visualization.

    Input:
    - `skeletons`: list[dict] (preferred) or dict (single person)

    Runtime always keeps ingesting and publishing UI payloads. If the detached
    viewer window is closed, render-side will pause drawing but reuse the latest
    payload immediately when the window is re-opened.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
            initial_state=initial_state,
        )
        self._config_loaded = False
        self._refresh_task: asyncio.Task[object] | None = None
        self._scheduled_refresh_ms: int | None = None
        self._last_refresh_ms: int | None = None

        self._dirty = False
        self._latest_people: list[_PersonViz] = []
        self._last_input_ts_ms: int = 0

        self._throttle_ms = 33
        self._world_up = "y"
        self._show_person_boxes = True
        self._show_person_names = False
        self._show_bone_points = True
        self._show_skeleton_lines = True
        self._show_bone_axes = False
        self._show_bone_names = False
        self._max_people = 64
        self._max_bones_per_person = 256
        self._auto_zoom_on_new_people = False
        self._ui_fps_cap = 60
        self._marker_scale = 1.0

        self._warned_signatures: set[str] = set()

    async def close(self) -> None:
        task = self._refresh_task
        self._refresh_task = None
        self._scheduled_refresh_ms = None
        if task is not None:
            try:
                task.cancel()
            except (RuntimeError, TypeError):
                pass
            try:
                await asyncio.gather(task, return_exceptions=True)
            except (RuntimeError, TypeError):
                pass
        emit_ui_command(self.node_id, "skeleton3d.detach", {}, ts_ms=int(time.time() * 1000))

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if str(port or "").strip() != "skeletons":
            return
        await self._ensure_config_loaded()

        now_ms = int(time.time() * 1000)
        self._last_input_ts_ms = int(ts_ms) if ts_ms is not None else now_ms
        self._latest_people = self._extract_people(value)
        self._dirty = True
        await self._schedule_refresh(now_ms=now_ms)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        await self._ensure_config_loaded()
        name = str(field or "").strip()
        if not name:
            return

        updated = False
        if name == "throttleMs":
            self._throttle_ms = self._coerce_int(value, default=self._throttle_ms, minimum=0, maximum=60000)
            updated = True
        elif name == "worldUp":
            self._world_up = self._coerce_world_up(value, default=self._world_up)
            updated = True
        elif name == "showPersonBoxes":
            self._show_person_boxes = self._coerce_bool(value, default=self._show_person_boxes)
            updated = True
        elif name == "showPersonNames":
            self._show_person_names = self._coerce_bool(value, default=self._show_person_names)
            updated = True
        elif name == "showBonePoints":
            self._show_bone_points = self._coerce_bool(value, default=self._show_bone_points)
            updated = True
        elif name == "showSkeletonLines":
            self._show_skeleton_lines = self._coerce_bool(value, default=self._show_skeleton_lines)
            updated = True
        elif name == "showBoneAxes":
            self._show_bone_axes = self._coerce_bool(value, default=self._show_bone_axes)
            updated = True
        elif name == "showBoneNames":
            self._show_bone_names = self._coerce_bool(value, default=self._show_bone_names)
            updated = True
        elif name == "maxPeople":
            self._max_people = self._coerce_int(value, default=self._max_people, minimum=1, maximum=4096)
            updated = True
        elif name == "maxBonesPerPerson":
            self._max_bones_per_person = self._coerce_int(value, default=self._max_bones_per_person, minimum=1, maximum=8192)
            updated = True
        elif name == "autoZoomOnNewPeople":
            self._auto_zoom_on_new_people = self._coerce_bool(value, default=self._auto_zoom_on_new_people)
            updated = True
        elif name == "uiFpsCap":
            self._ui_fps_cap = self._coerce_int(value, default=self._ui_fps_cap, minimum=1, maximum=120)
            updated = True
        elif name == "markerScale":
            self._marker_scale = self._coerce_float(value, default=self._marker_scale, minimum=0.1, maximum=20.0)
            updated = True

        if not updated:
            return
        self._dirty = True
        now_ms = int(time.time() * 1000)
        await self._schedule_refresh(now_ms=now_ms)

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._throttle_ms = self._coerce_int(
            await self._get_state_or_initial("throttleMs", 33), default=33, minimum=0, maximum=60000
        )
        self._world_up = self._coerce_world_up(await self._get_state_or_initial("worldUp", "y"), default="y")
        self._show_person_boxes = self._coerce_bool(
            await self._get_state_or_initial("showPersonBoxes", True), default=True
        )
        self._show_person_names = self._coerce_bool(
            await self._get_state_or_initial("showPersonNames", False), default=False
        )
        self._show_bone_points = self._coerce_bool(
            await self._get_state_or_initial("showBonePoints", True), default=True
        )
        self._show_skeleton_lines = self._coerce_bool(
            await self._get_state_or_initial("showSkeletonLines", True), default=True
        )
        self._show_bone_axes = self._coerce_bool(await self._get_state_or_initial("showBoneAxes", False), default=False)
        self._show_bone_names = self._coerce_bool(await self._get_state_or_initial("showBoneNames", False), default=False)
        self._max_people = self._coerce_int(
            await self._get_state_or_initial("maxPeople", 64), default=64, minimum=1, maximum=4096
        )
        self._max_bones_per_person = self._coerce_int(
            await self._get_state_or_initial("maxBonesPerPerson", 256), default=256, minimum=1, maximum=8192
        )
        self._auto_zoom_on_new_people = self._coerce_bool(
            await self._get_state_or_initial("autoZoomOnNewPeople", False), default=False
        )
        self._ui_fps_cap = self._coerce_int(
            await self._get_state_or_initial("uiFpsCap", 60), default=60, minimum=1, maximum=120
        )
        self._marker_scale = self._coerce_float(
            await self._get_state_or_initial("markerScale", 1.0), default=1.0, minimum=0.1, maximum=20.0
        )
        self._config_loaded = True

    async def _get_state_or_initial(self, name: str, default: Any) -> Any:
        value: Any = None
        try:
            value = await self.get_state_value(name)
        except Exception:
            value = None
        if value is not None:
            return value
        return self._initial_state.get(name, default)

    async def _schedule_refresh(self, *, now_ms: int) -> None:
        throttle_ms = max(0, int(self._throttle_ms))
        last_refresh = int(self._last_refresh_ms or 0)
        if throttle_ms <= 0 or last_refresh <= 0:
            await self._flush(now_ms=now_ms)
            return

        target_ms = last_refresh + throttle_ms
        if int(now_ms) >= int(target_ms):
            await self._flush(now_ms=now_ms)
            return

        if self._refresh_task is not None and not self._refresh_task.done():
            return

        delay_ms = max(0, int(target_ms) - int(now_ms))
        self._scheduled_refresh_ms = int(target_ms)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._refresh_task = loop.create_task(
            self._flush_after(delay_ms=delay_ms), name=f"pystudio:skeleton3d:flush:{self.node_id}"
        )

    async def _flush_after(self, *, delay_ms: int) -> None:
        try:
            await asyncio.sleep(float(max(0, int(delay_ms))) / 1000.0)
        except (RuntimeError, TypeError, ValueError):
            return
        await self._flush(now_ms=int(time.time() * 1000))

    async def _flush(self, *, now_ms: int) -> None:
        self._scheduled_refresh_ms = None
        if not self._dirty and not self._latest_people:
            self._last_refresh_ms = int(now_ms)
            return

        payload = self._build_payload(now_ms=now_ms, people=self._latest_people)
        emit_ui_command(self.node_id, "skeleton3d.set", payload, ts_ms=int(now_ms))
        self._last_refresh_ms = int(now_ms)
        self._dirty = False

    def _build_payload(self, *, now_ms: int, people: list[_PersonViz]) -> dict[str, Any]:
        people_json: list[dict[str, Any]] = []
        for person in list(people)[: self._max_people]:
            nodes_json: list[dict[str, Any]] = []
            for node in list(person.nodes)[: self._max_bones_per_person]:
                item: dict[str, Any] = {
                    "index": int(node.index),
                    "name": str(node.name),
                    "pos": [float(node.pos[0]), float(node.pos[1]), float(node.pos[2])],
                }
                if node.rot is not None:
                    item["rot"] = [float(node.rot[0]), float(node.rot[1]), float(node.rot[2]), float(node.rot[3])]
                nodes_json.append(item)

            bbox = None
            if person.bbox is not None:
                bbox = [
                    float(person.bbox[0]),
                    float(person.bbox[1]),
                    float(person.bbox[2]),
                    float(person.bbox[3]),
                    float(person.bbox[4]),
                    float(person.bbox[5]),
                ]
            people_json.append(
                {
                    "name": str(person.name),
                    "bbox": bbox,
                    "skeletonProtocol": str(person.skeleton_protocol),
                    "skeletonEdges": (
                        [[int(a), int(b)] for a, b in person.skeleton_edges]
                        if person.skeleton_edges is not None
                        else None
                    ),
                    "nodes": nodes_json,
                }
            )

        return {
            "tsMs": int(now_ms if now_ms > 0 else self._last_input_ts_ms or int(time.time() * 1000)),
            "worldUp": str(self._world_up),
            "uiFpsCap": int(self._ui_fps_cap),
            "renderFlags": {
                "showPersonBoxes": bool(self._show_person_boxes),
                "showPersonNames": bool(self._show_person_names),
                "showBonePoints": bool(self._show_bone_points),
                "showSkeletonLines": bool(self._show_skeleton_lines),
                "showBoneAxes": bool(self._show_bone_axes),
                "showBoneNames": bool(self._show_bone_names),
                "autoZoomOnNewPeople": bool(self._auto_zoom_on_new_people),
                "markerScale": float(self._marker_scale),
            },
            "limits": {
                "maxPeople": int(self._max_people),
                "maxBonesPerPerson": int(self._max_bones_per_person),
            },
            "people": people_json,
        }

    def _extract_people(self, value: Any) -> list[_PersonViz]:
        raw_people: list[dict[str, Any]] = []
        if isinstance(value, dict):
            raw_people = [value]
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    raw_people.append(item)
        else:
            self._log_bad_input_once(value)
            return []

        out: list[_PersonViz] = []
        for index, payload in enumerate(raw_people):
            person = self._extract_person(payload=payload, index=index)
            if person is not None:
                out.append(person)
            if len(out) >= self._max_people:
                break
        return out

    def _extract_person(self, *, payload: dict[str, Any], index: int) -> _PersonViz | None:
        person_name = self._extract_person_name(payload=payload, index=index)
        skeleton_protocol = self._extract_protocol(payload)
        skeleton_edges = skeleton_edges_for_protocol(skeleton_protocol)
        bones_any = payload.get("bones")
        if not isinstance(bones_any, list):
            return _PersonViz(
                name=person_name,
                bbox=None,
                skeleton_protocol=skeleton_protocol,
                skeleton_edges=skeleton_edges,
                nodes=[],
            )

        nodes: list[_NodeViz] = []
        for bone_index, raw_bone in enumerate(bones_any):
            if not isinstance(raw_bone, dict):
                continue
            node = self._extract_node(raw_bone=raw_bone, index=bone_index)
            if node is None:
                continue
            nodes.append(node)
            if len(nodes) >= self._max_bones_per_person:
                break
        return _PersonViz(
            name=person_name,
            bbox=self._compute_bbox(nodes),
            skeleton_protocol=skeleton_protocol,
            skeleton_edges=skeleton_edges,
            nodes=nodes,
        )

    @staticmethod
    def _extract_protocol(payload: dict[str, Any]) -> str:
        value = payload.get("skeletonProtocol")
        text = str(value or "").strip().lower()
        if not text:
            return "none"
        return text

    @staticmethod
    def _extract_person_name(*, payload: dict[str, Any], index: int) -> str:
        for key in ("modelName", "name", "character", "actor"):
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return f"Person_{index + 1}"

    @staticmethod
    def _extract_node(*, raw_bone: dict[str, Any], index: int) -> _NodeViz | None:
        pos = PyStudioSkeleton3DRuntimeNode._coerce_vec3(raw_bone.get("pos"))
        if pos is None:
            return None
        name_any = raw_bone.get("name")
        node_name = str(name_any).strip() if name_any is not None else ""
        if not node_name:
            node_name = f"bone_{index}"
        rot = PyStudioSkeleton3DRuntimeNode._coerce_quat(raw_bone.get("rot"))
        return _NodeViz(index=int(index), name=node_name, pos=pos, rot=rot)

    @staticmethod
    def _compute_bbox(nodes: list[_NodeViz]) -> tuple[float, float, float, float, float, float] | None:
        if not nodes:
            return None
        min_x = float(nodes[0].pos[0])
        min_y = float(nodes[0].pos[1])
        min_z = float(nodes[0].pos[2])
        max_x = float(nodes[0].pos[0])
        max_y = float(nodes[0].pos[1])
        max_z = float(nodes[0].pos[2])
        for node in nodes[1:]:
            x = float(node.pos[0])
            y = float(node.pos[1])
            z = float(node.pos[2])
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if z < min_z:
                min_z = z
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
            if z > max_z:
                max_z = z
        return (min_x, min_y, min_z, max_x, max_y, max_z)

    @staticmethod
    def _coerce_vec3(value: Any) -> tuple[float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            x = float(value[0])
            y = float(value[1])
            z = float(value[2])
        except (TypeError, ValueError):
            return None
        return (x, y, z)

    @staticmethod
    def _coerce_quat(value: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            qw = float(value[0])
            qx = float(value[1])
            qy = float(value[2])
            qz = float(value[3])
        except (TypeError, ValueError):
            return None
        return (qw, qx, qy, qz)

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        return bool(default)

    @staticmethod
    def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            out = int(value) if value is not None else int(default)
        except (TypeError, ValueError):
            out = int(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

    @staticmethod
    def _coerce_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
        try:
            out = float(value) if value is not None else float(default)
        except (TypeError, ValueError):
            out = float(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

    @staticmethod
    def _coerce_world_up(value: Any, *, default: str) -> str:
        text = str(value or "").strip().lower()
        if text in ("y", "z"):
            return text
        return str(default or "y")

    def _log_bad_input_once(self, value: Any) -> None:
        sig = f"{type(value).__name__}"
        if sig in self._warned_signatures:
            return
        self._warned_signatures.add(sig)
        logger.warning("skeleton3d ignored invalid input type=%s nodeId=%s", type(value).__name__, self.node_id)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PyStudioSkeleton3DRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=SERVICE_CLASS,
            operatorClass=OPERATOR_CLASS,
            version="0.0.1",
            label="Skeleton3D",
            description="3D viewer for multi-person skeleton streams (Studio UI-only).",
            tags=["viz", "3d", "skeleton", "ui"],
            dataInPorts=[
                F8DataPortSpec(
                    name="skeletons",
                    description="List of skeleton payloads (or single skeleton dict).",
                    valueSchema=any_schema(),
                ),
            ],
            dataOutPorts=[],
            rendererClass=RENDERER_CLASS,
            stateFields=[
                F8StateSpec(
                    name="throttleMs",
                    label="Push Throttle (ms)",
                    description="Runtime push interval to UI command channel.",
                    valueSchema=integer_schema(default=33, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="worldUp",
                    label="World Up",
                    description="World up axis for viewer transform.",
                    valueSchema=string_schema(default="y", enum=["y", "z"]),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="showPersonBoxes",
                    label="Show Person Boxes",
                    description="Display per-person 3D bounding boxes.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="showPersonNames",
                    label="Show Person Names",
                    description="Display per-person labels.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="showBonePoints",
                    label="Show Bone Points",
                    description="Display bone node points.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="showSkeletonLines",
                    label="Show Skeleton Lines",
                    description="Display protocol-based links for known skeleton protocols.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="showBoneAxes",
                    label="Show Bone Axes",
                    description="Display RGB axes per bone node.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="showBoneNames",
                    label="Show Bone Names",
                    description="Display labels per bone node.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="maxPeople",
                    label="Max People",
                    description="Maximum people rendered from each frame.",
                    valueSchema=integer_schema(default=64, minimum=1, maximum=4096),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="maxBonesPerPerson",
                    label="Max Bones Per Person",
                    description="Maximum bones rendered for each person.",
                    valueSchema=integer_schema(default=256, minimum=1, maximum=8192),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="autoZoomOnNewPeople",
                    label="Auto Zoom On New People",
                    description="Auto fit view when the person set changes.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="uiFpsCap",
                    label="UI FPS Cap",
                    description="Front-end render FPS cap.",
                    valueSchema=integer_schema(default=60, minimum=1, maximum=120),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="markerScale",
                    label="Marker Scale",
                    description="Global scale for bone point size and bone axis size.",
                    valueSchema=number_schema(default=1.0, minimum=0.0),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                *viz_sampling_state_fields(show_on_node=False),
            ],
        ),
        overwrite=True,
    )
    return reg
