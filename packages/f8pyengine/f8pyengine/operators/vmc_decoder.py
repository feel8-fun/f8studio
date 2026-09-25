from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import asyncio
import logging
import math
import struct
from dataclasses import dataclass
from typing import Any

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    array_schema,
    complex_object_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.time_utils import now_ms

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.vmc_decoder"
_VMC_MODEL_NAME = "VMC"
_VMC_SKELETON_PROTOCOL = "unity_humanoid"
_VMC_BONE_POS_ADDR = "/VMC/Ext/Bone/Pos"
_VMC_ROOT_POS_ADDR = "/VMC/Ext/Root/Pos"
_VMC_OK_ADDR = "/VMC/Ext/OK"
logger = logging.getLogger(__name__)

_BONE_PARENT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Hips": (),
    "Spine": ("Hips",),
    "Chest": ("Spine",),
    "UpperChest": ("Chest", "Spine"),
    "Neck": ("UpperChest", "Chest", "Spine"),
    "Head": ("Neck", "UpperChest", "Chest", "Spine"),
    "LeftEye": ("Head",),
    "RightEye": ("Head",),
    "Jaw": ("Head",),
    "LeftUpperLeg": ("Hips",),
    "LeftLowerLeg": ("LeftUpperLeg",),
    "LeftFoot": ("LeftLowerLeg",),
    "LeftToes": ("LeftFoot",),
    "RightUpperLeg": ("Hips",),
    "RightLowerLeg": ("RightUpperLeg",),
    "RightFoot": ("RightLowerLeg",),
    "RightToes": ("RightFoot",),
    "LeftShoulder": ("UpperChest", "Chest", "Spine"),
    "LeftUpperArm": ("LeftShoulder", "UpperChest", "Chest", "Spine"),
    "LeftLowerArm": ("LeftUpperArm",),
    "LeftHand": ("LeftLowerArm",),
    "RightShoulder": ("UpperChest", "Chest", "Spine"),
    "RightUpperArm": ("RightShoulder", "UpperChest", "Chest", "Spine"),
    "RightLowerArm": ("RightUpperArm",),
    "RightHand": ("RightLowerArm",),
    "LeftThumbMetacarpal": ("LeftHand",),
    "LeftThumbProximal": ("LeftThumbMetacarpal", "LeftHand"),
    "LeftThumbIntermediate": ("LeftThumbProximal",),
    "LeftThumbDistal": ("LeftThumbIntermediate", "LeftThumbProximal"),
    "LeftIndexProximal": ("LeftHand",),
    "LeftIndexIntermediate": ("LeftIndexProximal",),
    "LeftIndexDistal": ("LeftIndexIntermediate",),
    "LeftMiddleProximal": ("LeftHand",),
    "LeftMiddleIntermediate": ("LeftMiddleProximal",),
    "LeftMiddleDistal": ("LeftMiddleIntermediate",),
    "LeftRingProximal": ("LeftHand",),
    "LeftRingIntermediate": ("LeftRingProximal",),
    "LeftRingDistal": ("LeftRingIntermediate",),
    "LeftLittleProximal": ("LeftHand",),
    "LeftLittleIntermediate": ("LeftLittleProximal",),
    "LeftLittleDistal": ("LeftLittleIntermediate",),
    "RightThumbMetacarpal": ("RightHand",),
    "RightThumbProximal": ("RightThumbMetacarpal", "RightHand"),
    "RightThumbIntermediate": ("RightThumbProximal",),
    "RightThumbDistal": ("RightThumbIntermediate", "RightThumbProximal"),
    "RightIndexProximal": ("RightHand",),
    "RightIndexIntermediate": ("RightIndexProximal",),
    "RightIndexDistal": ("RightIndexIntermediate",),
    "RightMiddleProximal": ("RightHand",),
    "RightMiddleIntermediate": ("RightMiddleProximal",),
    "RightMiddleDistal": ("RightMiddleIntermediate",),
    "RightRingProximal": ("RightHand",),
    "RightRingIntermediate": ("RightRingProximal",),
    "RightRingDistal": ("RightRingIntermediate",),
    "RightLittleProximal": ("RightHand",),
    "RightLittleIntermediate": ("RightLittleProximal",),
    "RightLittleDistal": ("RightLittleIntermediate",),
}
_BONE_ALIAS_TO_CANONICAL: dict[str, str] = {name.lower(): name for name in _BONE_PARENT_CANDIDATES}


@dataclass(frozen=True)
class _SkeletonEntry:
    rx_ts_ms: int
    payload: Any


@dataclass(frozen=True)
class _InputPacket:
    rx_ts_ms: int
    raw: bytes


def _skeleton_bone_schema():
    return complex_object_schema(
        properties={
            "name": string_schema(),
            "pos": array_schema(items=number_schema()),
            "rot": array_schema(items=number_schema()),
        }
    )


def _skeleton_payload_schema():
    return complex_object_schema(
        properties={
            "type": string_schema(),
            "modelName": string_schema(),
            "timestampMs": integer_schema(),
            "schema": string_schema(),
            "skeletonProtocol": string_schema(),
            "boneCount": integer_schema(),
            "bones": array_schema(items=_skeleton_bone_schema()),
        }
    )


class VmcDecoderRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
            exec_in_ports=[str(p) for p in (node.execInPorts or [])],
            exec_out_ports=[str(p) for p in (node.execOutPorts or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._models_lock = asyncio.Lock()
        self._cleanup_after_ms = self._coerce_int_or_default(
            self._initial_state.get("cleanupAfterMs", 10000),
            default=10000,
        )
        self._selected_key = str(self._initial_state.get("selectedKey", "") or "")
        self._skeletons_by_key: dict[str, _SkeletonEntry] = {}
        self._bones_by_key: dict[str, dict[str, dict[str, Any]]] = {}
        self._pending_bones_by_key: dict[str, dict[str, dict[str, Any]]] = {}
        self._pending_root_by_key: dict[str, dict[str, Any]] = {}
        self._output_version = 0
        self._last_synced_keys: list[str] = []
        self._ctx_output_cache: dict[tuple[str, str | int | None], tuple[int, Any]] = {}

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        packet_value = await self.pull("packet", ctx_id=exec_id)
        input_packet = self._coerce_input_packet(packet_value)
        if input_packet is None:
            return []

        await self._cleanup_stale(now_ts_ms=input_packet.rx_ts_ms)
        messages = self._decode_osc_packet(input_packet.raw)
        if not messages:
            return []

        keys_changed = False
        committed = False
        async with self._models_lock:
            pending_existing = self._pending_bones_by_key.get(_VMC_MODEL_NAME, {})
            pending_working: dict[str, dict[str, Any]] = {
                name: dict(bone_payload) for name, bone_payload in pending_existing.items()
            }
            root_working = self._pending_root_by_key.get(_VMC_MODEL_NAME)
            has_bone_update = False
            saw_ok = False

            for address, args in messages:
                normalized_address = str(address or "").strip().lower()
                if normalized_address == _VMC_BONE_POS_ADDR.lower():
                    updated = self._update_bone_from_vmc(args)
                    if updated is None:
                        continue
                    bone_name, bone_payload = updated
                    pending_working[bone_name] = bone_payload
                    has_bone_update = True
                    continue
                if normalized_address == _VMC_ROOT_POS_ADDR.lower():
                    root_parsed = self._parse_root_from_vmc(args)
                    if root_parsed is None:
                        continue
                    root_working = root_parsed
                    continue
                if normalized_address == _VMC_OK_ADDR.lower():
                    saw_ok = True

            self._pending_bones_by_key[_VMC_MODEL_NAME] = pending_working
            if root_working is not None:
                self._pending_root_by_key[_VMC_MODEL_NAME] = root_working

            should_commit = has_bone_update or (saw_ok and bool(pending_working))
            if should_commit:
                local_bones = {name: dict(bone_payload) for name, bone_payload in pending_working.items()}
                root_for_commit = self._pending_root_by_key.get(_VMC_MODEL_NAME)
                live_bones = self._compute_world_bones(local_bones=local_bones, root=root_for_commit)
                self._bones_by_key[_VMC_MODEL_NAME] = live_bones
                payload = self._build_payload(rx_ts_ms=int(input_packet.rx_ts_ms), bones_map=live_bones)
                keys_changed = _VMC_MODEL_NAME not in self._skeletons_by_key
                self._skeletons_by_key[_VMC_MODEL_NAME] = _SkeletonEntry(
                    rx_ts_ms=int(input_packet.rx_ts_ms),
                    payload=payload,
                )
                self._bump_output_version()
                committed = True

        if keys_changed:
            await self._sync_available_keys_and_selection()
        if not committed:
            return []
        return ["packet"]

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        del ctx_id
        p = str(port or "").strip()
        if p not in ("skeletons", "selectedSkeleton"):
            return None

        await self._cleanup_stale()
        await self._sync_available_keys_and_selection()

        cache_key = (p, None)
        async with self._models_lock:
            current_version = int(self._output_version)
            cached = self._ctx_output_cache.get(cache_key)
            if cached is not None and int(cached[0]) == current_version:
                return cached[1]
            keys = sorted(self._skeletons_by_key.keys())
            if p == "skeletons":
                value = [self._skeletons_by_key[key].payload for key in keys]
            else:
                selected = self._skeletons_by_key.get(self._selected_key) if self._selected_key else None
                value = None if selected is None else selected.payload
            if len(self._ctx_output_cache) > 512:
                self._ctx_output_cache.clear()
            self._ctx_output_cache[cache_key] = (current_version, value)
            return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        field_name = str(field or "").strip()
        if field_name == "cleanupAfterMs":
            self._cleanup_after_ms = self._coerce_int_or_default(value, default=self._cleanup_after_ms)
            await self._cleanup_stale()
            await self._sync_available_keys_and_selection()
            return
        if field_name == "selectedKey":
            selected_key = str(value or "").strip()
            if selected_key != self._selected_key:
                self._selected_key = selected_key
                self._bump_output_version()
            await self._sync_available_keys_and_selection()

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        field_name = str(field or "").strip()
        if field_name == "cleanupAfterMs":
            cleanup_after_ms = self._coerce_int_or_default(value, default=self._cleanup_after_ms)
            if cleanup_after_ms < 0 or cleanup_after_ms > 60_000_000:
                raise ValueError("cleanupAfterMs must be in range 0..60000000")
            return cleanup_after_ms
        if field_name == "selectedKey":
            return str(value or "").strip()
        return value

    @staticmethod
    def _coerce_int_or_default(value: Any, *, default: int) -> int:
        if value is None or isinstance(value, bool):
            return int(default)
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return int(default)

    def _bump_output_version(self) -> None:
        self._output_version += 1
        if len(self._ctx_output_cache) > 512:
            self._ctx_output_cache.clear()

    @staticmethod
    def _normalize_selected_key(keys: list[str], selected_key: str) -> str:
        if not keys:
            return ""
        if selected_key in keys:
            return selected_key
        return keys[0]

    @staticmethod
    def _coerce_input_packet(packet_value: Any) -> _InputPacket | None:
        if isinstance(packet_value, dict):
            raw_value = packet_value.get("raw")
            if not isinstance(raw_value, (bytes, bytearray)):
                return None
            timestamp_value = packet_value.get("timestampMs")
            try:
                rx_ts_ms = int(timestamp_value)
            except (TypeError, ValueError):
                rx_ts_ms = int(now_ms())
            return _InputPacket(rx_ts_ms=rx_ts_ms, raw=bytes(raw_value))
        if isinstance(packet_value, (bytes, bytearray)):
            return _InputPacket(rx_ts_ms=int(now_ms()), raw=bytes(packet_value))
        return None

    def _build_payload(self, *, rx_ts_ms: int, bones_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
        bones = [bones_map[name] for name in bones_map]
        return {
            "type": "skeleton_binary",
            "modelName": _VMC_MODEL_NAME,
            "timestampMs": int(rx_ts_ms),
            "schema": "f8.skeleton.v1",
            "skeletonProtocol": _VMC_SKELETON_PROTOCOL,
            "boneCount": len(bones),
            "bones": bones,
            "trailer": None,
        }

    @staticmethod
    def _coerce_finite_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return float(numeric)

    @staticmethod
    def _canonicalize_bone_name(name: str) -> str:
        key = str(name or "").strip().lower()
        if not key:
            return ""
        canonical = _BONE_ALIAS_TO_CANONICAL.get(key)
        if canonical is not None:
            return canonical
        return str(name).strip()

    @staticmethod
    def _quat_mul(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    @staticmethod
    def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return (q[0], -q[1], -q[2], -q[3])

    @staticmethod
    def _quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if norm <= 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        inv = 1.0 / norm
        return (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)

    @classmethod
    def _quat_rotate_vec(cls, q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
        normalized_q = cls._quat_normalize(q)
        point = (0.0, v[0], v[1], v[2])
        rotated = cls._quat_mul(cls._quat_mul(normalized_q, point), cls._quat_conjugate(normalized_q))
        return (rotated[1], rotated[2], rotated[3])

    def _parse_root_from_vmc(self, args: tuple[Any, ...]) -> dict[str, Any] | None:
        values: list[float]
        if len(args) >= 8 and isinstance(args[0], str):
            values = []
            for index in range(1, 8):
                value = self._coerce_finite_float(args[index])
                if value is None:
                    return None
                values.append(value)
        elif len(args) >= 7:
            values = []
            for index in range(0, 7):
                value = self._coerce_finite_float(args[index])
                if value is None:
                    return None
                values.append(value)
        else:
            return None
        px, py, pz, qx, qy, qz, qw = values
        return {
            "pos": [px, py, pz],
            "rot": [qw, qx, qy, qz],
        }

    def _parse_pose(self, pose: dict[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
        pos_raw = pose.get("pos")
        rot_raw = pose.get("rot")
        if not isinstance(pos_raw, list) or not isinstance(rot_raw, list) or len(pos_raw) != 3 or len(rot_raw) != 4:
            return None
        try:
            pos = (float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
            rot = (float(rot_raw[0]), float(rot_raw[1]), float(rot_raw[2]), float(rot_raw[3]))
        except (TypeError, ValueError):
            return None
        return (pos, self._quat_normalize(rot))

    def _compose_pose(
        self,
        parent_pos: tuple[float, float, float],
        parent_rot: tuple[float, float, float, float],
        local_pos: tuple[float, float, float],
        local_rot: tuple[float, float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        rotated_local_pos = self._quat_rotate_vec(parent_rot, local_pos)
        world_pos = (
            parent_pos[0] + rotated_local_pos[0],
            parent_pos[1] + rotated_local_pos[1],
            parent_pos[2] + rotated_local_pos[2],
        )
        world_rot = self._quat_normalize(self._quat_mul(parent_rot, local_rot))
        return (world_pos, world_rot)

    def _resolve_existing_parent(self, bone_name: str, local_bones: dict[str, dict[str, Any]]) -> str | None:
        candidates = _BONE_PARENT_CANDIDATES.get(bone_name)
        if not candidates:
            return None
        for candidate in candidates:
            if candidate in local_bones:
                return candidate
        return None

    def _compute_world_bones(
        self, *, local_bones: dict[str, dict[str, Any]], root: dict[str, Any] | None
    ) -> dict[str, dict[str, Any]]:
        parsed_root = self._parse_pose(root) if root is not None else None
        if parsed_root is None:
            root_pos = (0.0, 0.0, 0.0)
            root_rot = (1.0, 0.0, 0.0, 0.0)
        else:
            root_pos, root_rot = parsed_root

        world_pose_cache: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {}
        visit_state: dict[str, int] = {}

        def _compute_for(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
            pose_raw = local_bones.get(name)
            if pose_raw is None:
                return None
            cached = world_pose_cache.get(name)
            if cached is not None:
                return cached
            state = visit_state.get(name, 0)
            if state == 1:
                return None
            visit_state[name] = 1
            parsed_local = self._parse_pose(pose_raw)
            if parsed_local is None:
                visit_state[name] = 2
                return None
            local_pos, local_rot = parsed_local
            if name == "Hips":
                result = self._compose_pose(root_pos, root_rot, local_pos, local_rot)
            else:
                parent_name = self._resolve_existing_parent(name, local_bones)
                if parent_name is None:
                    result = (local_pos, local_rot)
                else:
                    parent_world = _compute_for(parent_name)
                    if parent_world is None:
                        result = (local_pos, local_rot)
                    else:
                        parent_pos, parent_rot = parent_world
                        result = self._compose_pose(parent_pos, parent_rot, local_pos, local_rot)
            world_pose_cache[name] = result
            visit_state[name] = 2
            return result

        world_bones: dict[str, dict[str, Any]] = {}
        for name in local_bones:
            world_pose = _compute_for(name)
            if world_pose is None:
                continue
            pos, rot = world_pose
            world_bones[name] = {
                "name": name,
                "pos": [pos[0], pos[1], pos[2]],
                "rot": [rot[0], rot[1], rot[2], rot[3]],
            }
        return world_bones

    def _update_bone_from_vmc(self, args: tuple[Any, ...]) -> tuple[str, dict[str, Any]] | None:
        if len(args) < 8:
            return None
        bone_name = self._canonicalize_bone_name(str(args[0]).strip())
        if not bone_name:
            return None
        px = self._coerce_finite_float(args[1])
        py = self._coerce_finite_float(args[2])
        pz = self._coerce_finite_float(args[3])
        qx = self._coerce_finite_float(args[4])
        qy = self._coerce_finite_float(args[5])
        qz = self._coerce_finite_float(args[6])
        qw = self._coerce_finite_float(args[7])
        if px is None or py is None or pz is None or qx is None or qy is None or qz is None or qw is None:
            return None
        return (
            bone_name,
            {
                "name": bone_name,
                "pos": [px, py, pz],
                "rot": [qw, qx, qy, qz],
            },
        )

    async def _cleanup_stale(self, *, now_ts_ms: int | None = None) -> None:
        ttl_ms = int(self._cleanup_after_ms)
        if ttl_ms <= 0:
            return
        if now_ts_ms is None:
            now_ts_ms = int(now_ms())
        cutoff = int(now_ts_ms) - ttl_ms
        removed = False
        async with self._models_lock:
            for key, entry in list(self._skeletons_by_key.items()):
                if int(entry.rx_ts_ms) < cutoff:
                    self._skeletons_by_key.pop(key, None)
                    self._bones_by_key.pop(key, None)
                    self._pending_bones_by_key.pop(key, None)
                    self._pending_root_by_key.pop(key, None)
                    removed = True
            if removed:
                self._bump_output_version()
        if removed:
            await self._sync_available_keys_and_selection()

    async def _sync_available_keys_and_selection(self) -> None:
        async with self._models_lock:
            keys = sorted(self._skeletons_by_key.keys())
            selected_key = str(self._selected_key or "").strip()
        if keys != self._last_synced_keys:
            self._last_synced_keys = list(keys)
            await self.set_state("availableKeys", list(keys))
        next_selected_key = self._normalize_selected_key(keys, selected_key)
        if next_selected_key != self._selected_key:
            self._selected_key = next_selected_key
            self._bump_output_version()
            await self.set_state("selectedKey", next_selected_key)

    @staticmethod
    def _read_osc_string(buf: bytes, offset: int) -> tuple[str, int]:
        end = buf.find(b"\x00", offset)
        if end == -1:
            raise ValueError("Missing OSC string terminator")
        value = buf[offset:end].decode("utf-8", errors="ignore")
        end += 1
        pad = (4 - (end & 0x03)) & 0x03
        return value, end + pad

    @staticmethod
    def _decode_osc_message(data: bytes) -> tuple[str, tuple[Any, ...]] | None:
        if not data:
            return None
        try:
            offset = 0
            address, offset = VmcDecoderRuntimeNode._read_osc_string(data, offset)
            if not address:
                return None
            typetags, offset = VmcDecoderRuntimeNode._read_osc_string(data, offset)
            if not typetags.startswith(","):
                return (address, ())
            values: list[Any] = []
            for tag in typetags[1:]:
                if tag == "i":
                    if offset + 4 > len(data):
                        return None
                    values.append(struct.unpack(">i", data[offset : offset + 4])[0])
                    offset += 4
                elif tag == "f":
                    if offset + 4 > len(data):
                        return None
                    values.append(struct.unpack(">f", data[offset : offset + 4])[0])
                    offset += 4
                elif tag == "h":
                    if offset + 8 > len(data):
                        return None
                    values.append(struct.unpack(">q", data[offset : offset + 8])[0])
                    offset += 8
                elif tag == "d":
                    if offset + 8 > len(data):
                        return None
                    values.append(struct.unpack(">d", data[offset : offset + 8])[0])
                    offset += 8
                elif tag == "s":
                    text, offset = VmcDecoderRuntimeNode._read_osc_string(data, offset)
                    values.append(text)
                elif tag == "T":
                    values.append(True)
                elif tag == "F":
                    values.append(False)
                elif tag in {"N", "I", "[", "]"}:
                    values.append(None)
                elif tag in {"c", "r", "m"}:
                    if offset + 4 > len(data):
                        return None
                    offset += 4
                elif tag == "b":
                    if offset + 4 > len(data):
                        return None
                    size = struct.unpack(">i", data[offset : offset + 4])[0]
                    offset += 4
                    if size < 0 or offset + size > len(data):
                        return None
                    blob = data[offset : offset + size]
                    offset += size
                    pad = (4 - (size & 0x03)) & 0x03
                    if offset + pad > len(data):
                        return None
                    offset += pad
                    values.append(blob)
                else:
                    if offset + 4 > len(data):
                        return None
                    offset += 4
            return (address, tuple(values))
        except (ValueError, struct.error):
            return None

    @staticmethod
    def _decode_osc_packet(packet: bytes) -> list[tuple[str, tuple[Any, ...]]]:
        messages: list[tuple[str, tuple[Any, ...]]] = []
        if not packet:
            return messages
        if packet.startswith(b"#bundle\x00"):
            if len(packet) < 16:
                return messages
            offset = 16
            while offset + 4 <= len(packet):
                size = struct.unpack(">I", packet[offset : offset + 4])[0]
                offset += 4
                if size <= 0 or offset + size > len(packet):
                    break
                element = packet[offset : offset + size]
                offset += size
                nested = VmcDecoderRuntimeNode._decode_osc_packet(element)
                if nested:
                    messages.extend(nested)
            return messages
        decoded = VmcDecoderRuntimeNode._decode_osc_message(packet)
        if decoded is not None:
            messages.append(decoded)
        return messages


VmcDecoderRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="VMC Decoder",
    description="Decodes udp_in packet payloads carrying VMC OSC messages into skeleton streams.",
    tags=["decode", "vmc", "osc", "skeleton", "mocap"],
    execInPorts=exec_port_specs(["packet"]),
    execOutPorts=exec_port_specs(["packet"]),
    dataInPorts=[
        F8DataPortSpec(
            name="packet",
            description="Packet payload from udp_in.packet.",
            valueSchema=any_schema(),
        )
    ],
    dataOutPorts=[
        F8DataPortSpec(
            name="skeletons",
            description="List of latest payloads (ordered by key).",
            valueSchema=array_schema(items=_skeleton_payload_schema()),
        ),
        F8DataPortSpec(
            name="selectedSkeleton",
            description="Latest payload matching `selectedKey` (or None).",
            valueSchema=_skeleton_payload_schema(),
        ),
    ],
    stateFields=[
        F8StateSpec(
            name="cleanupAfterMs",
            label="Cleanup After (ms)",
            description="Remove models that haven't updated for this many ms (<=0 disables cleanup).",
            valueSchema=integer_schema(default=10000, minimum=0, maximum=60_000_000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="selectedKey",
            label="Selected Key",
            description="If set and matches an available key, outputs `selectedSkeleton`; otherwise None.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            valueRequired=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select, optionsFromState="availableKeys"),
            showOnNode=True,
        ),
        F8StateSpec(
            name="availableKeys",
            label="Available Keys",
            description="Read-only list of current keys (updated only on changes).",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(VmcDecoderRuntimeNode.SPEC, VmcDecoderRuntimeNode, overwrite=True)
    return registry
