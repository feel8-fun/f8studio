from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from f8pysdk.motion import SkeletonPacketDecodeError, decode_skeleton_datagram
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

OPERATOR_CLASS = "f8.skeleton_decoder"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SkeletonEntry:
    rx_ts_ms: int
    payload: Any


@dataclass
class _ChunkFrameBuffer:
    frame_id: int
    chunk_count: int
    started_rx_ts_ms: int
    last_rx_ts_ms: int
    chunks: dict[int, list[dict[str, Any]]]


@dataclass(frozen=True)
class _InputPacket:
    rx_ts_ms: int
    key_fallback: str
    raw: bytes


def _skeleton_anim_schema():
    return complex_object_schema(
        properties={
            "normalizedTime": number_schema(),
            "layerIndex": integer_schema(),
            "clipName": string_schema(),
            "poseKey": string_schema(),
        }
    )


def _skeleton_trailer_schema():
    return complex_object_schema(
        properties={
            "magic": string_schema(),
            "extVersion": integer_schema(),
            "frameId": integer_schema(),
            "chunkIndex": integer_schema(),
            "chunkCount": integer_schema(),
            "totalBoneCount": integer_schema(),
            "characterId": integer_schema(),
            "assembledChunkCount": integer_schema(),
            "anim": _skeleton_anim_schema(),
        }
    )


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
            "boneCount": integer_schema(),
            "bones": array_schema(items=_skeleton_bone_schema()),
            "trailer": _skeleton_trailer_schema(),
        }
    )


class SkeletonDecoderRuntimeNode(OperatorNode):
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
        self._chunk_frames_by_key: dict[str, _ChunkFrameBuffer] = {}
        self._last_completed_frame_id_by_key: dict[str, int] = {}
        self._output_version = 0
        self._last_synced_keys: list[str] = []
        self._ctx_output_cache: dict[tuple[str, str | int | None], tuple[int, Any]] = {}
        self._reported_decode_errors: set[str] = set()

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        packet_value = await self.pull("packet", ctx_id=exec_id)
        input_packet = self._coerce_input_packet(packet_value)
        if input_packet is None:
            return []

        await self._cleanup_stale(now_ts_ms=input_packet.rx_ts_ms)
        payload = self._decode_payload(input_packet.raw)
        if not self._is_skeleton_payload(payload):
            return []
        payload["receivedAtMs"] = int(input_packet.rx_ts_ms)

        model_name = self._extract_model_name(payload)
        key = str(model_name or "").strip()
        if not key:
            key = input_packet.key_fallback
        if not key:
            return []

        keys_changed = False
        async with self._models_lock:
            merged_payload = self._merge_or_defer_chunk_payload(
                key=key,
                payload=payload,
                rx_ts_ms=int(input_packet.rx_ts_ms),
            )
            if merged_payload is None:
                return []
            entry = _SkeletonEntry(rx_ts_ms=int(input_packet.rx_ts_ms), payload=merged_payload)
            keys_changed = key not in self._skeletons_by_key
            self._skeletons_by_key[key] = entry
            self._bump_output_version()

        if keys_changed:
            await self._sync_available_keys_and_selection()
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
            remote_address = str(packet_value.get("remoteAddress") or "").strip()
            remote_port_value = packet_value.get("remotePort")
            remote_port_text = ""
            if remote_port_value is not None:
                try:
                    remote_port_text = str(int(remote_port_value))
                except (TypeError, ValueError):
                    remote_port_text = ""
            key_fallback = ""
            if remote_address and remote_port_text:
                key_fallback = f"{remote_address}:{remote_port_text}"
            return _InputPacket(rx_ts_ms=rx_ts_ms, key_fallback=key_fallback, raw=bytes(raw_value))
        if isinstance(packet_value, (bytes, bytearray)):
            return _InputPacket(rx_ts_ms=int(now_ms()), key_fallback="", raw=bytes(packet_value))
        return None

    @staticmethod
    def _extract_model_name(payload: Any) -> str | None:
        if isinstance(payload, dict):
            for key in ("modelName", "name", "character", "actor"):
                value = payload.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    return text
        return None

    @staticmethod
    def _is_skeleton_payload(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        bones_value = payload.get("bones")
        if not isinstance(bones_value, list):
            return False
        return bool(str(payload.get("modelName") or "").strip())

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                return None
        return None

    def _merge_or_defer_chunk_payload(self, *, key: str, payload: Any, rx_ts_ms: int) -> Any | None:
        if not isinstance(payload, dict):
            return None
        payload_type = payload.get("type")
        if payload_type != "skeleton_binary":
            return payload

        trailer_raw = payload.get("trailer")
        if not isinstance(trailer_raw, dict):
            return payload

        chunk_count = self._to_int(trailer_raw.get("chunkCount"))
        chunk_index = self._to_int(trailer_raw.get("chunkIndex"))
        frame_id = self._to_int(trailer_raw.get("frameId"))

        if chunk_count is None or chunk_index is None or frame_id is None:
            return payload
        if chunk_count <= 1:
            self._last_completed_frame_id_by_key[key] = frame_id
            return payload
        if chunk_index < 0 or chunk_index >= chunk_count:
            return None

        last_completed = self._last_completed_frame_id_by_key.get(key)
        if last_completed is not None and frame_id <= int(last_completed):
            return None

        active_buffer = self._chunk_frames_by_key.get(key)
        if active_buffer is not None:
            if frame_id < int(active_buffer.frame_id):
                return None
            if frame_id > int(active_buffer.frame_id):
                active_buffer = None
            elif chunk_count != int(active_buffer.chunk_count):
                active_buffer = None

        if active_buffer is None:
            active_buffer = _ChunkFrameBuffer(
                frame_id=frame_id,
                chunk_count=chunk_count,
                started_rx_ts_ms=rx_ts_ms,
                last_rx_ts_ms=rx_ts_ms,
                chunks={},
            )
            self._chunk_frames_by_key[key] = active_buffer

        bones_raw = payload.get("bones")
        if not isinstance(bones_raw, list):
            return None
        chunk_bones: list[dict[str, Any]] = []
        for bone in bones_raw:
            if isinstance(bone, dict):
                chunk_bones.append(bone)

        active_buffer.chunks[chunk_index] = chunk_bones
        active_buffer.last_rx_ts_ms = rx_ts_ms

        if len(active_buffer.chunks) < active_buffer.chunk_count:
            return None

        merged_bones: list[dict[str, Any]] = []
        for chunk_offset in range(active_buffer.chunk_count):
            merged_bones.extend(active_buffer.chunks.get(chunk_offset, []))

        merged_payload = dict(payload)
        merged_payload["bones"] = merged_bones
        merged_payload["boneCount"] = len(merged_bones)

        merged_trailer = dict(trailer_raw)
        merged_trailer["chunkIndex"] = 0
        merged_trailer["chunkCount"] = 1
        merged_trailer["assembledChunkCount"] = active_buffer.chunk_count
        merged_payload["trailer"] = merged_trailer

        self._chunk_frames_by_key.pop(key, None)
        self._last_completed_frame_id_by_key[key] = frame_id
        return merged_payload

    def _decode_payload(self, raw: bytes) -> Any:
        try:
            return decode_skeleton_datagram(raw)
        except SkeletonPacketDecodeError as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message not in self._reported_decode_errors:
                if len(self._reported_decode_errors) >= 16:
                    self._reported_decode_errors.clear()
                self._reported_decode_errors.add(message)
                logger.warning("[%s:skeleton_decoder] rejected datagram: %s", self.node_id, message, exc_info=True)
            return None

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
                    removed = True
            for key, pending in list(self._chunk_frames_by_key.items()):
                if int(pending.last_rx_ts_ms) < cutoff:
                    self._chunk_frames_by_key.pop(key, None)
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


SkeletonDecoderRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Skeleton Decoder",
    description="Decodes udp_in packet payloads into skeleton streams with chunk reassembly.",
    tags=["decode", "skeleton", "mocap", "udp"],
    execInPorts=["packet"],
    execOutPorts=["packet"],
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
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="selectedKey",
            label="Selected Key",
            description="If set and matches an available key, outputs `selectedSkeleton`; otherwise None.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select, optionsFromState="availableKeys"),
            showOnNode=True,
        ),
        F8StateSpec(
            name="availableKeys",
            label="Available Keys",
            description="Read-only list of current keys (updated only on changes).",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(SkeletonDecoderRuntimeNode.SPEC, SkeletonDecoderRuntimeNode, overwrite=True)
    return registry
