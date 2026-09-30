from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Final

import msgspec

from f8pysdk.codec import coerce_bool
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataPortSpec, F8OperatorSchemaVersion, F8OperatorSpec, F8RuntimeNode,
    F8StateAccess, F8StateSpec, any_schema, boolean_schema, number_schema, string_schema,
)

from ..constants import SERVICE_CLASS

OPERATOR_CLASS: Final[str] = "f8.fbx_skeleton_player"
logger = logging.getLogger(__name__)


class AnimationClip(msgspec.Struct, rename="camel"):
    frame_rate: float
    bone_names: list[str]
    parents: list[int]
    frames: list[list[tuple[float, float, float, float, float, float, float]]]


def _blender_executable(configured: str) -> str:
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Blender executable does not exist: {path}")
        return str(path)
    found = shutil.which("blender")
    if found is not None:
        return found
    program_files = os.environ.get("PROGRAMFILES")
    if program_files:
        root = Path(program_files) / "Blender Foundation"
        for path in sorted(root.glob("Blender */blender.exe"), reverse=True):
            if path.is_file():
                return str(path)
    raise FileNotFoundError("Blender is required to read FBX; set the Blender Path state field")


def load_fbx_clip(source: Path, blender_path: str = "") -> AnimationClip:
    source = source.expanduser().resolve()
    if source.suffix.lower() != ".fbx":
        raise ValueError(f"Expected an FBX file: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"FBX file does not exist: {source}")
    exporter = Path(__file__).with_name("fbx_export.py")
    with tempfile.TemporaryDirectory(prefix="f8-fbx-") as directory:
        destination = Path(directory) / "clip.json"
        result = subprocess.run(
            [_blender_executable(blender_path), "--background", "--factory-startup", "--python", str(exporter),
             "--", str(source), str(destination)],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Blender failed to read {source}: {result.stderr.strip() or result.stdout.strip()}")
        clip = msgspec.json.decode(destination.read_bytes(), type=AnimationClip)
    if clip.frame_rate <= 0 or not clip.frames or not clip.bone_names or len(clip.parents) != len(clip.bone_names):
        raise ValueError(f"FBX contains no valid skeleton animation: {source}")
    if any(len(frame) != len(clip.bone_names) for frame in clip.frames):
        raise ValueError(f"FBX animation has inconsistent bone counts: {source}")
    return clip


class FbxSkeletonPlayerRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[port.name for port in (node.dataInPorts or [])],
            data_out_ports=[port.name for port in (node.dataOutPorts or [])],
            state_fields=[field.name for field in (node.stateFields or [])],
        )
        state = initial_state or {}
        self._path = str(state.get("path") or "").strip()
        self._blender_path = str(state.get("blenderPath") or "").strip()
        self._loop = coerce_bool(state.get("loop"), default=True)
        self._clip: AnimationClip | None = None
        self._attempted = False
        self._load_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        if self._load_task is not None:
            self._load_task.cancel()
            await asyncio.gather(self._load_task, return_exceptions=True)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if field == "path" or field == "blenderPath":
            if self._load_task is not None:
                self._load_task.cancel()
                self._load_task = None
            if field == "path":
                self._path = str(value or "").strip()
            else:
                self._blender_path = str(value or "").strip()
            self._clip = None
            self._attempted = False
        elif field == "loop":
            self._loop = coerce_bool(value, default=self._loop)

    async def _load_clip(self, path: str, blender_path: str) -> None:
        try:
            self._clip = await asyncio.to_thread(load_fbx_clip, Path(path), blender_path)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, msgspec.MsgspecError):
            logger.exception("FBX skeleton player failed to load path=%s node_id=%s", path, self.node_id)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if port != "skeletons":
            return None
        external_time = await self.pull("timeSec", ctx_id=ctx_id)
        if external_time is None:
            return None
        if isinstance(external_time, bool) or not isinstance(external_time, (int, float)):
            raise ValueError("FBX Player timeSec must be a finite number of seconds")
        elapsed = float(external_time)
        if not math.isfinite(elapsed):
            raise ValueError("FBX Player timeSec must be a finite number of seconds")
        if self._clip is None and not self._attempted and self._path:
            self._attempted = True
            self._load_task = asyncio.create_task(self._load_clip(self._path, self._blender_path), name=f"fbx-load:{self.node_id}")
        load_task = self._load_task
        if load_task is not None:
            try:
                await load_task
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
                return None
            finally:
                if self._load_task is load_task:
                    self._load_task = None
        clip = self._clip
        if clip is None:
            return None
        sequence_length = len(clip.frames) / clip.frame_rate
        if not self._loop and elapsed >= sequence_length:
            return None
        position = max(0.0, elapsed % sequence_length if self._loop else elapsed)
        index = min(int(position * clip.frame_rate), len(clip.frames) - 1)
        frame = clip.frames[index]
        bones = [
            {"name": name, "parent": clip.bone_names[clip.parents[bone_index]] if clip.parents[bone_index] >= 0 else "",
             "pos": list(sample[:3]), "rot": list(sample[3:])}
            for bone_index, (name, sample) in enumerate(zip(clip.bone_names, frame, strict=False))
        ]
        return {"modelName": Path(self._path).stem, "skeletonProtocol": "fbx", "bones": bones}


FbxSkeletonPlayerRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.1.0",
    label="FBX Skeleton Player",
    description="Play an animated FBX armature as a skeleton stream using Blender for import.",
    tags=["skeleton", "fbx", "animation", "source"],
    dataInPorts=[F8DataPortSpec(name="timeSec", description="Playback time in seconds.", valueSchema=number_schema())],
    dataOutPorts=[F8DataPortSpec(name="skeletons", description="Current animated skeleton pose.", valueSchema=any_schema())],
    stateFields=[
        F8StateSpec(name="path", label="FBX Path", valueSchema=string_schema(default=""), access=F8StateAccess.rw, valueRequired=True, showOnNode=True),
        F8StateSpec(name="blenderPath", label="Blender Path", valueSchema=string_schema(default=""), access=F8StateAccess.rw, valueRequired=True),
        F8StateSpec(name="loop", label="Loop", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, valueRequired=True, showOnNode=True),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(FbxSkeletonPlayerRuntimeNode.SPEC, FbxSkeletonPlayerRuntimeNode, overwrite=True)
    return registry
