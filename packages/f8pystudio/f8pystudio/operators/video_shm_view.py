from __future__ import annotations

import asyncio
import time
from typing import Any

from f8pysdk import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    integer_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.shm import video_shm_name

from ..constants import SERVICE_CLASS
from ..ui_bus import emit_ui_command


OPERATOR_CLASS = "f8.video_shm_view"
RENDERER_CLASS = "pystudio_videoshm"


def _default_video_shm_name(service_id: str) -> str:
    s = str(service_id or "").strip()
    return video_shm_name(s) if s else ""


class PyStudioVideoShmViewRuntimeNode(OperatorNode):
    """
    Studio-only visualization node: view a Video SHM (BGRA32) in a Qt widget.

    This runtime node only pushes config to the UI layer; the Qt widget reads
    shared memory directly (avoids pushing frame payloads through UiCommand).
    """

    SPEC = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=SERVICE_CLASS,
        operatorClass=OPERATOR_CLASS,
        version="0.0.1",
        label="Video SHM View",
        description="Display frames from a VideoSHM region (BGRA32).",
        tags=["ui", "shm", "video", "viewer"],
        dataInPorts=[],
        dataOutPorts=[],
        rendererClass=RENDERER_CLASS,
        stateFields=[
            F8StateSpec(
                name="uiUpdate",
                label="UI Update",
                description="Pause/resume embedded viewer updates in the editor.",
                valueSchema=boolean_schema(default=True),
                access=F8StateAccess.rw,
                showOnNode=False,
            ),
            F8StateSpec(
                name="serviceId",
                label="Service Id",
                description="If set and shmName is empty, uses shm.<serviceId>.video",
                valueSchema=string_schema(default=""),
                access=F8StateAccess.rw,
                showOnNode=False,
            ),
            F8StateSpec(
                name="shmName",
                label="SHM Name",
                description="Video SHM mapping name (e.g. shm.implayer.video). Overrides serviceId.",
                valueSchema=string_schema(default=""),
                access=F8StateAccess.rw,
                showOnNode=True,
            ),
            F8StateSpec(
                name="throttleMs",
                label="Refresh (ms)",
                description="UI refresh interval in milliseconds (0 = as fast as possible).",
                valueSchema=integer_schema(default=33, minimum=0, maximum=60000),
                access=F8StateAccess.rw,
                showOnNode=False,
            ),
        ],
    )

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._config_loaded = False
        self._service_id = ""
        self._shm_name = ""
        self._throttle_ms = 33
        self._pending_task: asyncio.Task[object] | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._ensure_config_loaded(), name=f"pystudio:videoshm:init:{self.node_id}")
        except RuntimeError:
            pass

    async def close(self) -> None:
        try:
            t = self._pending_task
            self._pending_task = None
            if t is not None:
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
        except (RuntimeError, TypeError):
            pass
        emit_ui_command(self.node_id, "videoshm.detach", {}, ts_ms=int(time.time() * 1000))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        f = str(field or "").strip()
        if f not in ("serviceId", "shmName", "throttleMs"):
            return
        await self._ensure_config_loaded()
        if f == "serviceId":
            self._service_id = str(await self._get_str_state("serviceId", default=self._service_id)).strip()
        elif f == "shmName":
            self._shm_name = str(await self._get_str_state("shmName", default=self._shm_name)).strip()
        elif f == "throttleMs":
            self._throttle_ms = await self._get_int_state("throttleMs", default=self._throttle_ms, minimum=0, maximum=60000)
        await self._push_config(now_ms=int(ts_ms) if ts_ms is not None else int(time.time() * 1000))

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._service_id = str(await self._get_str_state("serviceId", default=str(self._initial_state.get("serviceId", "")))).strip()
        self._shm_name = str(await self._get_str_state("shmName", default=str(self._initial_state.get("shmName", "")))).strip()
        self._throttle_ms = await self._get_int_state("throttleMs", default=33, minimum=0, maximum=60000)
        self._config_loaded = True
        await self._push_config(now_ms=int(time.time() * 1000))

    async def _push_config(self, *, now_ms: int) -> None:
        if self._pending_task is not None and not self._pending_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending_task = loop.create_task(self._push_config_async(now_ms), name=f"pystudio:videoshm:cfg:{self.node_id}")

    async def _push_config_async(self, now_ms: int) -> None:
        shm_name = str(self._shm_name or "").strip()
        if not shm_name:
            shm_name = _default_video_shm_name(self._service_id)
        emit_ui_command(
            self.node_id,
            "videoshm.set",
            {
                "shmName": shm_name,
                "serviceId": str(self._service_id or "").strip(),
                "throttleMs": int(self._throttle_ms),
            },
            ts_ms=int(now_ms),
        )

    async def _get_int_state(self, name: str, *, default: int, minimum: int, maximum: int) -> int:
        v: Any = None
        try:
            v = await self.get_state_value(name)
        except Exception:
            v = None
        if v is None:
            v = self._initial_state.get(name)
        try:
            out = int(v) if v is not None else int(default)
        except Exception:
            out = int(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

    async def _get_str_state(self, name: str, *, default: str) -> str:
        v: Any = None
        try:
            v = await self.get_state_value(name)
        except Exception:
            v = None
        if v is None:
            v = self._initial_state.get(name)
        try:
            s = str(v) if v is not None else str(default)
        except Exception:
            s = str(default)
        return s


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PyStudioVideoShmViewRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(PyStudioVideoShmViewRuntimeNode.SPEC, overwrite=True)
    return reg
