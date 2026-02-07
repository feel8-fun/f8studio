from __future__ import annotations

import asyncio
import time
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    integer_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ..ui_bus import emit_ui_command

OPERATOR_CLASS = "f8.print"
RENDERER_CLASS = "pystudio_print"


class PyStudioPrintRuntimeNode(OperatorNode):
    """
    Studio-side runtime node for `f8.print`.

    This node runs inside the Studio process (`serviceId=studio`) and periodically
    pulls its `inputData` buffer, then emits UI commands for preview updates.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._task: asyncio.Task[object] | None = None
        self._last_preview_value: Any = None
        self._last_preview_ts: int | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except Exception:
            return
        self._task = loop.create_task(self._run(), name=f"pystudio:print:{self.node_id}")

    async def close(self) -> None:
        t = self._task
        self._task = None
        if t is None:
            return
        try:
            t.cancel()
        except Exception:
            pass
        try:
            await asyncio.gather(t, return_exceptions=True)
        except Exception:
            pass

    async def _run(self) -> None:
        while True:
            throttle = None
            try:
                throttle = await self.get_state_value("throttleMs")
            except Exception:
                throttle = None
            if throttle is None:
                throttle = self._initial_state.get("throttleMs", 100)
            try:
                throttle_ms = max(0, int(throttle) if throttle is not None else 100)
            except Exception:
                throttle_ms = 100

            try:
                v = await self.pull("inputData")
            except Exception:
                v = None

            if v is not None:
                changed = True
                try:
                    if self._last_preview_ts is not None and self._last_preview_value == v:
                        changed = False
                except Exception:
                    changed = True
                if changed:
                    ts_ms = int(time.time() * 1000)
                    self._last_preview_value = v
                    self._last_preview_ts = ts_ms
                    emit_ui_command(self.node_id, "preview.update", {"value": v}, ts_ms=ts_ms)

            await asyncio.sleep(max(0.02, float(throttle_ms) / 1000.0))


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    """
    Register:
    - runtime factory (studio in-process)
    - operator spec (for discovery/UI)
    """
    reg = registry or RuntimeNodeRegistry.instance()

    def _print_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PyStudioPrintRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _print_factory, overwrite=True)

    reg.register_operator_spec(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=SERVICE_CLASS,
            operatorClass=OPERATOR_CLASS,
            version="0.0.1",
            label="Print Node",
            description="Operator that displays incoming data in the editor (preview).",
            tags=["print", "console"],
            dataInPorts=[
                F8DataPortSpec(
                    name="inputData",
                    description="Data input to display (preview).",
                    valueSchema=any_schema(),
                ),
            ],
            dataOutPorts=[],
            rendererClass=RENDERER_CLASS,
            stateFields=[
                F8StateSpec(
                    name="throttleMs",
                    label="Throttle (ms)",
                    description="UI refresh interval in milliseconds (0 = refresh every tick).",
                    valueSchema=integer_schema(default=100, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
            ],
        ),
        overwrite=True,
    )

    return reg
