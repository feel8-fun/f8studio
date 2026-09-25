from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    integer_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.registry import Registry

from ..identifiers import SERVICE_CLASS
from .categories import PALETTE_CATEGORY_VIZ
from ._runtime_errors import OPERATOR_PULL_ERRORS, OPERATOR_STATE_READ_ERRORS, OPERATOR_VALUE_COMPARE_ERRORS
from ._viz_base import StudioVizRuntimeNodeBase, viz_sampling_state_fields

OPERATOR_CLASS = "f8.viz.text"
RENDERER_CLASS = "viz_text"
logger = logging.getLogger(__name__)


class VizTextRuntimeNode(StudioVizRuntimeNodeBase):
    """
    Studio-side runtime node for `f8.viz.text`.

    This node runs inside the Studio process (`serviceId=studio`) and periodically
    pulls its `inputData` buffer, then emits presentation updates.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
            initial_state=initial_state,
        )
        self._task: asyncio.Task[object] | None = None
        self._last_preview_value: Any = None
        self._last_preview_ts: int | None = None
        self._last_loop_error_sig = ""
        self._last_loop_error_log_ts_ms = 0

    def _should_log_repeating_error(self, sig: str, *, now_ms: int) -> bool:
        if sig != self._last_loop_error_sig:
            self._last_loop_error_sig = sig
            self._last_loop_error_log_ts_ms = int(now_ms)
            return True
        if (int(now_ms) - int(self._last_loop_error_log_ts_ms)) >= 5000:
            self._last_loop_error_log_ts_ms = int(now_ms)
            return True
        return False

    def _log_loop_exception_once(self, *, kind: str, exc: Exception) -> None:
        now_ms = int(time.time() * 1000.0)
        sig = f"{kind}:{type(exc).__name__}:{exc}"
        if self._should_log_repeating_error(sig, now_ms=now_ms):
            logger.exception("[%s:viz_text] %s failed", self.node_id, kind)

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self._run(), name=f"pystudio:print:{self.node_id}")

    async def close(self) -> None:
        t = self._task
        self._task = None
        if t is None:
            return
        try:
            t.cancel()
        except (RuntimeError, TypeError):
            pass
        try:
            await asyncio.gather(t, return_exceptions=True)
        except (RuntimeError, TypeError):
            pass

    async def _run(self) -> None:
        while True:
            throttle = None
            try:
                throttle = await self.get_state_value("throttleMs")
            except OPERATOR_STATE_READ_ERRORS as exc:
                self._log_loop_exception_once(kind="read throttleMs", exc=exc)
                throttle = None
            if throttle is None:
                throttle = self._initial_state.get("throttleMs", 100)
            try:
                throttle_ms = max(0, int(throttle) if throttle is not None else 100)
            except (TypeError, ValueError):
                throttle_ms = 100

            try:
                v = await self.pull("inputData")
            except OPERATOR_PULL_ERRORS as exc:
                self._log_loop_exception_once(kind="pull inputData", exc=exc)
                v = None

            if v is not None:
                changed = True
                try:
                    if self._last_preview_ts is not None and self._last_preview_value == v:
                        changed = False
                except OPERATOR_VALUE_COMPARE_ERRORS as exc:
                    self._log_loop_exception_once(kind="compare preview value", exc=exc)
                    changed = True
                if changed:
                    ts_ms = int(time.time() * 1000)
                    self._last_preview_value = v
                    self._last_preview_ts = ts_ms
                    self.presentation.emit(self.node_id, "viz.text.update", {"value": v}, ts_ms=ts_ms)

            await asyncio.sleep(max(0.02, float(throttle_ms) / 1000.0))


def register_operator(registry: Registry) -> Registry:
    """
    Register:
    - runtime factory (studio in-process)
    - operator spec (for discovery/UI)
    """

    registry.register_operator(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=SERVICE_CLASS,
            paletteCategory=PALETTE_CATEGORY_VIZ,
            operatorClass=OPERATOR_CLASS,
            version="0.0.1",
            label="Text Viz",
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
                    name="uiUpdate",
                    label="UI Update",
                    description="Pause/resume embedded preview updates in the editor.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="uiWrap",
                    label="UI Wrap",
                    description="Whether embedded preview text wraps long lines.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="throttleMs",
                    label="Throttle (ms)",
                    description="UI refresh interval in milliseconds (0 = refresh every tick).",
                    valueSchema=integer_schema(default=100, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                *viz_sampling_state_fields(show_on_node=False),
            ],
        ),
        VizTextRuntimeNode,
        overwrite=True,
    )

    return registry
