from __future__ import annotations

import asyncio
import sys
from typing import Any

from f8pysdk import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    F8DataPortSpec,
    boolean_schema,
    integer_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.capabilities import EntrypointNode
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from f8pysdk.executors.exec_flow import EntrypointContext

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.tick"


class TickRuntimeNode(OperatorNode, EntrypointNode):
    """
    Source operator that periodically emits exec triggers.

    The engine is responsible for calling `start_entrypoint/stop_entrypoint`.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._exec_out_ports = list(node.execOutPorts or []) or ["exec"]
        self._stop = asyncio.Event()
        self._hires_enabled = False
        try:
            self._tick_ms = self._coerce_tick_ms(self._initial_state.get("tickMs"), default=100)
        except ValueError:
            self._tick_ms = 100
        self._want_hires = self._coerce_bool(self._initial_state.get("hiResTimer"), default=True)

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def validate_state(
        self, field: str, value: Any, *, ts_ms: int | None = None, meta: dict[str, Any] | None = None
    ) -> Any:
        name = str(field or "").strip()
        if name == "tickMs":
            return self._coerce_tick_ms(value, default=100)
        if name == "hiResTimer":
            return self._coerce_bool(value, default=False)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "tickMs":
            self._tick_ms = self._coerce_tick_ms(value, default=self._tick_ms)
            return
        if name == "hiResTimer":
            self._want_hires = self._coerce_bool(value, default=self._want_hires)
            return

    def _apply_windows_timer_resolution(self, enabled: bool) -> None:
        """
        Best-effort: request 1ms timer resolution on Windows to reduce sleep jitter.
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes

            winmm = ctypes.windll.winmm  # type: ignore[attr-defined]
            if enabled:
                winmm.timeBeginPeriod(1)
            else:
                winmm.timeEndPeriod(1)
        except Exception:
            return

    async def start_entrypoint(self, ctx: EntrypointContext) -> None:
        self._stop.clear()

        async def _loop() -> None:
            loop = asyncio.get_running_loop()
            last_tick_ms: int | None = None
            next_deadline_s = loop.time()
            last_tick_start_s: float | None = None

            try:
                while not self._stop.is_set():
                    ms = int(self._tick_ms)
                    want_hires_bool = bool(self._want_hires)
                    if want_hires_bool != self._hires_enabled:
                        self._apply_windows_timer_resolution(want_hires_bool)
                        self._hires_enabled = want_hires_bool

                    period_s = float(ms) / 1000.0
                    if last_tick_ms is None or ms != last_tick_ms:
                        last_tick_ms = ms
                        next_deadline_s = loop.time() + period_s

                    sleep_s = next_deadline_s - loop.time()
                    if sleep_s > 0:
                        try:
                            await asyncio.wait_for(self._stop.wait(), timeout=sleep_s)
                            break
                        except asyncio.TimeoutError:
                            pass

                    tick_start_s = loop.time()
                    exec_id = int(tick_start_s * 1000)

                    interval_ms = 0
                    if last_tick_start_s is not None:
                        interval_ms = max(0, int((tick_start_s - last_tick_start_s) * 1000))
                    last_tick_start_s = tick_start_s
                    await self.emit("intervalMs", interval_ms)

                    lateness_ms = max(0, int((tick_start_s - next_deadline_s) * 1000))
                    await self.emit("latenessMs", lateness_ms)

                    for p in list(self._exec_out_ports):
                        await ctx.emit_exec(str(p), exec_id=exec_id)

                    processing_ms = max(0, int((loop.time() - tick_start_s) * 1000))
                    await self.emit("processingMs", processing_ms)

                    next_deadline_s += period_s
                    now_s = loop.time()
                    if next_deadline_s <= now_s:
                        missed = int((now_s - next_deadline_s) / period_s) + 1
                        next_deadline_s += missed * period_s
            finally:
                if self._hires_enabled:
                    self._apply_windows_timer_resolution(False)
                    self._hires_enabled = False

        ctx.create_task(_loop(), name=f"tick:{self.node_id}")

    async def stop_entrypoint(self) -> None:
        self._stop.set()

    @staticmethod
    def _coerce_tick_ms(value: Any, *, default: int) -> int:
        try:
            ms = int(value)
        except (TypeError, ValueError):
            raise ValueError("tickMs must be an integer") from None
        if ms < 1:
            raise ValueError("tickMs must be >= 1")
        if ms > 50000:
            raise ValueError("tickMs must be <= 50000")
        return int(ms)

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value or "").strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        return bool(default)


TickRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Tick",
    description="Source operator that generates periodic exec ticks.",
    tags=["execution", "timer", "start", "clock", "entrypoint"],
    stateFields=[
        F8StateSpec(
            name="tickMs",
            label="Tick (ms)",
            description="Interval in milliseconds for emitting exec ticks.",
            valueSchema=integer_schema(default=100, minimum=1, maximum=50000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="hiResTimer",
            label="High-res Timer (Windows)",
            description="Request 1ms system timer resolution to reduce jitter on Windows.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
    ],
    execOutPorts=["exec"],
    dataOutPorts=[
        F8DataPortSpec(
            name="processingMs",
            description="Per-tick processing time in milliseconds (excluding sleep).",
            valueSchema=integer_schema(default=0, minimum=0),
            showOnNode=False,
        ),
        F8DataPortSpec(
            name="intervalMs",
            description="Actual interval between tick starts in milliseconds.",
            valueSchema=integer_schema(default=0, minimum=0),
            showOnNode=False,
        ),
        F8DataPortSpec(
            name="latenessMs",
            description="How late this tick started relative to its scheduled deadline (ms).",
            valueSchema=integer_schema(default=0, minimum=0),
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return TickRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(TickRuntimeNode.SPEC, overwrite=True)
    return reg
