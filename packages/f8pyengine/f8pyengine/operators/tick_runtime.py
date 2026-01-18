from __future__ import annotations

import asyncio
from typing import Any

from f8pysdk import F8RuntimeNode
from f8pysdk.runtime import ServiceRuntimeNode, ensure_token

from ..engine_executor import SourceContext


class TickRuntimeNode(ServiceRuntimeNode):
    """
    Source operator that periodically emits exec triggers.

    The engine is responsible for calling `start_source/stop_source`.
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

    async def on_exec(self, _ctx_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def start_source(self, ctx: SourceContext) -> None:
        self._stop.clear()

        async def _loop() -> None:
            while not self._stop.is_set():
                tick_ms = await self.get_state("tickMs")
                if tick_ms is None:
                    tick_ms = self._initial_state.get("tickMs", 100)
                try:
                    ms = max(1, int(tick_ms))
                except Exception:
                    ms = 100
                await asyncio.sleep(float(ms) / 1000.0)
                ctx_id = int(asyncio.get_running_loop().time() * 1000)
                for p in list(self._exec_out_ports):
                    await ctx.emit_exec(str(p), ctx_id=ctx_id)

        ctx.create_task(_loop(), name=f"tick:{self.node_id}")

    async def stop_source(self) -> None:
        self._stop.set()

