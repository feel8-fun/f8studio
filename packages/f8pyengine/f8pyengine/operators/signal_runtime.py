from __future__ import annotations

import math
import time
from typing import Any

from f8pysdk import F8RuntimeNode
from f8pysdk.runtime import ServiceRuntimeNode, ensure_token


class SineRuntimeNode(ServiceRuntimeNode):
    """
    Exec-driven sine source (not a graph source): on exec, emits a numeric sample.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._exec_out_ports = list(getattr(node, "execOutPorts", None) or [])

    async def on_exec(self, _ctx_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port) != "value":
            return None
        hz = await self.get_state("hz")
        if hz is None:
            hz = self._initial_state.get("hz", 1.0)
        amp = await self.get_state("amp")
        if amp is None:
            amp = self._initial_state.get("amp", 1.0)
        try:
            hz_f = float(hz)
        except Exception:
            hz_f = 1.0
        try:
            amp_f = float(amp)
        except Exception:
            amp_f = 1.0

        t = time.time()
        return amp_f * math.sin(2.0 * math.pi * hz_f * t)


class PrintRuntimeNode(ServiceRuntimeNode):
    """
    Prints incoming values.

    For the demo flow, printing happens on data arrival (no exec required).
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

    async def on_exec(self, ctx_id: str | int, _in_port: str | None = None) -> list[str]:
        v = await self.pull("value", ctx_id=ctx_id)
        print(f"[{self.node_id}] ctx={ctx_id} value={v}")
        return []

