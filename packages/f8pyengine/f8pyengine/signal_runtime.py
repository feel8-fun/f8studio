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

    async def on_exec(self, _in_port: str | None = None) -> list[str]:
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
        y = amp_f * math.sin(2.0 * math.pi * hz_f * t)
        await self.emit("value", y)
        return []


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

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if str(port) != "value":
            return
        print(f"[{self.node_id}] ts={ts_ms} value={value}")

