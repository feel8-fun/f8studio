from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from f8pysdk import F8OperatorSpec

from ..runtime.service_runtime_node import ServiceRuntimeNode


@dataclass
class OperatorContext:
    spec: F8OperatorSpec
    initial_state: dict[str, Any]


class OperatorRuntimeNode(ServiceRuntimeNode):
    """
    Engine-side runtime node for an operator instance.

    The engine executor drives `on_exec()` and may call `flush()` for sink nodes.
    """

    def __init__(self, *, node_id: str, ctx: OperatorContext) -> None:
        super().__init__(
            node_id=node_id,
            data_in_ports=[p.name for p in (ctx.spec.dataInPorts or [])],
            data_out_ports=[p.name for p in (ctx.spec.dataOutPorts or [])],
            state_fields=[s.name for s in (ctx.spec.states or [])],
        )
        self.ctx = ctx

    async def on_exec(self, _in_port: str | None = None) -> list[str]:
        return list(self.ctx.spec.execOutPorts or [])

    async def flush(self) -> None:
        return


class StartNode(OperatorRuntimeNode):
    async def on_exec(self, _in_port: str | None = None) -> list[str]:
        # Entry node: does no work, just fires exec outs.
        return list(self.ctx.spec.execOutPorts or [])


class ConstantNode(OperatorRuntimeNode):
    async def on_exec(self, _in_port: str | None = None) -> list[str]:
        # Prefer KV state; fallback to initial topology state.
        v = await self.get_state("value2")
        if v is None:
            v = self.ctx.initial_state.get("value2")
        await self.emit("value", v)
        return list(self.ctx.spec.execOutPorts or [])


class AddNode(OperatorRuntimeNode):
    async def on_exec(self, _in_port: str | None = None) -> list[str]:
        a = await self.pull("a")
        b = await self.pull("b")
        if a is None or b is None:
            return list(self.ctx.spec.execOutPorts or [])
        try:
            v = float(a) + float(b)
        except Exception:
            return list(self.ctx.spec.execOutPorts or [])
        await self.emit("sum", v)
        return list(self.ctx.spec.execOutPorts or [])


class LogNode(OperatorRuntimeNode):
    async def flush(self) -> None:
        v = await self.pull("value")
        if v is None:
            return
        label = await self.get_state("label")
        if label is None:
            label = self.ctx.initial_state.get("label") or "Log"
        print(f"[engine][{self.node_id}][{label}] value={v}")

