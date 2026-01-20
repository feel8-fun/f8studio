from __future__ import annotations

import asyncio
from typing import Any

from f8pysdk import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    integer_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from f8pysdk.executors.exec_flow import EntrypointContext

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.tick"


class TickRuntimeNode(RuntimeNode):
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

    async def on_exec(self, _ctx_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def start_entrypoint(self, ctx: EntrypointContext) -> None:
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

    async def stop_entrypoint(self) -> None:
        self._stop.set()


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
    ],
    execOutPorts=["exec"],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return TickRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(TickRuntimeNode.SPEC, overwrite=True)
    return reg

