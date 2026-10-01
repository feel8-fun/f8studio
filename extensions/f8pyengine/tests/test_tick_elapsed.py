from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from f8pysdk.executors.exec_flow import EntrypointContext
from f8pysdk.specs import F8RuntimeNode

from f8pyengine.constants import SERVICE_CLASS
from f8pyengine.operators.tick import TickRuntimeNode


class _TickContext:
    def __init__(self) -> None:
        self.tasks: list[asyncio.Task[Any]] = []

    def create_task(self, coro: Any, *, name: str | None = None) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        return task

    async def emit_exec(self, out_port: str, *, exec_id: str | int) -> None:
        del out_port, exec_id


def test_tick_exposes_elapsed_seconds_as_data_output() -> None:
    async def scenario() -> None:
        node = F8RuntimeNode(
            nodeId="tick", serviceId="engine", serviceClass=SERVICE_CLASS,
            operatorClass=TickRuntimeNode.SPEC.operatorClass,
            dataOutPorts=list(TickRuntimeNode.SPEC.dataOutPorts),
        )
        tick = TickRuntimeNode(node_id="tick", node=node, initial_state={"tickMs": 5})
        context = _TickContext()
        emitted = asyncio.Event()
        samples: list[float] = []

        async def capture(port: str, value: Any) -> None:
            if port == "elapsedSec":
                samples.append(float(value))
                emitted.set()

        with patch.object(tick, "emit", new=AsyncMock(side_effect=capture)):
            await tick.start_entrypoint(cast(EntrypointContext, context))
            await asyncio.wait_for(emitted.wait(), timeout=1.0)
            assert samples[0] >= 0.0
            assert await tick.compute_output("elapsedSec") == samples[-1]
            await tick.stop_entrypoint()
            await asyncio.gather(*context.tasks)

    asyncio.run(scenario())
