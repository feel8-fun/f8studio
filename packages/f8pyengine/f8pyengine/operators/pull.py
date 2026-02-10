from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from f8pysdk import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode, OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

logger = logging.getLogger(__name__)

OPERATOR_CLASS = "f8.pull"


class PullRuntimeNode(OperatorNode):
    """
    Exec-driven pull sink.

    When executed, this node pulls every connected data input (including ports
    added dynamically via `editableDataInPorts`) to drive upstream computation.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        _ = initial_state
        self._last_log_ms_by_sig: dict[str, int] = {}

    def _should_log(self, sig: str, *, now_ms: int, interval_ms: int = 2000) -> bool:
        last_ms = int(self._last_log_ms_by_sig.get(sig, 0))
        if last_ms != 0 and (now_ms - last_ms) < int(interval_ms):
            return False
        self._last_log_ms_by_sig[sig] = int(now_ms)
        return True

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        ports = [str(p) for p in list(self.data_in_ports or []) if str(p).strip()]
        if not ports:
            return []

        results = await asyncio.gather(
            *[self.pull(p, ctx_id=exec_id) for p in ports],
            return_exceptions=True,
        )
        for port, result in zip(ports, results, strict=True):
            if isinstance(result, Exception):
                now_ms = int(time.time() * 1000.0)
                sig = f"{type(result).__name__}:{result}:port={port}"
                if self._should_log(sig, now_ms=now_ms):
                    logger.exception("[%s:pull] pull failed (port=%s)", self.node_id, port)
        return []


PullRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Pull",
    description="Exec-driven sink that pulls all data inputs to drive upstream computation.",
    tags=["exec", "pull", "sink", "driver", "debug"],
    execInPorts=["exec"],
    dataInPorts=[],
    dataOutPorts=[],
    editableDataInPorts=True,
    editableDataOutPorts=False,
    editableExecInPorts=False,
    editableExecOutPorts=False,
    editableStateFields=False,
    stateFields=[],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return PullRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(PullRuntimeNode.SPEC, overwrite=True)
    return reg
