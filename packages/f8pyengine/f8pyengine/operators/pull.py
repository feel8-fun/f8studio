from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from f8pysdk import (
    F8StateAccess,
    F8StateSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    boolean_schema,
    integer_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

logger = logging.getLogger(__name__)

OPERATOR_CLASS = "f8.pull"


class PullRuntimeNode(OperatorNode):
    """
    Periodic pull sink.

    This node periodically pulls every connected data input (including ports
    added dynamically via `editableDataInPorts`) to drive upstream computation.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._task: asyncio.Task[object] | None = None
        self._stop = asyncio.Event()
        self._last_log_ms_by_sig: dict[str, int] = {}

    def _should_log(self, sig: str, *, now_ms: int, interval_ms: int = 2000) -> bool:
        last_ms = int(self._last_log_ms_by_sig.get(sig, 0))
        if last_ms != 0 and (now_ms - last_ms) < int(interval_ms):
            return False
        self._last_log_ms_by_sig[sig] = int(now_ms)
        return True

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        if getattr(bus, "active", True):
            self._start_loop()

    async def close(self) -> None:
        await self._stop_loop()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        _ = meta
        if bool(active):
            self._start_loop()
            return
        await self._stop_loop()

    def _start_loop(self) -> None:
        task = self._task
        if task is not None and not task.done():
            return
        self._stop.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.exception("[%s:pull] start periodic loop failed: no running event loop", self.node_id)
            return
        self._task = loop.create_task(self._run_periodic(), name=f"pull:{self.node_id}")

    async def _stop_loop(self) -> None:
        self._stop.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _run_periodic(self) -> None:
        while not self._stop.is_set():
            enabled = await self._get_bool_state("autoTriggerEnabled", default=False)
            hz = await self._get_int_state("autoTriggerHz", default=10, minimum=1, maximum=120)
            period_s = 1.0 / float(max(1, hz))

            if enabled:
                exec_id = int(time.time() * 1000.0)
                await self._pull_all_ports(exec_id=exec_id)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=period_s)
            except asyncio.TimeoutError:
                pass

    async def _pull_all_ports(self, *, exec_id: str | int) -> None:
        ports = [str(p) for p in list(self.data_in_ports or []) if str(p).strip()]
        if not ports:
            return

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

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        # This node is timer-driven; exec input is intentionally unsupported.
        return []

    async def _get_int_state(self, name: str, *, default: int, minimum: int, maximum: int) -> int:
        value: Any = None
        try:
            value = await self.get_state_value(name)
        except Exception:
            value = None
        if value is None:
            value = self._initial_state.get(name)
        value = _unwrap_json_value(value)

        try:
            out = int(value) if value is not None else int(default)
        except (TypeError, ValueError):
            out = int(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

    async def _get_bool_state(self, name: str, *, default: bool) -> bool:
        value: Any = None
        try:
            value = await self.get_state_value(name)
        except Exception:
            value = None
        if value is None:
            value = self._initial_state.get(name)
        value = _unwrap_json_value(value)

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if value is None:
            return bool(default)
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        return bool(default)


def _unwrap_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_unwrap_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _unwrap_json_value(v) for k, v in value.items()}
    try:
        return _unwrap_json_value(value.root)
    except Exception:
        pass
    try:
        return _unwrap_json_value(value.model_dump(mode="json"))
    except Exception:
        pass
    return value


PullRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Pull",
    description="Hidden internal sink that periodically pulls all data inputs to trigger upstream computation.",
    tags=["__hidden__", "pull", "sink", "driver", "timer"],
    execInPorts=[],
    dataInPorts=[],
    dataOutPorts=[],
    editableDataInPorts=True,
    editableDataOutPorts=False,
    editableExecInPorts=False,
    editableExecOutPorts=False,
    editableStateFields=False,
    stateFields=[
        F8StateSpec(
            name="autoTriggerEnabled",
            label="Auto Trigger",
            description="When enabled, periodically pull all data inputs without exec.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="autoTriggerHz",
            label="Auto Trigger Hz",
            description="Periodic pull frequency in Hz when Auto Trigger is enabled.",
            valueSchema=integer_schema(default=10, minimum=1, maximum=120),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return PullRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(PullRuntimeNode.SPEC, overwrite=True)
    return reg
