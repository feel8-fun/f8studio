from __future__ import annotations

import logging
import os
from typing import Any

from f8pysdk.capabilities import RungraphHook
from f8pysdk.generated import F8RuntimeGraph
from f8pysdk.json_unwrap import unwrap_json_value
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_cli import ServiceCliTemplate
from f8pysdk.service_runtime import ServiceRuntime, ServiceRuntimeConfig

from .constants import EXPR_SERVICE_CLASS
from .expr_node_registry import register_expr_specs
from .expr_service_node import PythonExprServiceNode


class PythonExprServiceProgram(ServiceCliTemplate, RungraphHook):
    def __init__(self) -> None:
        self._runtime: ServiceRuntime | None = None

    @property
    def service_class(self) -> str:
        return EXPR_SERVICE_CLASS

    def register_specs(self, registry: RuntimeNodeRegistry) -> None:
        register_expr_specs(registry)

    def build_runtime_config(self, *, service_id: str, nats_url: str) -> ServiceRuntimeConfig:
        return ServiceRuntimeConfig.from_values(
            service_id=service_id,
            service_class=self.service_class,
            nats_url=nats_url,
            data_delivery="both",
        )

    async def setup(self, runtime: ServiceRuntime) -> None:
        self._runtime = runtime
        runtime.bus.register_rungraph_hook(self)

    async def teardown(self, runtime: ServiceRuntime) -> None:
        try:
            runtime.bus.unregister_rungraph_hook(self)
        except (RuntimeError, ValueError) as exc:
            logging.getLogger(__name__).error("unregister_rungraph_hook failed", exc_info=exc)
        self._runtime = None

    async def validate_rungraph(self, graph: F8RuntimeGraph) -> None:
        _ = graph

    async def on_rungraph(self, graph: F8RuntimeGraph) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        node_any = runtime.bus.get_node(runtime.bus.service_id)
        if not isinstance(node_any, PythonExprServiceNode):
            return
        service_snapshot: Any | None = None
        for node in list(graph.nodes or []):
            if str(node.nodeId) == str(runtime.bus.service_id) and node.operatorClass is None:
                service_snapshot = node
                break
        if service_snapshot is None:
            return
        node_any.data_in_ports = [str(p.name) for p in list(service_snapshot.dataInPorts or [])] or ["in"]
        node_any.data_out_ports = [str(p.name) for p in list(service_snapshot.dataOutPorts or [])] or ["out"]
        node_any.state_fields = [str(s.name) for s in list(service_snapshot.stateFields or [])]
        ts_ms: int | None = None
        if graph.meta is not None and graph.meta.ts is not None:
            try:
                ts_ms = int(graph.meta.ts)
            except (TypeError, ValueError):
                ts_ms = None
        state_values = service_snapshot.stateValues or {}
        if not isinstance(state_values, dict):
            return
        for field, raw_value in state_values.items():
            await node_any.on_state(str(field), unwrap_json_value(raw_value), ts_ms=ts_ms)


def _main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        raw = (os.environ.get("F8_LOG_LEVEL") or "").strip().upper()
        level = getattr(logging, raw, logging.WARNING) if raw else logging.WARNING
        logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    return PythonExprServiceProgram().cli(argv, program_name="F8PyExpr")


if __name__ == "__main__":
    raise SystemExit(_main())
