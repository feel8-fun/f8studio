from __future__ import annotations

import logging
from typing import Any

import msgspec

from f8pysdk.app import ServiceApp, ServiceAppDefaults
from f8pysdk.bus import ServiceBusConfig
from f8pysdk.capabilities import RungraphHook
from f8pysdk.codec import unwrap_json_value
from f8pysdk.logging_utils import configure_root_logging_from_env
from f8pysdk.registry import Registry
from f8pysdk.runtime import ServiceRuntime
from f8pysdk.specs import F8RuntimeGraph

from .constants import SERVICE_CLASS
from .script_node_registry import register_specs
from .script_service_node import PythonScriptServiceNode


class _ScriptRuntimeHooks(RungraphHook):
    def __init__(self) -> None:
        self._runtime: ServiceRuntime | None = None

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
        if not isinstance(node_any, PythonScriptServiceNode):
            return
        service_snapshot: Any | None = None
        for node in list(graph.nodes or []):
            if str(node.nodeId) == str(runtime.bus.service_id) and node.operatorClass is None:
                service_snapshot = node
                break
        if service_snapshot is None:
            return
        ts_ms: int | None = None
        meta = graph.meta
        if meta is not None and not isinstance(meta, msgspec.UnsetType) and meta.ts is not None and not isinstance(meta.ts, msgspec.UnsetType):
            try:
                ts_ms = int(meta.ts)
            except (TypeError, ValueError):
                ts_ms = None
        state_values = service_snapshot.stateValues or {}
        if not isinstance(state_values, dict):
            return
        for field, raw_value in state_values.items():
            await node_any.on_state(str(field), unwrap_json_value(raw_value), ts_ms=ts_ms)


def build_app() -> ServiceApp:
    registry = Registry()
    register_specs(registry)
    hooks = _ScriptRuntimeHooks()
    return ServiceApp(
        service_class=SERVICE_CLASS,
        registry=registry,
        defaults=ServiceAppDefaults(bus=ServiceBusConfig(data_delivery="callback")),
        setup=hooks.setup,
        teardown=hooks.teardown,
    )


def _main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        configure_root_logging_from_env()
    return build_app().cli(argv, program_name="F8PyScript")


if __name__ == "__main__":
    raise SystemExit(_main())
