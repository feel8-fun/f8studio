from __future__ import annotations

from f8pysdk.executors.exec_flow import ExecFlowExecutor
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_runtime import ServiceRuntime
from f8pysdk.service_cli import ServiceCliTemplate

from .engine_binder import EngineBinder
from .pyengine_node_registry import register_pyengine_specs


class PyEngineService(ServiceCliTemplate):
    """
    Fill-in-the-blanks service program for `f8.pyengine`.

    This is the canonical entry wiring:
    - register pyengine runtime node specs
    - attach ExecFlowExecutor (exec runtime)
    - bind exec-capable nodes via EngineBinder
    - pause/resume executor via ServiceBus lifecycle events
    """

    def __init__(self) -> None:
        self._executor: ExecFlowExecutor | None = None
        self._binder: EngineBinder | None = None

    @property
    def service_class(self) -> str:
        return "f8.pyengine"

    def register_specs(self, registry: RuntimeNodeRegistry) -> None:
        register_pyengine_specs(registry)

    async def setup(self, runtime: ServiceRuntime) -> None:
        executor = ExecFlowExecutor(runtime.bus)
        self._executor = executor

        self._binder = EngineBinder(bus=runtime.bus, executor=executor, service_class=self.service_class)

        async def _on_lifecycle(active: bool, _meta: dict[str, object]) -> None:
            await executor.set_active(active)

        runtime.bus.add_lifecycle_listener(_on_lifecycle)

    async def teardown(self, runtime: ServiceRuntime) -> None:
        executor = self._executor
        if executor is None:
            return
        try:
            await executor.stop_entrypoint()
        except Exception:
            pass
