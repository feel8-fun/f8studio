from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_runtime import ServiceRuntime, ServiceRuntimeConfig
from f8pysdk.capabilities import StateListenerBus

from .operators import register_operator, set_preview_sink
from .ui_bus import set_ui_command_sink, UiCommand
from .pystudio_node_registry import SERVICE_CLASS, STUDIO_SERVICE_ID


@dataclass(frozen=True)
class PyStudioServiceConfig:
    nats_url: str = "nats://127.0.0.1:4222"
    studio_service_id: str = STUDIO_SERVICE_ID


class PyStudioService:
    """
    PyStudio in-process "service" wiring.

    Mirrors the clarity of `pyengine_service.py`, but for the Qt app:
    - builds the runtime registry
    - registers studio runtime nodes/operators
    - constructs and starts a `ServiceRuntime` (ServiceBus + ServiceHost)
    - wires preview + state updates to UI callbacks
    """

    def __init__(
        self,
        config: PyStudioServiceConfig,
        *,
        registry: RuntimeNodeRegistry | None = None,
    ) -> None:
        self._cfg = config
        self._registry = registry or RuntimeNodeRegistry.instance()
        self.runtime: ServiceRuntime | None = None
        self._on_local_state: Callable[[str, str, Any, int, dict[str, Any]], Any] | None = None

    @property
    def studio_service_id(self) -> str:
        return str(self._cfg.studio_service_id)

    @property
    def bus(self):
        if self.runtime is None:
            return None
        return self.runtime.bus

    async def start(
        self,
        *,
        on_preview: Callable[[str, Any, int | None], None] | None,
        on_ui_command: Callable[[UiCommand], None] | None,
        on_local_state: Callable[[str, str, Any, int, dict[str, Any]], Any] | None,
    ) -> None:
        # Register studio operators into the shared registry.
        register_operator(self._registry)

        cfg = ServiceRuntimeConfig.from_values(
            service_id=str(self._cfg.studio_service_id),
            service_class=SERVICE_CLASS,
            nats_url=str(self._cfg.nats_url),
            publish_all_data=False,
            data_delivery="push",
        )
        self.runtime = ServiceRuntime(cfg, registry=self._registry)

        # In-process preview channel: runtime nodes can push UI-only preview updates without KV.
        if on_preview is not None:
            set_preview_sink(lambda node_id, value, ts_ms: on_preview(str(node_id), value, ts_ms))
        else:
            set_preview_sink(None)

        if on_ui_command is not None:
            set_ui_command_sink(on_ui_command)
        else:
            set_ui_command_sink(None)

        if on_local_state is not None:
            self._on_local_state = on_local_state
            try:
                self.runtime.bus.add_state_listener(on_local_state)
            except Exception:
                pass
        else:
            self._on_local_state = None

        await self.runtime.start()

    async def stop(self) -> None:
        set_preview_sink(None)
        set_ui_command_sink(None)
        rt = self.runtime
        self.runtime = None
        if rt is None:
            return
        cb = self._on_local_state
        self._on_local_state = None
        if cb is not None and isinstance(rt.bus, StateListenerBus):
            try:
                rt.bus.remove_state_listener(cb)
            except Exception:
                pass
        await rt.stop()
