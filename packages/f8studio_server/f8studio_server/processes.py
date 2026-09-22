from __future__ import annotations

import asyncio
import logging

import msgspec

from f8pysdk.service_runtime_tools.deploy import ServiceProcessConfig, ServiceProcessManager

from .catalog import CatalogService
from .events import EventJournal
from .runtime import RuntimeConfig


logger = logging.getLogger(__name__)


class ManagedProcessResult(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    service_id: str
    running: bool


class ManagedServiceProcesses:
    def __init__(
        self,
        *,
        catalog: CatalogService,
        runtime_config: RuntimeConfig,
        events: EventJournal,
    ) -> None:
        self._manager = ServiceProcessManager(catalog.sdk_catalog)
        self._runtime_config = runtime_config
        self._events = events
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self, service_id: str, *, service_class: str) -> ManagedProcessResult:
        self._loop = asyncio.get_running_loop()
        config = ServiceProcessConfig(
            service_class=service_class,
            service_id=service_id,
            supervision_mode="studio_owned",
            bus_backend=self._runtime_config.bus_backend,
            zenoh_config_path=self._runtime_config.zenoh_config_path,
            zenoh_connect=self._runtime_config.zenoh_connect,
            zenoh_listen=self._runtime_config.zenoh_listen,
            zenoh_shm_pool_bytes=self._runtime_config.zenoh_shm_pool_bytes,
        )
        await asyncio.to_thread(self._manager.start, config, on_output=self._on_output)
        result = ManagedProcessResult(service_id=service_id, running=self._manager.is_running(service_id))
        await self._events.publish(
            event_type="service.process_started",
            scope=f"service:{service_id}",
            payload={"serviceId": service_id, "serviceClass": service_class, "running": result.running},
        )
        return result

    async def stop(self, service_id: str) -> ManagedProcessResult:
        await asyncio.to_thread(self._manager.stop, service_id)
        result = ManagedProcessResult(service_id=service_id, running=self._manager.is_running(service_id))
        await self._events.publish(
            event_type="service.process_stopped",
            scope=f"service:{service_id}",
            payload={"serviceId": service_id, "running": result.running},
        )
        return result

    def is_running(self, service_id: str) -> bool:
        return self._manager.is_running(service_id)

    async def close(self) -> None:
        service_ids = tuple(self._manager.service_ids())
        for service_id in service_ids:
            stopped = await asyncio.to_thread(self._manager.stop, service_id)
            if not stopped:
                logger.error("managed service did not stop during shutdown service_id=%s", service_id)
        self._loop = None

    def _on_output(self, service_id: str, line: str) -> None:
        logger.info("service output service_id=%s: %s", service_id, line.rstrip())
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._schedule_log_event, service_id, line)

    def _schedule_log_event(self, service_id: str, line: str) -> None:
        task = asyncio.create_task(
            self._events.publish(
                event_type="service.log",
                scope=f"service:{service_id}",
                payload={"serviceId": service_id, "line": line},
                reliable=False,
            ),
            name=f"service-log-event:{service_id}",
        )
        task.add_done_callback(self._report_log_event_failure)

    @staticmethod
    def _report_log_event_failure(task: asyncio.Task[object]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("service log event publication failed", exc_info=error)


__all__ = ["ManagedProcessResult", "ManagedServiceProcesses"]
