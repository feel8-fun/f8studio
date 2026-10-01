from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging

import msgspec

from f8pysdk.service_runtime_tools.deploy import ServiceProcessConfig, ServiceProcessManager

from .catalog import CatalogService
from .events import EventJournal
from .errors import ConflictError
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
        service_enabled: Callable[[str], bool] = lambda _name: True,
    ) -> None:
        self._manager = ServiceProcessManager(catalog.sdk_catalog, catalog_provider=lambda: catalog.sdk_catalog)
        self._runtime_config = runtime_config
        self._events = events
        self._loop: asyncio.AbstractEventLoop | None = None
        self._log_tasks: set[asyncio.Task[object]] = set()
        self._running_classes: dict[str, str] = {}
        self._starting_classes: dict[str, str] = {}
        self._service_enabled = service_enabled

    def is_class_running(self, service_class: str) -> bool:
        return service_class in self._starting_classes.values() or any(name == service_class and self._manager.is_running(service_id)
                   for service_id, name in self._running_classes.items())

    def can_start(self, service_class: str) -> bool:
        return self._service_enabled(service_class) and self._manager.has_launcher(service_class)

    async def start(self, service_id: str, *, service_class: str) -> ManagedProcessResult:
        if not self._service_enabled(service_class):
            raise ConflictError(f'Extension service is disabled or uninstalled: {service_class}')
        if service_id in self._starting_classes:
            raise ConflictError(f'Service is already starting: {service_id}')
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
        self._starting_classes[service_id] = service_class
        startup = asyncio.create_task(asyncio.to_thread(self._manager.start, config, on_output=self._on_output))
        try:
            await asyncio.shield(startup)
        except asyncio.CancelledError:
            logger.debug('Service startup cancelled; stopping service %s', service_id, exc_info=True)
            try:
                await startup
            finally:
                await asyncio.to_thread(self._manager.stop, service_id)
            raise
        finally:
            self._starting_classes.pop(service_id)
        result = ManagedProcessResult(service_id=service_id, running=self._manager.is_running(service_id))
        if result.running:
            self._running_classes[service_id] = service_class
        await self._events.publish(
            event_type="service.process_started",
            scope=f"service:{service_id}",
            payload={"serviceId": service_id, "serviceClass": service_class, "running": result.running},
        )
        return result

    async def stop(self, service_id: str) -> ManagedProcessResult:
        await asyncio.to_thread(self._manager.stop, service_id)
        result = ManagedProcessResult(service_id=service_id, running=self._manager.is_running(service_id))
        if not result.running:
            self._running_classes.pop(service_id, None)
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
        self._running_classes.clear()
        self._loop = None
        tasks = tuple(self._log_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _on_output(self, service_id: str, line: str) -> None:
        line = line.rstrip()[:8192]
        if not line:
            return
        logger.info("service output service_id=%s: %s", service_id, line)
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._schedule_log_event, service_id, line)

    def _schedule_log_event(self, service_id: str, line: str) -> None:
        if self._loop is None:
            return
        task = asyncio.create_task(
            self._events.publish(
                event_type="service.log",
                scope=f"service:{service_id}",
                payload={"serviceId": service_id, "line": line},
                ),
            name=f"service-log-event:{service_id}",
        )
        self._log_tasks.add(task)
        task.add_done_callback(self._log_tasks.discard)
        task.add_done_callback(self._report_log_event_failure)

    @staticmethod
    def _report_log_event_failure(task: asyncio.Task[object]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("service log event publication failed", exc_info=error)


__all__ = ["ManagedProcessResult", "ManagedServiceProcesses"]
