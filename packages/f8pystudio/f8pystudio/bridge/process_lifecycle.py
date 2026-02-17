from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..service_process_manager import ServiceProcessConfig, ServiceProcessManager


@dataclass(frozen=True)
class StartServiceRequest:
    config: ServiceProcessConfig
    on_output: Callable[[str, str], None] | None = None


@dataclass(frozen=True)
class StopServiceRequest:
    service_id: str


@dataclass(frozen=True)
class StopServiceResult:
    service_id: str
    success: bool


class ServiceProcessGateway(Protocol):
    def service_ids(self) -> list[str]: ...

    def is_running(self, service_id: str) -> bool: ...

    def start(self, req: StartServiceRequest) -> None: ...

    def stop(self, req: StopServiceRequest) -> StopServiceResult: ...


@dataclass(frozen=True)
class LocalServiceProcessGateway:
    manager: ServiceProcessManager

    def service_ids(self) -> list[str]:
        return [str(service_id) for service_id in self.manager.service_ids()]

    def is_running(self, service_id: str) -> bool:
        return bool(self.manager.is_running(str(service_id)))

    def start(self, req: StartServiceRequest) -> None:
        self.manager.start(req.config, on_output=req.on_output)

    def stop(self, req: StopServiceRequest) -> StopServiceResult:
        service_id = str(req.service_id)
        success = bool(self.manager.stop(service_id))
        return StopServiceResult(service_id=service_id, success=success)
