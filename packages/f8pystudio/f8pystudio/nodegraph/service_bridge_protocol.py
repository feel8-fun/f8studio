from __future__ import annotations

from typing import Any, Protocol


class ServiceBridge(Protocol):
    """
    Minimal protocol for the Studio <-> service-process bridge used by nodegraph UI.

    This intentionally avoids importing Qt types here; signal objects are treated as `Any`.
    """

    service_process_state: Any

    def is_service_running(self, service_id: str) -> bool: ...

    def get_cached_service_active(self, service_id: str) -> bool | None: ...

    def set_service_active(self, service_id: str, active: bool) -> None: ...

    def start_service_and_deploy(self, service_id: str, *, service_class: str, compiled: Any | None = None) -> None: ...

    def restart_service_and_deploy(self, service_id: str, *, service_class: str, compiled: Any | None = None) -> None: ...

    def stop_service(self, service_id: str) -> None: ...

    def deploy_service_rungraph(self, service_id: str, *, compiled: Any | None = None) -> None: ...

    def invoke_remote_command(self, service_id: str, name: str, args: dict[str, Any] | None = None) -> None: ...

