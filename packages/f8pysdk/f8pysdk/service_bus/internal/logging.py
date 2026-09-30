"""Internal logging helpers owned by `service_bus`."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime import ServiceBus


log = logging.getLogger(__name__)


def log_error_once(
    bus: "ServiceBus",
    *,
    key: str,
    message: str,
    exc: BaseException | None = None,
) -> None:
    """
    Log an error once per bus instance to prevent high-frequency log spam.
    """
    if key in bus._error_once:
        return
    bus._error_once.add(key)
    if bus._monitor_collector.enabled:
        error_message = str(message)
        if exc is not None:
            error_message = f"{message}: {type(exc).__name__}: {exc}"
        bus.report_error(
            str(bus.service_id),
            code="SERVICE_BUS_ERROR",
            message=error_message,
            severity="error",
            fingerprint=str(key),
        )
    if exc is None:
        log.error("service_bus[%s] %s", bus.service_id, message)
        return
    log.error("service_bus[%s] %s", bus.service_id, message, exc_info=exc)


__all__ = ["log_error_once"]
