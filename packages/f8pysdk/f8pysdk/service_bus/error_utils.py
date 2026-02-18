from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bus import ServiceBus


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
    if exc is None:
        log.error("service_bus[%s] %s", bus.service_id, message)
        return
    log.error("service_bus[%s] %s", bus.service_id, message, exc_info=exc)
