from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, TypeAlias

from nats.js.api import StorageType  # type: ignore[import-not-found]


DataDeliveryMode: TypeAlias = Literal["pull", "push", "both"]


def _debug_state_enabled() -> bool:
    return str(os.getenv("F8_STATE_DEBUG", "")).lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ServiceBusConfig:
    service_id: str
    service_name: str | None = None
    service_class: str | None = None
    nats_url: str = "nats://127.0.0.1:4222"
    publish_all_data: bool = True
    kv_storage: StorageType = StorageType.MEMORY
    delete_bucket_on_start: bool = False
    delete_bucket_on_stop: bool = False
    data_delivery: DataDeliveryMode = "pull"
    state_sync_concurrency: int = 8
    state_cache_max_entries: int = 8192
    data_input_max_buffers: int = 4096
    data_input_default_queue_size: int = 256
