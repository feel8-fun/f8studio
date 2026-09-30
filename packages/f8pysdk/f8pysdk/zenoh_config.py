from __future__ import annotations

import json
from typing import Protocol


class ZenohConfigWriter(Protocol):
    def insert_json5(self, key: str, value: str) -> None: ...


def apply_zenoh_shared_memory_config(config: ZenohConfigWriter, *, shm_pool_bytes: int) -> None:
    """Configure the supported Zenoh 1.9 transport; invalid settings fail at startup."""
    config.insert_json5("transport/shared_memory/enabled", "true")
    config.insert_json5("transport/shared_memory/mode", json.dumps("init"))
    config.insert_json5("transport/shared_memory/transport_optimization/enabled", "true")
    if shm_pool_bytes > 0:
        config.insert_json5("transport/shared_memory/transport_optimization/pool_size", json.dumps(shm_pool_bytes))


def apply_zenoh_timestamping_config(config: ZenohConfigWriter) -> None:
    """Retained-state publishers require timestamps for sequencing metadata."""
    config.insert_json5("timestamping/enabled", "true")


__all__ = ["apply_zenoh_shared_memory_config", "apply_zenoh_timestamping_config"]
