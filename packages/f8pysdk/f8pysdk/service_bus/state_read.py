from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StateRead:
    """
    Canonical result type for reading state from the bus.

    - found: whether the KV key exists (distinguishes missing vs stored None)
    - value: decoded value (or raw bytes on decode error)
    - ts_ms: best-effort timestamp extracted from payload (None if missing)
    """

    found: bool
    value: Any
    ts_ms: int | None
