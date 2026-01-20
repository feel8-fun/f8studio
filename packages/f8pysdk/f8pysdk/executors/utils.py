from __future__ import annotations

import asyncio
from typing import Any

from ..time_utils import now_ms


async def maybe_await(v: Any) -> None:
    if asyncio.iscoroutine(v):
        await v
