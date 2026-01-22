from __future__ import annotations

import asyncio
import json
from typing import Any

from .nats_naming import kv_key_ready
from .nats_transport import NatsTransport
from .time_utils import now_ms


async def wait_service_ready(
    tr: NatsTransport,
    *,
    timeout_s: float = 6.0,
    min_ts_ms: int | None = None,
    max_age_ms: int | None = None,
) -> None:
    """
    Wait until a service announces readiness.

    Readiness is published via KV: key `ready` in the per-service bucket.
    This function waits using KV watch (non-polling) after an initial read.
    """
    min_ts = int(min_ts_ms) if min_ts_ms is not None else None
    max_age = int(max_age_ms) if max_age_ms is not None else None

    def _accept(payload: Any) -> bool:
        if not isinstance(payload, dict) or payload.get("ready") is not True:
            return False
        try:
            ts = int(payload.get("ts") or 0)
        except Exception:
            ts = 0
        if min_ts is not None and ts < min_ts:
            return False
        if max_age is not None:
            age = int(now_ms()) - ts
            if ts <= 0 or age > max_age:
                return False
        return True

    key = kv_key_ready()
    try:
        raw = await tr.kv_get(key)
    except Exception:
        raw = None
    if raw:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}
        if _accept(payload):
            return

    loop = asyncio.get_running_loop()
    fut: asyncio.Future[None] = loop.create_future()

    async def _on_kv(_key: str, value: bytes) -> None:
        if fut.done():
            return
        try:
            payload = json.loads((value or b"{}").decode("utf-8"))
        except Exception:
            payload = {}
        if _accept(payload):
            fut.set_result(None)

    watch = None
    try:
        watch = await tr.kv_watch(key, cb=_on_kv)
    except Exception:
        watch = None

    try:
        # Re-check KV after starting the watch to avoid a "missed update" race.
        try:
            raw2 = await tr.kv_get(key)
        except Exception:
            raw2 = None
        if raw2:
            try:
                payload2: Any = json.loads(raw2.decode("utf-8"))
            except Exception:
                payload2 = {}
            if _accept(payload2):
                return
        await asyncio.wait_for(fut, timeout=float(timeout_s))
    finally:
        if watch is not None:
            watcher, task = watch
            try:
                task.cancel()
            except Exception:
                pass
            try:
                await watcher.stop()
            except Exception:
                pass
