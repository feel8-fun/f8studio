from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from f8pysdk.nats_naming import (
    ensure_token,
    kv_bucket_for_service,
    kv_key_node_state,
    parse_kv_key_node_state,
)
from f8pysdk.nats_transport import NatsTransport, NatsTransportConfig
from f8pysdk.time_utils import now_ms

logger = logging.getLogger(__name__)


def _coerce_inbound_ts_ms(ts_raw: Any, *, default: int) -> int:
    """
    Best-effort coercion of inbound timestamps to milliseconds.

    Mirrors `f8pysdk.service_bus.payload.coerce_inbound_ts_ms(...)` but kept local
    to Studio so it can watch arbitrary services without depending on ServiceBus.
    """
    try:
        if ts_raw is None:
            return int(default)
        if isinstance(ts_raw, float):
            ts = int(ts_raw)
        elif isinstance(ts_raw, str):
            ts = int(ts_raw.strip() or "0")
        else:
            ts = int(ts_raw)
    except (TypeError, ValueError):
        return int(default)

    if ts <= 0:
        return int(default)

    if ts < 100_000_000_000:
        return int(ts * 1000)
    if ts >= 100_000_000_000_000_000:
        return int(ts // 1_000_000)
    if ts >= 100_000_000_000_000:
        return int(ts // 1000)
    return int(ts)


def _extract_ts_field(payload: dict[str, Any]) -> Any:
    return payload.get("tsMs")


@dataclass(frozen=True)
class WatchTarget:
    service_id: str
    node_id: str
    fields: tuple[str, ...]


class RemoteStateWatcher:
    """
    Studio-side remote KV watcher.

    Watches per-service KV buckets for node state updates and reports them via a
    callback so the UI can reflect runtime state without installing a monitor node.
    """

    def __init__(
        self,
        *,
        nats_url: str,
        studio_service_id: str,
        on_state: Callable[[str, str, str, Any, int, dict[str, Any]], Awaitable[None] | None],
    ) -> None:
        self._nats_url = str(nats_url or "").strip() or "nats://127.0.0.1:4222"
        self._studio_service_id = ensure_token(str(studio_service_id), label="studio_service_id")
        self._on_state = on_state

        # One shared transport for opening multiple KV buckets.
        self._tr = NatsTransport(
            NatsTransportConfig(url=self._nats_url, kv_bucket=kv_bucket_for_service(self._studio_service_id))
        )
        self._started = False
        self._watches: dict[tuple[str, str], Any] = {}  # (bucket, key_pattern) -> (watcher, task)
        self._targets: dict[tuple[str, str], WatchTarget] = {}
        self._field_filters: dict[tuple[str, str], frozenset[str]] = {}
        # Dedupe applied updates per (serviceId,nodeId,field).
        #
        # Important: do not treat `ts_ms` as a total order for UI updates. Many
        # runtime nodes propagate upstream timestamps, so "switching back" to an
        # older upstream sample can legitimately decrease ts while still
        # representing the latest KV write. The UI should reflect the KV's
        # current value in that case.
        self._last_by_key: dict[tuple[str, str, str], tuple[int, Any]] = {}  # -> (ts_ms, last_value)
        self._callback_error_once: set[tuple[str, str, str, str]] = set()

    @staticmethod
    async def _stop_watch_handle(watch: Any) -> None:
        watcher, task = watch
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await watcher.stop()

    async def start(self) -> None:
        if self._started:
            return
        await self._tr.connect()
        self._started = True

    async def stop(self) -> None:
        for (_bucket, _pattern), watch in list(self._watches.items()):
            try:
                await self._stop_watch_handle(watch)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.exception("Failed to stop remote state watch")
        self._watches.clear()
        self._targets.clear()
        self._field_filters.clear()
        self._last_by_key.clear()
        self._callback_error_once.clear()
        self._started = False
        try:
            await self._tr.close()
        except (AttributeError, OSError, RuntimeError, TypeError):
            pass

    async def apply_targets(self, targets: list[WatchTarget]) -> None:
        """
        Update desired watch set and perform best-effort initial sync.
        """
        await self.start()

        want_patterns: dict[tuple[str, str], WatchTarget] = {}
        for t in list(targets or []):
            try:
                sid = ensure_token(str(t.service_id), label="service_id")
                nid = ensure_token(str(t.node_id), label="node_id")
            except (AttributeError, TypeError, ValueError):
                continue
            bucket = kv_bucket_for_service(sid)
            pattern = f"nodes.{nid}.state.>"
            want_patterns[(bucket, pattern)] = WatchTarget(service_id=sid, node_id=nid, fields=tuple(t.fields or ()))

        changed_patterns: set[tuple[str, str]] = set()
        for key, target in want_patterns.items():
            prev = self._targets.get(key)
            if prev is None or tuple(prev.fields) != tuple(target.fields):
                changed_patterns.add(key)

        # Stop watches not needed.
        for k, watch in list(self._watches.items()):
            if k in want_patterns:
                continue
            try:
                await self._stop_watch_handle(watch)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.exception("Failed to stop remote state watch bucket=%s pattern=%s", k[0], k[1])
            self._watches.pop(k, None)
            self._targets.pop(k, None)

        self._targets = dict(want_patterns)
        self._field_filters = {
            (t.service_id, t.node_id): frozenset(str(f).strip() for f in t.fields if str(f).strip())
            for t in want_patterns.values()
        }

        # Start new watches + initial sync for new/changed targets.
        for (bucket, pattern), t in want_patterns.items():
            if (bucket, pattern) not in self._watches:
                async def _cb(key: str, val: bytes, *, _sid: str = t.service_id) -> None:
                    await self._on_kv(_sid, key, val)

                try:
                    self._watches[(bucket, pattern)] = await self._tr.kv_watch_in_bucket(bucket, pattern, cb=_cb)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.exception("Failed to start remote state watch bucket=%s pattern=%s", bucket, pattern)
                    continue

            if (bucket, pattern) not in changed_patterns:
                continue
            # Initial sync per declared field (best-effort).
            for field in list(t.fields or ()):
                f = str(field or "").strip()
                if not f:
                    continue
                try:
                    key = kv_key_node_state(node_id=t.node_id, field=f)
                except (AttributeError, TypeError, ValueError):
                    continue
                try:
                    raw = await self._tr.kv_get_in_bucket(bucket, key)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.exception("Failed to fetch initial remote state bucket=%s key=%s", bucket, key)
                    continue
                if not raw:
                    continue
                await self._on_kv(t.service_id, key, raw)

    async def _on_kv(self, service_id: str, key: str, value: bytes) -> None:
        parsed = parse_kv_key_node_state(key)
        if not parsed:
            return
        node_id, field = parsed
        allowed_fields = self._field_filters.get((str(service_id), str(node_id)))
        if allowed_fields is not None and allowed_fields and str(field) not in allowed_fields:
            return
        try:
            payload = json.loads(value.decode("utf-8")) if value else {}
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            payload = {}
        meta: dict[str, Any] = {}
        if isinstance(payload, dict):
            meta = dict(payload)
            v = payload.get("value")
            ts = _coerce_inbound_ts_ms(_extract_ts_field(payload), default=now_ms())
        else:
            v = payload
            ts = now_ms()

        k = (str(service_id), str(node_id), str(field))
        last = self._last_by_key.get(k)
        if last is not None:
            last_ts, last_v = last
            # If the *value* didn't change, suppress duplicate UI updates even if ts changes.
            # If the value changed, always apply (even when ts goes backwards).
            if v == last_v:
                if int(ts) > int(last_ts):
                    self._last_by_key[k] = (int(ts), last_v)
                return
        self._last_by_key[k] = (int(ts), v)

        try:
            r = self._on_state(str(service_id), str(node_id), str(field), v, int(ts), meta)
            if asyncio.iscoroutine(r):
                await r
        except Exception as exc:
            err_key = (str(service_id), str(node_id), str(field), type(exc).__name__)
            if err_key not in self._callback_error_once:
                self._callback_error_once.add(err_key)
                logger.exception(
                    "Remote state callback failed service_id=%s node_id=%s field=%s",
                    service_id,
                    node_id,
                    field,
                    exc_info=exc,
                )
