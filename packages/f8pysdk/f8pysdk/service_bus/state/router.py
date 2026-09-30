from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ...generated import F8Edge, F8EdgeKindEnum, F8RuntimeGraph, F8StateAccess
from ...f8_naming import ensure_token, parse_state_path_node_field
from ...zenoh_naming import zenoh_key_to_state_path, zenoh_state_key
from ...time_utils import now_ms
from ...runtime_transport import SubscriptionHandle
from ...codec import decode_obj
from ...state import StateWriteContext, StateWriteError, StateWriteOrigin, StateWriteSource
from ..internal.logging import log_error_once
from ._errors import STATE_ROUTE_PROPAGATION_ERRORS, STATE_TRANSPORT_ERRORS, STATE_WATCH_LIFECYCLE_ERRORS
from .helpers import build_cross_state_meta, build_state_validation_meta, coerce_inbound_ts_ms, extract_ts_field
from .options import StatePublishOptions
from .store import StateStore

if TYPE_CHECKING:
    from ..runtime import ServiceBus


StateRouteTarget = tuple[str, str, F8Edge]
StateRouteTable = dict[tuple[str, str], tuple[StateRouteTarget, ...]]
CrossStateBindingKey = tuple[str, str]
CrossStateBindingTable = dict[CrossStateBindingKey, tuple[StateRouteTarget, ...]]

log = logging.getLogger(__name__)

_CROSS_STATE_INITIAL_SYNC_BUDGET_S = 0.30


class StateRouter:
    def __init__(self, bus: "ServiceBus", *, store: StateStore) -> None:
        self._bus = bus
        self._store = store
        self._intra_state_out: StateRouteTable = {}
        self._cross_state_in_by_key: CrossStateBindingTable = {}
        self._remote_state_watches: dict[CrossStateBindingKey, SubscriptionHandle] = {}
        self._remote_latest: dict[CrossStateBindingKey, bytes] = {}
        self._cross_state_targets: set[tuple[str, str]] = set()
        self._cross_state_last_ts: dict[tuple[str, str], int] = {}

    @property
    def intra_state_out(self) -> StateRouteTable:
        return self._intra_state_out

    @property
    def cross_state_targets(self) -> set[tuple[str, str]]:
        return self._cross_state_targets

    def reset_remote_state_ordering(self) -> None:
        self._cross_state_last_ts.clear()

    def clear_intra_state_routes(self) -> None:
        self._intra_state_out.clear()

    def replace_intra_state_routes(self, routes: StateRouteTable) -> None:
        self._intra_state_out = dict(routes)

    def intra_targets(self, *, node_id: str, field: str) -> tuple[StateRouteTarget, ...]:
        return self._intra_state_out.get((str(node_id), str(field))) or ()

    def is_cross_state_target(self, *, node_id: str, field: str) -> bool:
        return (str(node_id), str(field)) in self._cross_state_targets

    def last_remote_ts(self, *, node_id: str, field: str) -> int | None:
        return self._cross_state_last_ts.get((str(node_id), str(field)))

    def record_remote_ts(self, *, node_id: str, field: str, ts_ms: int) -> None:
        self._cross_state_last_ts[(str(node_id), str(field))] = int(ts_ms)

    def update_cross_state_bindings(self, graph: F8RuntimeGraph) -> None:
        want: dict[CrossStateBindingKey, list[StateRouteTarget]] = {}
        targets: set[tuple[str, str]] = set()
        for edge in graph.edges if isinstance(graph.edges, list) else []:
            if edge.kind != F8EdgeKindEnum.state:
                continue
            if str(edge.fromServiceId) == str(edge.toServiceId):
                continue
            if str(edge.toServiceId) != self._bus.service_id:
                continue
            peer = str(edge.fromServiceId or "").strip()
            try:
                peer = ensure_token(peer, label="fromServiceId")
            except ValueError:
                continue

            if not edge.toOperatorId or not edge.fromOperatorId:
                continue

            local_node = str(edge.toOperatorId)
            local_field = str(edge.toPort)
            remote_node = str(edge.fromOperatorId)
            remote_field = str(edge.fromPort)
            remote_key = zenoh_state_key(peer, node_id=remote_node, field=remote_field)
            want.setdefault((peer, remote_key), []).append((local_node, local_field, edge))
            targets.add((local_node, local_field))

        self._cross_state_in_by_key = {key: tuple(values) for key, values in want.items()}
        self._cross_state_targets = targets

    async def stop_unused_watches(self) -> None:
        for key, watch in list(self._remote_state_watches.items()):
            if key in self._cross_state_in_by_key:
                continue
            await self._stop_watch_handle(watch, key=key)
            self._remote_state_watches.pop(key, None)
            self._remote_latest.pop(key, None)

    async def sync_cross_state_watches(self) -> None:
        initial_sync_jobs: list[tuple[str, str]] = []
        for peer, remote_key in self._cross_state_in_by_key.keys():
            if (peer, remote_key) not in self._remote_state_watches:

                async def _cb(key: str, val: bytes, *, _peer: str = peer) -> None:
                    await self.on_remote_state_retained(_peer, key, val, is_initial=False)

                try:
                    self._remote_state_watches[(peer, remote_key)] = await self._bus._transport.retained_watch(
                        remote_key, cb=_cb, with_initial=True
                    )
                except STATE_TRANSPORT_ERRORS as exc:
                    log_error_once(
                        self._bus,
                        key=f"cross_state_watch_start_failed:{peer}:{remote_key}",
                        message=f"failed to start cross-state watch peer={peer} key={remote_key}",
                        exc=exc,
                    )
                    continue

            initial_sync_jobs.append((peer, remote_key))

        if not initial_sync_jobs:
            return

        concurrency = max(1, int(self._bus._state_sync_concurrency))
        sem = asyncio.Semaphore(concurrency)
        tasks: list[asyncio.Task[None]] = []

        async def _sync_one(peer: str, remote_key: str) -> None:
            async with sem:
                try:
                    raw = self._remote_latest.get((peer, remote_key))
                except STATE_TRANSPORT_ERRORS as exc:
                    log_error_once(
                        self._bus,
                        key=f"cross_state_initial_get_failed:{peer}:{remote_key}",
                        message=f"cross-state initial sync read failed peer={peer} key={remote_key}",
                        exc=exc,
                    )
                    return
                if raw:
                    await self.on_remote_state_retained(peer, remote_key, raw, is_initial=True, no_fanout=True)

        for peer, remote_key in initial_sync_jobs:
            task = asyncio.create_task(
                _sync_one(peer, remote_key),
                name=f"service_bus:cross_state_sync:{peer}:{remote_key}",
            )
            tasks.append(task)

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=_CROSS_STATE_INITIAL_SYNC_BUDGET_S,
            )
        except asyncio.TimeoutError:
            pending_count = sum(1 for task in tasks if not task.done())
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            cancelled_count = sum(1 for task in tasks if task.cancelled())
            blocked_count = pending_count or cancelled_count
            if blocked_count > 0:
                log.info(
                    "service_bus[%s] cross-state initial sync budget exhausted jobs=%s pending=%s budget_s=%.3f",
                    self._bus.service_id,
                    len(tasks),
                    blocked_count,
                    _CROSS_STATE_INITIAL_SYNC_BUDGET_S,
                )

    async def on_remote_state_retained(
        self,
        peer_service_id: str,
        key: str,
        value: bytes,
        *,
        is_initial: bool,
        no_fanout: bool = False,
    ) -> None:
        from .pipeline import coerce_state_value, publish_state, validate_state_update

        peer_service_id_s = str(peer_service_id)
        key_s = str(key)
        state_path = zenoh_key_to_state_path(key_s) or key_s
        parsed = parse_state_path_node_field(state_path)
        if not parsed:
            return
        remote_node, remote_field = parsed
        remote_key = zenoh_state_key(peer_service_id_s, node_id=remote_node, field=remote_field)
        targets = self._cross_state_in_by_key.get((peer_service_id_s, remote_key)) or ()
        if not targets:
            return
        self._remote_latest[(peer_service_id_s, remote_key)] = bytes(value)
        try:
            payload = decode_obj(value)
        except ValueError:
            payload = {}
        if isinstance(payload, dict):
            payload_value = payload.get("value")
            ts_ms = coerce_inbound_ts_ms(extract_ts_field(payload), default=now_ms())
        else:
            payload_value = payload
            ts_ms = now_ms()
        ts_i = int(ts_ms)

        if self._bus._debug_state:
            try:
                value_text = repr(payload_value)
                if len(value_text) > 160:
                    value_text = value_text[:157] + "..."
                print(
                    "state_debug[%s] cross_state_in peer=%s key=%s ts=%s initial=%s value=%s targets=%s"
                    % (
                        self._bus.service_id,
                        peer_service_id_s,
                        key_s,
                        str(ts_i),
                        "1" if bool(is_initial) else "0",
                        value_text,
                        str(len(targets)),
                    )
                )
            except (TypeError, ValueError):
                pass

        for local_node_id, local_field, _edge in targets:
            local_node_id_s = str(local_node_id)
            local_field_s = str(local_field)
            access = self._store.access_for(node_id=local_node_id_s, field=local_field_s)
            if access == F8StateAccess.ro:
                continue
            try:
                meta_in = payload if isinstance(payload, dict) else {}

                last_ts = self.last_remote_ts(node_id=local_node_id_s, field=local_field_s)
                if not is_initial and last_ts is not None and ts_i < int(last_ts):
                    if self._bus._debug_state:
                        print(
                            "state_debug[%s] cross_state_skip_old_remote node=%s field=%s ts_last=%s ts_remote=%s peer=%s key=%s"
                            % (
                                self._bus.service_id,
                                local_node_id_s,
                                local_field_s,
                                str(last_ts),
                                str(ts_i),
                                peer_service_id_s,
                                key_s,
                            )
                        )
                    continue

                meta_out = build_cross_state_meta(
                    peer_service_id=peer_service_id_s,
                    remote_key=key_s,
                    inbound_meta=meta_in,
                )
                value2 = await validate_state_update(
                    self._bus,
                    node_id=local_node_id_s,
                    field=local_field_s,
                    value=payload_value,
                    ts_ms=ts_i,
                    meta=build_state_validation_meta(source=StateWriteSource.state_edge_cross, meta=meta_out),
                    ctx=StateWriteContext(origin=StateWriteOrigin.external, source=StateWriteSource.state_edge_cross),
                )
                value2 = coerce_state_value(value2)

                cached = self._store.cache_entry(node_id=local_node_id_s, field=local_field_s)
                try:
                    if cached is not None and ts_i <= int(cached[1]) and cached[0] == value2:
                        if self._bus._debug_state:
                            print(
                                "state_debug[%s] cross_state_skip_duplicate to=%s.%s ts=%s peer=%s remote_key=%s"
                                % (
                                    self._bus.service_id,
                                    local_node_id_s,
                                    local_field_s,
                                    str(ts_i),
                                    peer_service_id_s,
                                    key_s,
                                )
                            )
                        continue
                except (TypeError, ValueError):
                    pass

                if access is None:
                    if self._bus._debug_state:
                        print(
                            "state_debug[%s] cross_state_skip_unknown_field to=%s.%s peer=%s remote_key=%s"
                            % (
                                self._bus.service_id,
                                local_node_id_s,
                                local_field_s,
                                peer_service_id_s,
                                key_s,
                            )
                        )
                    continue
                await publish_state(
                    self._bus,
                    local_node_id_s,
                    local_field_s,
                    value2,
                    ts_ms=ts_i,
                    origin=StateWriteOrigin.external,
                    source=StateWriteSource.state_edge_cross,
                    meta=meta_out,
                    options=StatePublishOptions(fanout_intra_state_edges=not no_fanout),
                )
                self.record_remote_ts(node_id=local_node_id_s, field=local_field_s, ts_ms=ts_i)
            except StateWriteError as exc:
                log_error_once(
                    self._bus,
                    key=f"cross_state_apply_write_error:{local_node_id_s}:{local_field_s}",
                    message=f"cross-state apply rejected for {local_node_id_s}.{local_field_s}",
                    exc=exc,
                )
            except STATE_ROUTE_PROPAGATION_ERRORS as exc:
                log_error_once(
                    self._bus,
                    key=f"cross_state_apply_failed:{local_node_id_s}:{local_field_s}",
                    message=f"cross-state apply failed for {local_node_id_s}.{local_field_s}",
                    exc=exc,
                )

    async def stop(self) -> None:
        self._cross_state_in_by_key.clear()
        self._cross_state_targets.clear()
        self._intra_state_out.clear()
        self._cross_state_last_ts.clear()
        for key, watch in list(self._remote_state_watches.items()):
            await self._stop_watch_handle(watch, key=key)
        self._remote_state_watches.clear()

    async def _stop_watch_handle(self, watch: SubscriptionHandle, *, key: CrossStateBindingKey) -> None:
        try:
            await watch.unsubscribe()
        except STATE_WATCH_LIFECYCLE_ERRORS as exc:
            log_error_once(
                self._bus,
                key=f"cross_state_watch_stop_failed:{key[0]}:{key[1]}",
                message=f"failed to stop cross-state watch peer={key[0]} key={key[1]}",
                exc=exc,
            )


__all__ = [
    "CrossStateBindingKey",
    "CrossStateBindingTable",
    "StateRouteTable",
    "StateRouteTarget",
    "StateRouter",
]
