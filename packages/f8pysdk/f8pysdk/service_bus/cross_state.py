from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

from ..generated import F8Edge, F8EdgeKindEnum, F8StateAccess, F8RuntimeGraph
from ..nats_naming import (
    ensure_token,
    kv_bucket_for_service,
    kv_key_node_state,
    parse_kv_key_node_state,
)
from .payload import coerce_inbound_ts_ms, extract_ts_field
from .error_utils import log_error_once
from .state_publish import coerce_state_value, publish_state, validate_state_update
from .state_write import StateWriteContext, StateWriteError, StateWriteOrigin, StateWriteSource
from ..time_utils import now_ms

if TYPE_CHECKING:
    from .bus import ServiceBus


async def _stop_watch_handle(bus: "ServiceBus", watch: Any, *, key: tuple[str, str]) -> None:
    watcher: Any = None
    task: asyncio.Task[Any] | None = None
    try:
        watcher, task = watch
    except (TypeError, ValueError) as exc:
        log_error_once(
            bus,
            key=f"cross_state_watch_unpack_failed:{key[0]}:{key[1]}",
            message=f"invalid cross-state watch handle for {key}",
            exc=exc,
        )
        return
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log_error_once(
                bus,
                key=f"cross_state_watch_task_stop_failed:{key[0]}:{key[1]}",
                message=f"cross-state watch task stop failed for {key}",
                exc=exc,
            )
    if watcher is not None:
        try:
            await watcher.stop()
        except Exception as exc:
            log_error_once(
                bus,
                key=f"cross_state_watch_stop_failed:{key[0]}:{key[1]}",
                message=f"cross-state watcher stop failed for {key}",
                exc=exc,
            )


def update_cross_state_bindings(bus: "ServiceBus", graph: F8RuntimeGraph) -> None:
    """
    Build cross-state binding tables from the current graph without starting watches.

    Watches are started after rungraph hooks register nodes (see `service_bus.rungraph_apply.apply_rungraph`).
    """
    want: dict[tuple[str, str], list[tuple[str, str, F8Edge]]] = {}
    targets: set[tuple[str, str]] = set()
    for edge in graph.edges:
        if edge.kind != F8EdgeKindEnum.state:
            continue
        if str(edge.fromServiceId) == str(edge.toServiceId):
            continue
        if str(edge.toServiceId) != bus.service_id:
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
        remote_key = kv_key_node_state(node_id=remote_node, field=remote_field)
        want.setdefault((peer, remote_key), []).append((local_node, local_field, edge))
        targets.add((local_node, local_field))

    bus._cross_state_in_by_key = want
    bus._cross_state_targets = targets


async def stop_unused_cross_state_watches(bus: "ServiceBus") -> None:
    """
    Stop KV watches that are no longer needed based on current bindings.
    """
    want = bus._cross_state_in_by_key
    for k, watch in list(bus._remote_state_watches.items()):
        if k in want:
            continue
        await _stop_watch_handle(bus, watch, key=k)
        bus._remote_state_watches.pop(k, None)


async def sync_cross_state_watches(bus: "ServiceBus") -> None:
    """
    Cross-state binding via remote KV watch (read remote, apply to local).
    """
    # NOTE: bindings are computed in `ServiceBus._rebuild_routes` so `stateValues` application
    # can skip cross-state target fields.
    want = bus._cross_state_in_by_key

    initial_sync_jobs: list[tuple[str, str, str]] = []
    # Start missing watches and perform best-effort initial sync for every watched key.
    for peer, remote_key in want.keys():
        bucket = kv_bucket_for_service(peer)

        if (peer, remote_key) not in bus._remote_state_watches:

            async def _cb(key: str, val: bytes, *, _peer: str = peer) -> None:
                await on_remote_state_kv(bus, _peer, key, val, is_initial=False)

            try:
                bus._remote_state_watches[(peer, remote_key)] = await bus._transport.kv_watch_in_bucket(
                    bucket, remote_key, cb=_cb
                )
            except Exception as exc:
                log_error_once(
                    bus,
                    key=f"cross_state_watch_start_failed:{peer}:{remote_key}",
                    message=f"failed to start cross-state watch peer={peer} key={remote_key}",
                    exc=exc,
                )
                continue

        initial_sync_jobs.append((peer, bucket, remote_key))

    if not initial_sync_jobs:
        return

    concurrency = max(1, int(bus._state_sync_concurrency))
    sem = asyncio.Semaphore(concurrency)
    tasks: list[asyncio.Task[None]] = []

    async def _sync_one(peer: str, bucket: str, remote_key: str) -> None:
        async with sem:
            # Initial sync: fetch current value once so new targets receive the current value
            # even if the upstream doesn't change after deploy.
            try:
                raw = await bus._transport.kv_get_in_bucket(bucket, remote_key)
            except Exception as exc:
                log_error_once(
                    bus,
                    key=f"cross_state_initial_get_failed:{peer}:{remote_key}",
                    message=f"cross-state initial sync read failed peer={peer} key={remote_key}",
                    exc=exc,
                )
                return
            if raw:
                # Phase 1 (strong sync): apply remote values without triggering
                # intra-service fanout yet. After all cross-state targets are
                # materialized, we run a single ordered intra init sync.
                await on_remote_state_kv(bus, peer, remote_key, raw, is_initial=True, no_fanout=True)

    for peer, bucket, remote_key in initial_sync_jobs:
        task = asyncio.create_task(
            _sync_one(peer, bucket, remote_key),
            name=f"service_bus:cross_state_sync:{peer}:{remote_key}",
        )
        tasks.append(task)

    await asyncio.gather(*tasks)


async def on_remote_state_kv(
    bus: "ServiceBus",
    peer_service_id: str,
    key: str,
    value: bytes,
    *,
    is_initial: bool,
    no_fanout: bool = False,
) -> None:
    peer_service_id_s = str(peer_service_id)
    key_s = str(key)
    parsed = parse_kv_key_node_state(key)
    if not parsed:
        return
    remote_node, remote_field = parsed
    remote_key = kv_key_node_state(node_id=remote_node, field=remote_field)
    targets = bus._cross_state_in_by_key.get((peer_service_id_s, remote_key)) or []
    if not targets:
        return
    try:
        payload = json.loads(value.decode("utf-8")) if value else {}
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        payload = {}
    if isinstance(payload, dict):
        v = payload.get("value")
        ts = coerce_inbound_ts_ms(extract_ts_field(payload), default=now_ms())
    else:
        v = payload
        ts = now_ms()
    ts_i = int(ts)

    if bus._debug_state:
        try:
            v_s = repr(v)
            if len(v_s) > 160:
                v_s = v_s[:157] + "..."
            print(
                "state_debug[%s] cross_state_in peer=%s key=%s ts=%s initial=%s value=%s targets=%s"
                % (
                    bus.service_id,
                    peer_service_id_s,
                    key_s,
                    str(ts_i),
                    "1" if bool(is_initial) else "0",
                    v_s,
                    str(len(targets)),
                )
            )
        except (TypeError, ValueError):
            pass

    for local_node_id, local_field, _edge in targets:
        local_node_id_s = str(local_node_id)
        local_field_s = str(local_field)
        local_key = (local_node_id_s, local_field_s)
        access = bus._state_access_by_node_field.get(local_key)
        if access == F8StateAccess.ro:
            continue
        try:
            meta_in = payload if isinstance(payload, dict) else {}

            # Cross-service state edges are directional bindings: downstream follows
            # upstream. We only guard against out-of-order remote updates.
            last_ts = bus._cross_state_last_ts.get(local_key)
            if not is_initial and last_ts is not None and ts_i < int(last_ts):
                if bus._debug_state:
                    print(
                        "state_debug[%s] cross_state_skip_old_remote node=%s field=%s ts_last=%s ts_remote=%s peer=%s key=%s"
                        % (
                            bus.service_id,
                            local_node_id_s,
                            local_field_s,
                            str(last_ts),
                            str(ts_i),
                            peer_service_id_s,
                            key_s,
                        )
                    )
                continue

            meta_out = {
                "peerServiceId": peer_service_id_s,
                "remoteKey": key_s,
                "_noStateFanout": True if bool(no_fanout) else False,
                **{k: vv for k, vv in dict(meta_in).items() if k not in ("value", "actor", "ts", "source")},
            }
            if not bool(no_fanout):
                meta_out.pop("_noStateFanout", None)
            v2 = await validate_state_update(
                bus,
                node_id=local_node_id_s,
                field=local_field_s,
                value=v,
                ts_ms=ts_i,
                meta={"source": StateWriteSource.state_edge_cross.value, **meta_out},
                ctx=StateWriteContext(origin=StateWriteOrigin.external, source=StateWriteSource.state_edge_cross),
            )
            v2 = coerce_state_value(v2)

            # Skip duplicates (common when KV watch emits the current value and we
            # also perform an explicit initial `kv_get`).
            try:
                cached = bus._state_cache.get(local_key)
                if cached is not None and ts_i <= int(cached[1]) and cached[0] == v2:
                    if bus._debug_state:
                        print(
                            "state_debug[%s] cross_state_skip_duplicate to=%s.%s ts=%s peer=%s remote_key=%s"
                            % (
                                bus.service_id,
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
                if bus._debug_state:
                    print(
                        "state_debug[%s] cross_state_skip_unknown_field to=%s.%s peer=%s remote_key=%s"
                        % (
                            bus.service_id,
                            local_node_id_s,
                            local_field_s,
                            peer_service_id_s,
                            key_s,
                        )
                    )
                continue
            await publish_state(
                bus,
                local_node_id_s,
                local_field_s,
                v2,
                ts_ms=ts_i,
                origin=StateWriteOrigin.external,
                source=StateWriteSource.state_edge_cross,
                meta=meta_out,
            )
            if bus._debug_state:
                v2s = repr(v2)
                if len(v2s) > 160:
                    v2s = v2s[:157] + "..."
                print(
                    "state_debug[%s] cross_state_apply to=%s.%s ts=%s peer=%s remote_key=%s value=%s"
                    % (
                        bus.service_id,
                        local_node_id_s,
                        local_field_s,
                        str(ts_i),
                        peer_service_id_s,
                        key_s,
                        v2s,
                    )
                )
            bus._cross_state_last_ts[local_key] = ts_i
        except StateWriteError as exc:
            log_error_once(
                bus,
                key=f"cross_state_rejected:{local_node_id_s}:{local_field_s}",
                message=f"cross-state update rejected for {local_node_id_s}.{local_field_s}",
                exc=exc,
            )
        except Exception as exc:
            log_error_once(
                bus,
                key=f"cross_state_apply_failed:{local_node_id_s}:{local_field_s}",
                message=f"cross-state apply failed for {local_node_id_s}.{local_field_s}",
                exc=exc,
            )
