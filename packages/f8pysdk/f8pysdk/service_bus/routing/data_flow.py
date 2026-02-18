from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from ...capabilities import ComputableNode, DataReceivableNode
from ...generated import F8Edge, F8EdgeStrategyEnum
from ...nats_naming import data_subject, ensure_token
from ..error_utils import log_error_once
from ...time_utils import now_ms

if TYPE_CHECKING:
    from ..api.bus import ServiceBus


log = logging.getLogger(__name__)


@dataclass
class _InputBuffer:
    to_node: str
    to_port: str
    edge: F8Edge | None
    queue: deque[tuple[Any, int]] = field(default_factory=deque)
    last_seen_value: Any = None
    last_seen_ts: int | None = None
    last_seen_ctx_id: str | int | None = None
    last_pulled_value: Any = None
    last_pulled_ts: int | None = None
    last_pulled_ctx_id: str | int | None = None


def _ensure_input_buffer(
    bus: "ServiceBus",
    *,
    to_node: str,
    to_port: str,
    edge: F8Edge | None,
) -> _InputBuffer:
    key = (str(to_node), str(to_port))
    buf = bus._data_inputs.get(key)
    if buf is None:
        buf = _InputBuffer(to_node=str(to_node), to_port=str(to_port), edge=edge)
        bus._data_inputs[key] = buf
    if edge is not None:
        buf.edge = edge
    return buf


def precreate_input_buffers_for_cross_in(bus: "ServiceBus", cross_in: dict[str, list[tuple[str, str, F8Edge]]]) -> None:
    for _subject, targets in cross_in.items():
        for to_node, to_port, edge in targets:
            _ensure_input_buffer(
                bus,
                to_node=str(to_node),
                to_port=str(to_port),
                edge=edge,
            )


async def emit_data(bus: "ServiceBus", node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None:
    if not bus._active:
        return
    node_id = ensure_token(node_id, label="node_id")
    port = ensure_token(port, label="port_id")
    ts = int(ts_ms or now_ms())

    # Intra edges.
    for to_node, to_port in bus._intra_data_out.get((node_id, port), []):
        push_input(bus, to_node, to_port, value, ts_ms=ts)

    # Cross edges (fan-out) - publish once per (node, out_port).
    if bus._publish_all_data:
        subject = data_subject(bus.service_id, from_node_id=node_id, port_id=port)
    else:
        subject = bus._cross_out_subjects.get((node_id, port)) or ""
    if not subject:
        return
    payload = json.dumps({"value": value, "ts": ts}, ensure_ascii=False, default=str).encode("utf-8")
    await bus._transport.publish(subject, payload)


async def pull_data(bus: "ServiceBus", node_id: str, port: str, *, ctx_id: str | int | None = None) -> Any:
    """
    Pull-based access to buffered inputs.

    Strategy semantics:
    - `latest`: return newest and clear the buffer.
    - `queue`: pop the oldest buffered item (FIFO).
    - `timeoutMs`: if newest sample is stale, return None.
    """
    if not bus._active:
        return None
    node_id = ensure_token(node_id, label="node_id")
    port = ensure_token(port, label="port_id")
    buf = _ensure_input_buffer(bus, to_node=node_id, to_port=port, edge=None)
    edge = buf.edge
    _now_ms = now_ms()

    last_seen_ts = int(buf.last_seen_ts or _now_ms)
    if is_stale(edge, last_seen_ts):
        return None

    strategy = edge.strategy if edge is not None else F8EdgeStrategyEnum.latest
    if not isinstance(strategy, F8EdgeStrategyEnum):
        strategy = F8EdgeStrategyEnum.latest

    if strategy == F8EdgeStrategyEnum.queue:
        if not buf.queue:
            if ctx_id is None or buf.last_seen_ctx_id != ctx_id:
                await ensure_input_available(bus, node_id=node_id, port=port, ctx_id=ctx_id)
            if not buf.queue:
                return None
        v, ts = buf.queue.popleft()
        buf.last_pulled_value = v
        buf.last_pulled_ts = int(ts) if ts is not None else _now_ms
        buf.last_pulled_ctx_id = ctx_id
        return v

    # latest
    if not buf.queue and (ctx_id is None or buf.last_seen_ctx_id != ctx_id):
        await ensure_input_available(bus, node_id=node_id, port=port, ctx_id=ctx_id)
    v = buf.last_seen_value
    buf.queue.clear()
    if v is not None:
        buf.last_pulled_value = v
        buf.last_pulled_ts = _now_ms
        buf.last_pulled_ctx_id = ctx_id
    return v


async def ensure_input_available(bus: "ServiceBus", *, node_id: str, port: str, ctx_id: str | int | None = None) -> None:
    """
    Best-effort intra-service pull-triggered computation.

    If (node_id, port) has no buffered samples and the rungraph defines intra data
    edges feeding it, attempt to compute upstream outputs and buffer the results.
    """
    if not bus._graph:
        return

    upstream = bus._intra_data_in.get((str(node_id), str(port))) or []
    if not upstream:
        return

    stack: set[tuple[str, str]] = set()
    await compute_and_buffer_for_input(bus, node_id=str(node_id), port=str(port), ctx_id=ctx_id, stack=stack)


async def compute_and_buffer_for_input(
    bus: "ServiceBus",
    *,
    node_id: str,
    port: str,
    ctx_id: str | int | None,
    stack: set[tuple[str, str]],
) -> None:
    key = (str(node_id), str(port))
    if key in stack:
        return
    stack.add(key)
    try:
        for from_node, from_port, edge in list(bus._intra_data_in.get(key) or []):
            from_node_s = str(from_node)
            from_port_s = str(from_port)
            src = bus._nodes.get(from_node_s)
            if src is None:
                continue
            try:
                if isinstance(src, ComputableNode):
                    v = await src.compute_output(from_port_s, ctx_id=ctx_id)
                else:
                    v = None
            except Exception as exc:
                log_error_once(
                    bus,
                    key=f"compute_output_failed:{from_node_s}:{from_port_s}",
                    message=f"compute_output failed for {from_node_s}.{from_port_s}",
                    exc=exc,
                )
                continue
            if v is None:
                continue
            # Treat pull-triggered computation as producing a real output sample:
            # route it through `emit_data` so intra edges get buffered and any
            # cross-service subscribers can also receive the computed value.
            try:
                await emit_data(bus, from_node_s, from_port_s, v, ts_ms=now_ms())
            except Exception as exc:
                log_error_once(
                    bus,
                    key=f"emit_data_failed:{from_node_s}:{from_port_s}",
                    message=f"emit_data failed for {from_node_s}.{from_port_s}; using local fallback buffer",
                    exc=exc,
                )
                # Fallback: still satisfy the local pull.
                buffer_input(
                    bus,
                    str(node_id),
                    str(port),
                    v,
                    ts_ms=now_ms(),
                    edge=edge,
                    ctx_id=ctx_id,
                )
    finally:
        stack.discard(key)


async def on_cross_data_msg(bus: "ServiceBus", subject: str, payload: bytes) -> None:
    if not bus._active:
        return
    targets = bus._cross_in_by_subject.get(str(subject)) or []
    if not targets:
        return
    value: Any = None
    ts: int | None = None
    try:
        msg = json.loads(payload.decode("utf-8")) if payload else {}
        if isinstance(msg, dict):
            value = msg.get("value")
            ts = msg.get("ts")
        else:
            value = msg
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        value = payload

    ts_i = int(ts) if ts is not None else now_ms()
    for to_node, to_port, edge in targets:
        try:
            if is_stale(edge, ts_i):
                continue
            push_input(bus, to_node, to_port, value, ts_ms=ts_i, edge=edge)
        except Exception as exc:
            log_error_once(
                bus,
                key=f"cross_data_push_failed:{to_node}:{to_port}",
                message=f"cross-data delivery failed for {to_node}.{to_port}",
                exc=exc,
            )
            continue


def is_stale(edge: F8Edge | None, ts_ms: int) -> bool:
    if edge is None:
        return False
    try:
        timeout = edge.timeoutMs
        if timeout is None:
            return False
        t = int(timeout)
        if t <= 0:
            return False
        return (now_ms() - int(ts_ms)) > t
    except (AttributeError, TypeError, ValueError):
        return False


def push_input(bus: "ServiceBus", to_node: str, to_port: str, value: Any, *, ts_ms: int, edge: F8Edge | None = None) -> None:
    # Data inputs are buffered. We intentionally do not invoke `node.on_data`:
    # the system can be configured for pull-based or push-based data delivery.
    buffer_input(
        bus,
        to_node=str(to_node),
        to_port=str(to_port),
        value=value,
        ts_ms=int(ts_ms),
        edge=edge,
        ctx_id=None,
    )
    if bus._data_delivery in ("push", "both"):
        node = bus._nodes.get(str(to_node))
        if node is not None:
            try:
                if isinstance(node, DataReceivableNode):
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        node.on_data(str(to_port), value, ts_ms=int(ts_ms)),  # type: ignore[misc]
                        name=f"service_bus:on_data:{to_node}:{to_port}",
                    )
            except Exception as exc:
                log_error_once(
                    bus,
                    key=f"push_on_data_schedule_failed:{to_node}:{to_port}",
                    message=f"failed to schedule on_data for {to_node}.{to_port}",
                    exc=exc,
                )


def buffer_input(
    bus: "ServiceBus",
    to_node: str,
    to_port: str,
    value: Any,
    *,
    ts_ms: int,
    edge: F8Edge | None,
    ctx_id: str | int | None,
) -> None:
    to_node = str(to_node)
    to_port = str(to_port)
    buf = _ensure_input_buffer(bus, to_node=to_node, to_port=to_port, edge=edge)

    buf.last_seen_value = value
    buf.last_seen_ts = int(ts_ms)
    buf.last_seen_ctx_id = ctx_id

    buf.queue.append((value, int(ts_ms)))
    max_n = int(bus._data_input_default_queue_size)
    if buf.edge is not None:
        try:
            max_n = max(1, int(buf.edge.queueSize))
        except (AttributeError, TypeError, ValueError):
            max_n = int(bus._data_input_default_queue_size)
    if len(buf.queue) > max_n:
        while len(buf.queue) > max_n:
            buf.queue.popleft()

    return


async def sync_subscriptions(bus: "ServiceBus", want_subjects: set[str]) -> None:
    for subject in list(bus._data_route_subs.keys()):
        if subject in want_subjects:
            continue
        sub = bus._data_route_subs.pop(subject, None)
        if sub is None:
            continue
        try:
            await sub.unsubscribe()
        except Exception as exc:
            log.error("failed to unsubscribe routed subject=%s", subject, exc_info=exc)

    for subject in want_subjects:
        if subject in bus._data_route_subs:
            continue

        async def _cb(s: str, p: bytes) -> None:
            await on_cross_data_msg(bus, s, p)

        handle = await bus._transport.subscribe(subject, cb=_cb)
        bus._data_route_subs[subject] = handle


async def subscribe_subject(
    bus: "ServiceBus",
    subject: str,
    *,
    queue: str | None = None,
    cb: Callable[[str, bytes], Awaitable[None]] | None = None,
) -> Any:
    """
    Subscribe to an arbitrary NATS subject (not tied to rungraph routing).

    Returns the subscription handle (must be unsubscribed by caller or via `unsubscribe_subject`).
    """
    subject = str(subject or "").strip()
    if not subject:
        raise ValueError("subject must be non-empty")
    handle = await bus._transport.subscribe(subject, queue=str(queue) if queue else None, cb=cb)
    bus._custom_subs.append(handle)
    return handle


async def unsubscribe_subject(bus: "ServiceBus", handle: Any) -> None:
    if handle is None:
        return

    await handle.unsubscribe()

    if handle in bus._custom_subs:
        bus._custom_subs.remove(handle)
