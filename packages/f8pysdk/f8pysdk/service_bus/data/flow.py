from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ...generated import F8Edge
from .emit import DataEmitOptions
from .router import DataRouter

if TYPE_CHECKING:
    from ..runtime import ServiceBus


def precreate_input_buffers_for_cross_in(
    bus: "ServiceBus",
    cross_in: dict[str, tuple[tuple[str, str, F8Edge], ...]],
) -> None:
    bus.data_router.precreate_input_buffers_for_cross_in(cross_in)


async def emit_data(
    bus: "ServiceBus",
    node_id: str,
    port: str,
    value: Any,
    *,
    ts_ms: int | None = None,
    options: DataEmitOptions | None = None,
) -> None:
    await bus.data_router.emit_data(node_id, port, value, ts_ms=ts_ms, options=options)


async def pull_data(bus: "ServiceBus", node_id: str, port: str, *, ctx_id: str | int | None = None) -> Any:
    return await bus.data_router.pull_data(node_id, port, ctx_id=ctx_id)


async def ensure_input_available(bus: "ServiceBus", *, node_id: str, port: str, ctx_id: str | int | None = None) -> None:
    await bus.data_router.ensure_input_available(node_id=node_id, port=port, ctx_id=ctx_id)


async def compute_and_buffer_for_input(
    bus: "ServiceBus",
    *,
    node_id: str,
    port: str,
    ctx_id: str | int | None,
    stack: set[tuple[str, str]],
) -> None:
    await bus.data_router.compute_and_buffer_for_input(node_id=node_id, port=port, ctx_id=ctx_id, stack=stack)


async def on_cross_data_msg(bus: "ServiceBus", key: str, payload: bytes) -> None:
    await bus.data_router.on_cross_data_msg(key, payload)


def is_stale(edge: F8Edge | None, ts_ms: int) -> bool:
    return DataRouter.is_stale(edge, ts_ms)


def push_input(bus: "ServiceBus", to_node: str, to_port: str, value: Any, *, ts_ms: int, edge: F8Edge | None = None) -> None:
    bus.data_router.push_input(to_node, to_port, value, ts_ms=ts_ms, edge=edge)


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
    bus.data_router.buffer_input(
        to_node,
        to_port,
        value,
        ts_ms=ts_ms,
        edge=edge,
        ctx_id=ctx_id,
    )


async def sync_subscriptions(bus: "ServiceBus", want_keys: set[str]) -> None:
    await bus.data_router.sync_subscriptions(want_keys)


async def subscribe_key(
    bus: "ServiceBus",
    key_expr: str,
    *,
    queue: str | None = None,
    cb: Callable[[str, bytes], Awaitable[None]] | None = None,
) -> Any:
    return await bus.data_router.subscribe_key(key_expr, queue=queue, cb=cb)


async def unsubscribe_key(bus: "ServiceBus", handle: Any) -> None:
    await bus.data_router.unsubscribe_key(handle)
