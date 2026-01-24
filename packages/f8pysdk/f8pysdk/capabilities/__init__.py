from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING
from collections.abc import Awaitable, Callable

if TYPE_CHECKING:
    from ..generated import F8RuntimeGraph


@runtime_checkable
class BusAttachableNode(Protocol):
    """
    ServiceBus-attachable node (capability).

    This is the minimal interface required by `ServiceBus.register_node(...)`.
    """

    node_id: str

    def attach(self, bus: Any) -> None: ...


@runtime_checkable
class ExecutableNode(Protocol):
    """
    Exec-capable node behavior (capability).

    Nodes that participate in exec-flow should implement this interface.
    """

    node_id: str

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]: ...


@runtime_checkable
class EntrypointNode(ExecutableNode, Protocol):
    """
    Exec entrypoint capability (source node).

    This is for nodes that *initiate* exec flow (timer/event based), managed by the engine.
    """

    async def start_entrypoint(self, ctx: Any) -> None: ...

    async def stop_entrypoint(self) -> None: ...


@runtime_checkable
class StatefulNode(Protocol):
    """
    State callback capability.

    Nodes can opt-in to receiving state updates.
    """

    node_id: str

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None: ...


@runtime_checkable
class DataReceivableNode(Protocol):
    """
    Push-based data callback capability.

    Nodes can opt-in to receiving data inputs via `ServiceBus` push delivery.
    """

    node_id: str

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None: ...


@runtime_checkable
class ComputableNode(Protocol):
    """
    Pull-based computation capability.

    Nodes can opt-in to computing outputs on demand during a pull.
    """

    node_id: str

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any: ...


@runtime_checkable
class CommandableNode(Protocol):
    """
    Command handling capability.

    Intended for exposing user-defined commands via a unified command endpoint.
    """

    node_id: str

    async def on_command(self, name: str, args: dict[str, Any] | None = None, *, meta: dict[str, Any] | None = None) -> Any: ...


@runtime_checkable
class ClosableNode(Protocol):
    """
    Optional lifecycle capability for resources (subscriptions/tasks).
    """

    async def close(self) -> None: ...


@runtime_checkable
class DataListenerBus(Protocol):
    """
    Data listener registration capability (bus-side).

    Consumers (e.g. UI tools) can subscribe to buffered input updates for a given (node_id, port).
    """

    def add_data_listener(self, node_id: str, port: str, cb: Callable[[str, str, Any, int], Awaitable[None] | None]) -> None: ...

    def remove_data_listener(self, node_id: str, port: str, cb: Callable[[str, str, Any, int], Awaitable[None] | None]) -> None: ...


@runtime_checkable
class StateListenerBus(Protocol):
    """
    State listener registration capability (bus-side).

    Consumers can subscribe to local KV state updates.
    """

    def add_state_listener(
        self, cb: Callable[[str, str, Any, int, dict[str, Any]], Awaitable[None] | None]
    ) -> None: ...

    def remove_state_listener(
        self, cb: Callable[[str, str, Any, int, dict[str, Any]], Awaitable[None] | None]
    ) -> None: ...


@runtime_checkable
class RungraphListenerBus(Protocol):
    """
    Rungraph listener registration capability (bus-side).

    Consumers can subscribe to validated rungraph updates.
    """

    def add_rungraph_listener(self, cb: Callable[["F8RuntimeGraph"], Awaitable[None] | None]) -> None: ...

    def remove_rungraph_listener(self, cb: Callable[["F8RuntimeGraph"], Awaitable[None] | None]) -> None: ...


@runtime_checkable
class LifecycleListenerBus(Protocol):
    """
    Lifecycle listener registration capability (bus-side).

    Consumers can subscribe to activate/deactivate transitions.
    """

    def add_lifecycle_listener(self, cb: Callable[[bool, dict[str, Any]], Awaitable[None] | None]) -> None: ...

    def remove_lifecycle_listener(self, cb: Callable[[bool, dict[str, Any]], Awaitable[None] | None]) -> None: ...


@runtime_checkable
class BusActive(Protocol):
    """
    Exposes current bus active state.
    """

    @property
    def active(self) -> bool: ...


@runtime_checkable
class DataIOBus(Protocol):
    """
    Data input/output operations against the bus.
    """

    async def emit_data(self, node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None: ...

    async def pull_data(self, node_id: str, port: str, *, ctx_id: str | int | None = None) -> Any: ...


@runtime_checkable
class StateIOBus(Protocol):
    """
    State read/write operations against the bus.
    """

    async def set_state(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None: ...

    async def get_state(self, node_id: str, field: str) -> Any: ...


@runtime_checkable
class NodeBus(StateIOBus, DataIOBus, LifecycleListenerBus, BusActive, Protocol):
    """
    Composition of bus capabilities used by `RuntimeNode`.
    """
