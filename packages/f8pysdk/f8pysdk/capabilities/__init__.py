from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..generated import F8RuntimeGraph
    from ..service_bus.bus import ServiceBus
    from ..service_bus.state_read import StateRead


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

    Nodes must implement state validation and callbacks.
    """

    node_id: str

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any: ...

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
class LifecycleNode(Protocol):
    """
    Node lifecycle callback capability.

    Nodes can opt-in to receiving service activate/deactivate transitions.
    """

    node_id: str

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None: ...


@runtime_checkable
class RungraphHook(Protocol):
    """
    Rungraph update hook (component-side).
    """

    async def validate_rungraph(self, graph: "F8RuntimeGraph") -> None: ...

    async def on_rungraph(self, graph: "F8RuntimeGraph") -> None: ...


@runtime_checkable
class ServiceHook(Protocol):
    """
    Service bus hooks (component-side).

    This unifies readiness + activation/deactivation into one capability.
    """

    async def on_before_ready(self, bus: "ServiceBus") -> None: ...

    async def on_after_ready(self, bus: "ServiceBus") -> None: ...

    async def on_before_stop(self, bus: "ServiceBus") -> None: ...

    async def on_after_stop(self, bus: "ServiceBus") -> None: ...

    async def on_activate(self, bus: "ServiceBus", meta: dict[str, Any]) -> None: ...

    async def on_deactivate(self, bus: "ServiceBus", meta: dict[str, Any]) -> None: ...


class ServiceHookBase:
    """
    Convenience base class so hook implementers can override only what they need.
    """

    async def on_before_ready(self, _bus: "ServiceBus") -> None:
        return

    async def on_after_ready(self, _bus: "ServiceBus") -> None:
        return

    async def on_before_stop(self, _bus: "ServiceBus") -> None:
        return

    async def on_after_stop(self, _bus: "ServiceBus") -> None:
        return

    async def on_activate(self, _bus: "ServiceBus", _meta: dict[str, Any]) -> None:
        return

    async def on_deactivate(self, _bus: "ServiceBus", _meta: dict[str, Any]) -> None:
        return


@runtime_checkable
class RungraphHookBus(Protocol):
    """
    Rungraph hook registration capability (bus-side).
    """

    def register_rungraph_hook(self, hook: RungraphHook) -> None: ...

    def unregister_rungraph_hook(self, hook: RungraphHook) -> None: ...

@runtime_checkable
class ServiceHookBus(Protocol):
    """
    Service hook registration capability (bus-side).
    """

    def register_service_hook(self, hook: ServiceHook) -> None: ...

    def unregister_service_hook(self, hook: ServiceHook) -> None: ...


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

    async def publish_state_runtime(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None: ...

    async def get_state(self, node_id: str, field: str) -> "StateRead": ...

    def get_state_cached(self, node_id: str, field: str, default: Any = None) -> Any: ...


@runtime_checkable
class NodeBus(StateIOBus, DataIOBus, BusActive, Protocol):
    """
    Composition of bus capabilities used by `RuntimeNode`.
    """
