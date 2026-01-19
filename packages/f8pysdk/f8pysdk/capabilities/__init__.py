from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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

    async def on_exec(self, ctx_id: str | int, in_port: str | None = None) -> list[str]: ...


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
