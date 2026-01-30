from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from .capabilities import BusAttachableNode, ComputableNode, DataReceivableNode, LifecycleNode, NodeBus, StatefulNode


@dataclass
class RuntimeNode(BusAttachableNode, StatefulNode, DataReceivableNode, ComputableNode, LifecycleNode):
    """
    Base class for service runtime nodes.

    This is NOT a UI node. It's the runtime-side abstraction that receives
    inputs from intra/cross edges and emits outputs (fanout handled by runtime).

    Capabilities:
    - `BusAttachableNode` (attach to `ServiceBus`)
    - `StatefulNode` (optional state callback)
    - `ComputableNode` (optional pull-based compute)
    """

    node_id: str
    data_in_ports: list[str] = field(default_factory=list)
    data_out_ports: list[str] = field(default_factory=list)
    state_fields: list[str] = field(default_factory=list)

    _bus: NodeBus | None = field(default=None, init=False, repr=False)

    # ---- lifecycle ------------------------------------------------------
    def attach(self, bus: Any) -> None:
        self._bus = cast(NodeBus, bus)

    # ---- inbound --------------------------------------------------------
    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        """
        Override in subclasses.
        """
        return

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        """
        Push-based data callback (optional).
        """
        return

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        """
        Optional lifecycle callback (activate/deactivate).
        """
        return

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        """
        Pull-based output computation hook (optional).

        A consumer may trigger computation by pulling an input whose buffer is empty.
        The runtime may then call `compute_output()` on upstream nodes to produce the
        needed values.
        """
        return None

    # ---- outbound -------------------------------------------------------
    async def emit(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if self._bus is None:
            return
        await self._bus.emit_data(self.node_id, port, value, ts_ms=ts_ms)

    async def pull(self, port: str, *, ctx_id: str | int | None = None) -> Any:
        """
        Pull an input value for the given port from the runtime buffers.
        """
        if self._bus is None:
            return None
        return await self._bus.pull_data(self.node_id, port, ctx_id=ctx_id)

    async def set_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        if self._bus is None:
            return
        await self._bus.set_state(self.node_id, field, value, ts_ms=ts_ms)

    async def get_state(self, field: str) -> Any:
        if self._bus is None:
            return None
        return await self._bus.get_state(self.node_id, field)


@dataclass
class ServiceNode(RuntimeNode):
    """
    Marker base class for service/container nodes.

    Service nodes typically expose lifecycle/commands/state and may provide data outputs.
    """


@dataclass
class OperatorNode(RuntimeNode):
    """
    Marker base class for operator nodes.

    Operator nodes are the executable/functional units within a service graph.
    """
