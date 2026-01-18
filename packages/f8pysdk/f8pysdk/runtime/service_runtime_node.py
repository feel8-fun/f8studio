from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class _RuntimeLike(Protocol):
    async def emit_data(self, node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None: ...

    async def pull_data(self, node_id: str, port: str, *, ctx_id: str | int | None = None) -> Any: ...

    async def set_state(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None: ...

    async def get_state(self, node_id: str, field: str) -> Any: ...


@dataclass
class ServiceRuntimeNode:
    """
    Base class for service runtime nodes.

    This is NOT a UI node. It's the runtime-side abstraction that receives
    inputs from intra/cross edges and emits outputs (fanout handled by runtime).
    """

    node_id: str
    data_in_ports: list[str] = field(default_factory=list)
    data_out_ports: list[str] = field(default_factory=list)
    state_fields: list[str] = field(default_factory=list)

    _runtime: _RuntimeLike | None = field(default=None, init=False, repr=False)

    # ---- lifecycle ------------------------------------------------------
    def attach(self, runtime: _RuntimeLike) -> None:
        self._runtime = runtime

    # ---- inbound --------------------------------------------------------
    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        """
        Override in subclasses.
        """
        return

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        """
        Override in subclasses.
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
        if self._runtime is None:
            return
        await self._runtime.emit_data(self.node_id, port, value, ts_ms=ts_ms)

    async def pull(self, port: str, *, ctx_id: str | int | None = None) -> Any:
        """
        Pull an input value for the given port from the runtime buffers.
        """
        if self._runtime is None:
            return None
        return await self._runtime.pull_data(self.node_id, port, ctx_id=ctx_id)

    async def set_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        if self._runtime is None:
            return
        await self._runtime.set_state(self.node_id, field, value, ts_ms=ts_ms)

    async def get_state(self, field: str) -> Any:
        if self._runtime is None:
            return None
        return await self._runtime.get_state(self.node_id, field)

