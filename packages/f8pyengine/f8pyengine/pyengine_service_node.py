from __future__ import annotations

from typing import Any, Protocol, cast

from f8pysdk.nats_naming import ensure_token
from f8pysdk.generated import F8RuntimeNode
from f8pysdk.runtime_node import ServiceNode


class _DataDeliveryBus(Protocol):
    def set_data_delivery(self, value: Any, *, source: str = "service") -> None: ...


def _coerce_data_delivery(value: Any) -> str | None:
    v = str(value or "").strip().lower()
    if v in ("pull", "push", "both"):
        return v
    return None


class PyEngineServiceNode(ServiceNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=[],
            state_fields=[s.name for s in list(node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name != "dataDelivery":
            return value
        mode = _coerce_data_delivery(value)
        if mode is None:
            raise ValueError("invalid dataDelivery (expected pull, push, or both)")
        return mode

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name != "dataDelivery":
            return
        raw = await self.get_state_value("dataDelivery")
        if raw is None:
            raw = value
        if raw is None:
            raw = self._initial_state.get("dataDelivery")
        mode = _coerce_data_delivery(raw)
        if mode is None:
            return
        if self._bus is None:
            return
        try:
            cast(_DataDeliveryBus, self._bus).set_data_delivery(mode, source="state")
        except Exception:
            return
