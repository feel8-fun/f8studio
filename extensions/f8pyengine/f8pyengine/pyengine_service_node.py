from __future__ import annotations

import logging
from typing import Any, Protocol, cast

from f8pysdk.f8_naming import ensure_token
from f8pysdk.specs import F8RuntimeNode
from f8pysdk.nodes import ServiceNode

logger = logging.getLogger(__name__)


class _DataDeliveryBus(Protocol):
    def set_data_delivery(self, value: Any, *, source: str = "service") -> None: ...


def _coerce_data_delivery(value: Any) -> str | None:
    v = str(value or "").strip().lower()
    if v in ("buffered", "callback"):
        return v
    return None


class PyEngineServiceNode(ServiceNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        state = {"dataDelivery": "buffered"}
        state.update(dict(initial_state or {}))
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=[],
            state_fields=[s.name for s in list(node.stateFields or [])],
        )
        self._initial_state = state

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        mode = _coerce_data_delivery(self._initial_state.get("dataDelivery"))
        if mode is None:
            return
        try:
            cast(_DataDeliveryBus, bus).set_data_delivery(mode, source="initial_state")
        except Exception:
            logger.exception("set_data_delivery failed during attach node=%s mode=%s", self.node_id, mode)

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name != "dataDelivery":
            return value
        mode = _coerce_data_delivery(value)
        if mode is None:
            raise ValueError("invalid dataDelivery (expected callback or buffered)")
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
            logger.exception("set_data_delivery failed node=%s mode=%s", self.node_id, mode)
            return
