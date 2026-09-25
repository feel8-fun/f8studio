from __future__ import annotations

from f8pysdk.specs import exec_port_specs

from typing import Any

from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    complex_object_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.time_utils import now_ms

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.stream_watchdog"


class StreamWatchdogRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[port.name for port in (node.dataInPorts or [])],
            data_out_ports=[port.name for port in (node.dataOutPorts or [])],
            state_fields=[state.name for state in (node.stateFields or [])],
            exec_in_ports=[str(port) for port in (node.execInPorts or [])],
            exec_out_ports=[str(port) for port in (node.execOutPorts or [])],
        )
        state = dict(initial_state or {})
        self._timeout_ms = _integer_or_default(state.get("timeoutMs"), 250)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if str(field or "").strip() == "timeoutMs":
            self._timeout_ms = _integer_or_default(value, self._timeout_ms)

    async def validate_state(
        self,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        del ts_ms, meta
        if str(field or "").strip() != "timeoutMs":
            return value
        timeout_ms = _integer_or_default(value, -1)
        if timeout_ms < 10 or timeout_ms > 60_000:
            raise ValueError("timeoutMs must be in range 10..60000")
        return timeout_ms

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        value = await self.pull("value", ctx_id=exec_id)
        valid, _age_ms, _reason = self._freshness(value)
        return ["valid"] if valid else []

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_name = str(port or "").strip()
        if port_name not in {"value", "valid", "ageMs", "status"}:
            return None
        value = await self.pull("value", ctx_id=ctx_id)
        valid, age_ms, reason = self._freshness(value)
        if port_name == "value":
            return value if valid else None
        if port_name == "valid":
            return valid
        if port_name == "ageMs":
            return age_ms
        return {"valid": valid, "ageMs": age_ms, "timeoutMs": self._timeout_ms, "reason": reason}

    def _freshness(self, value: Any) -> tuple[bool, float | None, str]:
        timestamps = _receive_timestamps(value)
        if not timestamps:
            return False, None, "missing_receive_timestamp"
        oldest_timestamp = min(timestamps)
        age_ms = max(0.0, float(now_ms()) - oldest_timestamp)
        if age_ms > self._timeout_ms:
            return False, age_ms, "stale"
        return True, age_ms, "ok"


def _receive_timestamps(value: Any) -> list[float]:
    if isinstance(value, list):
        timestamps: list[float] = []
        for item in value:
            timestamps.extend(_receive_timestamps(item))
        return timestamps
    if not isinstance(value, dict):
        return []
    raw_timestamp = value.get("receivedAtMs")
    if raw_timestamp is None:
        raw_timestamp = value.get("timestampMs")
    timestamp = _finite_number(raw_timestamp)
    return [] if timestamp is None else [timestamp]


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return None
    return numeric


def _integer_or_default(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


StreamWatchdogRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.1.0",
    label="Stream Watchdog",
    description="Invalidate stale timestamped data and gate exec flow when a stream stops.",
    tags=["stream", "watchdog", "safety", "timeout", "gate"],
    execInPorts=exec_port_specs(["check"]),
    execOutPorts=exec_port_specs(["valid"]),
    dataInPorts=[F8DataPortSpec(name="value", description="Timestamped stream value.", valueSchema=any_schema())],
    dataOutPorts=[
        F8DataPortSpec(name="value", description="Input while fresh, otherwise None.", valueSchema=any_schema()),
        F8DataPortSpec(name="valid", description="Whether the input is fresh.", valueSchema=boolean_schema(default=False)),
        F8DataPortSpec(name="ageMs", description="Age of the oldest input sample.", valueSchema=number_schema(minimum=0.0)),
        F8DataPortSpec(
            name="status",
            description="Per-check freshness status.",
            valueSchema=complex_object_schema(
                properties={
                    "valid": boolean_schema(),
                    "ageMs": number_schema(minimum=0.0),
                    "timeoutMs": integer_schema(minimum=10),
                    "reason": string_schema(),
                }
            ),
        ),
    ],
    stateFields=[
        F8StateSpec(
            name="timeoutMs",
            label="Timeout (ms)",
            description="Maximum input age before output and exec flow are blocked.",
            valueSchema=integer_schema(default=250, minimum=10, maximum=60_000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        )
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(StreamWatchdogRuntimeNode.SPEC, StreamWatchdogRuntimeNode, overwrite=True)
    return registry


__all__ = ["StreamWatchdogRuntimeNode", "register_operator"]
