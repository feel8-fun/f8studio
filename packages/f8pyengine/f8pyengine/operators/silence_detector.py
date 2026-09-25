from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import time
from typing import Any, Final

from f8pysdk.codec import parse_number
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    integer_schema,
    number_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

OPERATOR_CLASS: Final[str] = "f8.silence_detector"


class SilenceDetectorRuntimeNode(OperatorNode):
    """
    Detect whether an input signal has stopped changing by more than `deltaThreshold`
    for at least `silenceMs`.

    The result is published through state fields so downstream graph logic can consume
    sparse state changes instead of per-sample data.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._exec_out_ports = exec_out_ports(node, default=["exec"])
        self._silence_s = 0.5
        self._delta_threshold = 0.001
        self._last_value: float | None = None
        self._last_active_s: float | None = None
        self._is_silent = False
        self._refresh_runtime_params(self._initial_state)

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        del exec_id, in_port
        await self._sample_and_publish()
        return list(self._exec_out_ports)

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        if not bool(active):
            self._last_active_s = None

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name not in ("silenceMs", "deltaThreshold"):
            return
        self._refresh_runtime_params({name: value})

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "silenceMs":
            number = parse_number(value)
            if number is None:
                raise ValueError("silenceMs must be numeric")
            return int(max(0, int(number)))
        if name == "deltaThreshold":
            number = parse_number(value)
            if number is None:
                raise ValueError("deltaThreshold must be numeric")
            return float(max(0.0, float(number)))
        return value

    async def _sample_and_publish(self) -> None:
        raw = await self.pull("value", ctx_id=None)
        value = parse_number(raw)
        now_s = time.monotonic()

        if value is not None:
            value_f = float(value)
            if self._last_value is None:
                self._last_active_s = now_s
            elif abs(value_f - float(self._last_value)) > float(self._delta_threshold):
                self._last_active_s = now_s
            self._last_value = value_f

        if self._last_active_s is None:
            self._last_active_s = now_s

        next_is_silent = False
        if float(self._silence_s) > 0.0:
            next_is_silent = (now_s - float(self._last_active_s)) >= float(self._silence_s)

        if next_is_silent != self._is_silent:
            self._is_silent = bool(next_is_silent)
            await self.set_state("isSilent", bool(self._is_silent))

    def _refresh_runtime_params(self, values: dict[str, Any]) -> None:
        if "silenceMs" in values:
            silence_ms = parse_number(values.get("silenceMs"))
            if silence_ms is not None:
                self._silence_s = float(max(0.0, float(silence_ms)) / 1000.0)
        if "deltaThreshold" in values:
            delta = parse_number(values.get("deltaThreshold"))
            if delta is not None:
                self._delta_threshold = float(max(0.0, float(delta)))


SilenceDetectorRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.analysis",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Silence Detector",
    description="Detect whether a signal has stayed nearly unchanged for long enough to be considered silent.",
    tags=["analysis", "silence", "activity", "state", "gate"],
    execInPorts=exec_port_specs(["exec"]),
    execOutPorts=exec_port_specs(["exec"]),
    dataInPorts=[
        F8DataPortSpec(name="value", description="Signal to analyze", valueSchema=number_schema()),
    ],
    dataOutPorts=[],
    stateFields=[
        F8StateSpec(
            name="silenceMs",
            label="Silence (ms)",
            description="If the input changes less than deltaThreshold for this long, mark it silent.",
            valueSchema=integer_schema(default=500, minimum=0, maximum=60_000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="deltaThreshold",
            label="Delta Threshold",
            description="Absolute change threshold to treat the input as active.",
            valueSchema=number_schema(default=0.001, minimum=0.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="isSilent",
            label="Is Silent",
            description="Readonly sparse state output indicating whether the signal is currently silent.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(SilenceDetectorRuntimeNode.SPEC, SilenceDetectorRuntimeNode, overwrite=True)
    return registry
