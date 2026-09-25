from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import math
import time
from dataclasses import dataclass
from typing import Any, Final

from f8pysdk.codec import parse_number, unwrap_json_value
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    editable_collection_edit_policy,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

OPERATOR_CLASS: Final[str] = "f8.switch_mixer"


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass
class _Slew:
    value: float
    target: float

    def step(self, *, dt_s: float, tau_s: float) -> float:
        dt = max(0.0, float(dt_s))
        tau = max(0.0, float(tau_s))
        if tau <= 0.0:
            self.value = float(self.target)
            return float(self.value)
        alpha = 1.0 - math.exp(-dt / tau)
        self.value = float(self.value + alpha * (self.target - self.value))
        return float(self.value)


class SwitchMixerRuntimeNode(OperatorNode):
    """
    Multi-channel switch mixer.

    - Users can define any number of numeric data input ports.
    - `currentChannel` selects which input channel should be heard now.
    - When the channel changes, the node crossfades from the previous mixed output
      to the newly selected channel.
    - If a channel temporarily has no valid fresh input, the last valid sample for
      that channel is held and repeated.
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
        self._last_time_s: float | None = None
        self._last_ctx_id: str | int | None = None
        self._cache: dict[str, Any] = {}
        self._fade_tau_s = 0.0
        self._requested_channel = ""
        self._resolved_channel = ""
        self._last_values: dict[str, float] = {}
        self._transition_from_value = 0.0
        self._alpha = _Slew(value=1.0, target=1.0)
        self._last_resolved_published: str | None = None
        self._refresh_runtime_params(self._initial_state)

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del active, meta
        self._last_time_s = None

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name not in ("fadeMs", "currentChannel"):
            return
        self._refresh_runtime_params({name: value})

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "fadeMs":
            fade_ms = unwrap_json_value(value)
            if fade_ms is None:
                return 0
            return int(max(0, int(fade_ms)))
        if name == "currentChannel":
            return str(unwrap_json_value(value) or "").strip()
        return value

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        name = str(port or "")
        if name not in ("out", "alpha"):
            return None
        if ctx_id is not None and ctx_id == self._last_ctx_id and name in self._cache:
            return self._cache.get(name)

        out = await self._step(ctx_id=ctx_id)
        self._last_ctx_id = ctx_id
        self._cache = dict(out)
        return self._cache.get(name)

    async def _publish_resolved_channel_if_needed(self) -> None:
        if self._last_resolved_published == self._resolved_channel:
            return
        self._last_resolved_published = str(self._resolved_channel)
        await self.set_state("resolvedChannel", str(self._resolved_channel))

    def _resolve_desired_channel(self) -> str:
        requested = str(self._requested_channel or "").strip()
        if requested and requested in self.data_in_ports:
            return requested
        if self._resolved_channel and self._resolved_channel in self.data_in_ports:
            return str(self._resolved_channel)
        if self.data_in_ports:
            return str(self.data_in_ports[0])
        return ""

    def _current_channel_value(self, channel: str) -> float | None:
        name = str(channel or "").strip()
        if not name:
            return None
        return self._last_values.get(name)

    @staticmethod
    def _coerce_float_or_none(value: Any) -> float | None:
        raw = unwrap_json_value(value)
        if raw is None:
            return None
        number = parse_number(raw)
        if number is None:
            return None
        return float(number)

    async def _step(self, *, ctx_id: str | int | None) -> dict[str, Any]:
        now_s = time.monotonic()
        if self._last_time_s is None:
            dt_s = 0.0
        else:
            dt_s = max(0.0, now_s - float(self._last_time_s))
        self._last_time_s = now_s

        for port_name in list(self.data_in_ports):
            raw = await super().pull(port_name, ctx_id=ctx_id)
            value = self._coerce_float_or_none(raw)
            if value is not None:
                self._last_values[str(port_name)] = float(value)

        desired_channel = self._resolve_desired_channel()
        if desired_channel != self._resolved_channel:
            first_resolution = (not self._resolved_channel) and ("out" not in self._cache)
            current_out = float(self._cache.get("out", self._current_channel_value(self._resolved_channel) or 0.0))
            next_target = self._current_channel_value(desired_channel)
            if next_target is None:
                next_target = current_out
            self._transition_from_value = float(current_out)
            self._resolved_channel = str(desired_channel)
            self._alpha.value = 1.0 if first_resolution else 0.0
            self._alpha.target = 1.0
            if first_resolution:
                self._transition_from_value = float(next_target)
        await self._publish_resolved_channel_if_needed()

        target_value = self._current_channel_value(self._resolved_channel)
        if target_value is None:
            target_value = float(self._cache.get("out", self._transition_from_value))

        alpha = _clamp01(self._alpha.step(dt_s=dt_s, tau_s=self._fade_tau_s))
        out_value = (1.0 - alpha) * float(self._transition_from_value) + alpha * float(target_value)

        if alpha >= 0.999999:
            self._transition_from_value = float(target_value)

        return {"out": float(out_value), "alpha": float(alpha)}

    def _refresh_runtime_params(self, values: dict[str, Any]) -> None:
        if "fadeMs" in values:
            fade_raw = unwrap_json_value(values.get("fadeMs"))
            if fade_raw is not None:
                fade_ms = max(0.0, float(fade_raw))
                fade_s = float(fade_ms / 1000.0)
                self._fade_tau_s = float(fade_s / 3.0) if fade_s > 0.0 else 0.0
        if "currentChannel" in values:
            self._requested_channel = str(unwrap_json_value(values.get("currentChannel")) or "").strip()


SwitchMixerRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.0.3",
    label="Switch Mixer",
    description="Switch between any number of user-defined input channels with an optional smooth crossfade.",
    tags=["mix", "switch", "channel", "track", "crossfade"],
    execInPorts=exec_port_specs(["exec"]),
    execOutPorts=exec_port_specs(["exec"]),
    dataInPorts=[
        F8DataPortSpec(name="ch1", description="Input channel 1", valueSchema=number_schema(), definitionProtected=False),
        F8DataPortSpec(name="ch2", description="Input channel 2", valueSchema=number_schema(), definitionProtected=False),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="out", description="Mixed output", valueSchema=number_schema()),
        F8DataPortSpec(name="alpha", description="Transition progress (0..1)", valueSchema=number_schema()),
    ],
    editPolicy=F8SpecEditPolicy(dataInPorts=editable_collection_edit_policy()),
    stateFields=[
        F8StateSpec(
            name="currentChannel",
            label="Current Channel",
            description="Name of the selected input channel/track to play.",
            valueSchema=string_schema(default="ch1"),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="resolvedChannel",
            label="Resolved Channel",
            description="Readonly currently resolved input channel after validation/fallback.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="fadeMs",
            label="Fade (ms)",
            description="Transition duration in milliseconds. Set to 0 for an instant switch.",
            valueSchema=integer_schema(default=200, minimum=0, maximum=60_000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(SwitchMixerRuntimeNode.SPEC, SwitchMixerRuntimeNode, overwrite=True)
    return registry
