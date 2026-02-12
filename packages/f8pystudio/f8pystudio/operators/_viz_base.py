from __future__ import annotations

from typing import Any

from f8pysdk import F8StateAccess, F8StateSpec, integer_schema, string_schema
from f8pysdk.runtime_node import OperatorNode


UPSTREAM_SAMPLING_MODE_PASSIVE = "passive"
UPSTREAM_SAMPLING_MODE_AUTO = "auto"
UPSTREAM_SAMPLING_MODE_VALUES = (UPSTREAM_SAMPLING_MODE_PASSIVE, UPSTREAM_SAMPLING_MODE_AUTO)
UPSTREAM_SAMPLE_HZ_DEFAULT = 10
UPSTREAM_SAMPLE_HZ_MIN = 1
UPSTREAM_SAMPLE_HZ_MAX = 120


def viz_sampling_state_fields(*, show_on_node: bool = False) -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="upstreamSamplingMode",
            label="Upstream Sampling",
            description="passive: no auto sampling injection; auto: request upstream auto sampler injection.",
            valueSchema=string_schema(default=UPSTREAM_SAMPLING_MODE_PASSIVE, enum=list(UPSTREAM_SAMPLING_MODE_VALUES)),
            access=F8StateAccess.rw,
            showOnNode=show_on_node,
        ),
        F8StateSpec(
            name="upstreamSampleHz",
            label="Upstream Sample Hz",
            description="Requested upstream auto sampling frequency in Hz.",
            valueSchema=integer_schema(
                default=UPSTREAM_SAMPLE_HZ_DEFAULT,
                minimum=UPSTREAM_SAMPLE_HZ_MIN,
                maximum=UPSTREAM_SAMPLE_HZ_MAX,
            ),
            access=F8StateAccess.rw,
            showOnNode=show_on_node,
        ),
    ]


class StudioVizRuntimeNodeBase(OperatorNode):
    """
    Shared helpers for Studio visualization runtime nodes.
    """

    def __init__(
        self,
        *,
        node_id: str,
        data_in_ports: list[str],
        data_out_ports: list[str],
        state_fields: list[str],
        initial_state: dict[str, Any] | None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            data_in_ports=data_in_ports,
            data_out_ports=data_out_ports,
            state_fields=state_fields,
        )
        self._initial_state = dict(initial_state or {})

    async def get_upstream_sampling_mode(self) -> str:
        mode_any: Any = None
        try:
            mode_any = await self.get_state_value("upstreamSamplingMode")
        except Exception:
            mode_any = None
        if mode_any is None:
            mode_any = self._initial_state.get("upstreamSamplingMode", UPSTREAM_SAMPLING_MODE_PASSIVE)
        mode = str(mode_any or "").strip().lower()
        if mode not in UPSTREAM_SAMPLING_MODE_VALUES:
            return UPSTREAM_SAMPLING_MODE_PASSIVE
        return mode

    async def get_upstream_sample_hz(self) -> int:
        hz_any: Any = None
        try:
            hz_any = await self.get_state_value("upstreamSampleHz")
        except Exception:
            hz_any = None
        if hz_any is None:
            hz_any = self._initial_state.get("upstreamSampleHz", UPSTREAM_SAMPLE_HZ_DEFAULT)
        try:
            hz = int(hz_any) if hz_any is not None else UPSTREAM_SAMPLE_HZ_DEFAULT
        except (TypeError, ValueError):
            hz = UPSTREAM_SAMPLE_HZ_DEFAULT
        if hz < UPSTREAM_SAMPLE_HZ_MIN:
            hz = UPSTREAM_SAMPLE_HZ_MIN
        if hz > UPSTREAM_SAMPLE_HZ_MAX:
            hz = UPSTREAM_SAMPLE_HZ_MAX
        return hz
