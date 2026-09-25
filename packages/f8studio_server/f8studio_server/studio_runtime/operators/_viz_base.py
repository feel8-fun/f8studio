from __future__ import annotations

from typing import Any

from f8pysdk.specs import F8StateAccess, F8StateSpec, integer_schema, string_schema
from f8pysdk.nodes import OperatorNode

from ..presentation import PresentationOutlet


UPSTREAM_SAMPLING_MODE_PASSIVE = "passive"
UPSTREAM_SAMPLING_MODE_AUTO = "auto"
UPSTREAM_SAMPLING_MODE_VALUES = (UPSTREAM_SAMPLING_MODE_PASSIVE, UPSTREAM_SAMPLING_MODE_AUTO)
UPSTREAM_SAMPLE_INTERVAL_MS_DEFAULT = 100
UPSTREAM_SAMPLE_INTERVAL_MS_MIN = 8
UPSTREAM_SAMPLE_INTERVAL_MS_MAX = 5000


def viz_sampling_state_fields(*, show_on_node: bool = False) -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="upstreamSamplingMode",
            label="Upstream Sampling",
            description=(
                "passive: no upstream auto-sampler request; auto: request upstream periodic sampling when the "
                "source runtime supports it."
            ),
            valueSchema=string_schema(default=UPSTREAM_SAMPLING_MODE_AUTO, enum=list(UPSTREAM_SAMPLING_MODE_VALUES)),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=show_on_node,
        ),
        F8StateSpec(
            name="upstreamSampleIntervalMs",
            label="Upstream Sample Interval (ms)",
            description="Requested upstream auto sampling interval in milliseconds.",
            valueSchema=integer_schema(
                default=UPSTREAM_SAMPLE_INTERVAL_MS_DEFAULT,
                minimum=UPSTREAM_SAMPLE_INTERVAL_MS_MIN,
                maximum=UPSTREAM_SAMPLE_INTERVAL_MS_MAX,
            ),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=show_on_node,
        ),
    ]


class StudioVizRuntimeNodeBase(OperatorNode):
    """
    Shared helpers for Studio visualization runtime nodes.
    """

    presentation: PresentationOutlet

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
        except (RuntimeError, TypeError, ValueError):
            mode_any = None
        if mode_any is None:
            mode_any = self._initial_state.get("upstreamSamplingMode", UPSTREAM_SAMPLING_MODE_AUTO)
        mode = str(mode_any or "").strip().lower()
        if mode not in UPSTREAM_SAMPLING_MODE_VALUES:
            return UPSTREAM_SAMPLING_MODE_AUTO
        return mode

    async def get_upstream_sample_interval_ms(self) -> int:
        interval_any: Any = None
        try:
            interval_any = await self.get_state_value("upstreamSampleIntervalMs")
        except (RuntimeError, TypeError, ValueError):
            interval_any = None
        if interval_any is None:
            interval_any = self._initial_state.get(
                "upstreamSampleIntervalMs",
                UPSTREAM_SAMPLE_INTERVAL_MS_DEFAULT,
            )
        try:
            interval_ms = int(interval_any) if interval_any is not None else UPSTREAM_SAMPLE_INTERVAL_MS_DEFAULT
        except (TypeError, ValueError):
            interval_ms = UPSTREAM_SAMPLE_INTERVAL_MS_DEFAULT
        if interval_ms < UPSTREAM_SAMPLE_INTERVAL_MS_MIN:
            interval_ms = UPSTREAM_SAMPLE_INTERVAL_MS_MIN
        if interval_ms > UPSTREAM_SAMPLE_INTERVAL_MS_MAX:
            interval_ms = UPSTREAM_SAMPLE_INTERVAL_MS_MAX
        return interval_ms
