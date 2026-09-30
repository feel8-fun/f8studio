from __future__ import annotations

import logging
import math
from typing import Any

from f8pysdk.codec import coerce_bool, parse_int, parse_float

from f8pysdk.specs import F8StateAccess, F8StateSpec, integer_schema, string_schema
from f8pysdk.nodes import OperatorNode

from ..presentation import PresentationOutlet


logger = logging.getLogger(__name__)

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

    def __init__(
        self,
        *,
        node_id: str,
        data_in_ports: list[str],
        data_out_ports: list[str],
        state_fields: list[str],
        initial_state: dict[str, Any] | None,
        presentation: PresentationOutlet,
    ) -> None:
        super().__init__(
            node_id=node_id,
            data_in_ports=data_in_ports,
            data_out_ports=data_out_ports,
            state_fields=state_fields,
        )
        self._initial_state = dict(initial_state or {})
        self._presentation = presentation
        self._state_errors: dict[str, tuple[type[BaseException], str]] = {}

    @property
    def presentation(self) -> PresentationOutlet:
        return self._presentation

    async def _config_state_value(self, name: str, *, default: Any = None) -> Any:
        try:
            value = await self.get_state_value(name)
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            key = (type(exc), str(exc))
            if self._state_errors.get(name) != key:
                logger.exception(
                    "visualization state read failed node_id=%s field=%s; using initial value", self.node_id, name
                )
            self._state_errors[name] = key
            return self._initial_state.get(name, default)
        self._state_errors.pop(name, None)
        return value if value is not None else self._initial_state.get(name, default)

    async def _get_int_state(self, name: str, *, default: int, minimum: int, maximum: int) -> int:
        raw = await self._config_state_value(name)
        value = None if isinstance(raw, float) and not math.isfinite(raw) else parse_int(raw)
        return max(minimum, min(maximum, default if value is None else value))

    async def _get_float_state_optional(self, name: str) -> float | None:
        value = parse_float(await self._config_state_value(name))
        return value if value is not None and math.isfinite(value) else None

    async def _get_float_state(self, name: str, *, default: float, minimum: float, maximum: float) -> float:
        value = await self._get_float_state_optional(name)
        return max(minimum, min(maximum, default if value is None else value))

    async def _get_bool_state(self, name: str, *, default: bool) -> bool:
        return coerce_bool(await self._config_state_value(name), default=default)

    async def _get_str_state(self, name: str, *, default: str) -> str:
        value = await self._config_state_value(name)
        return str(value) if value is not None else default

    async def get_upstream_sampling_mode(self) -> str:
        mode = (await self._get_str_state("upstreamSamplingMode", default=UPSTREAM_SAMPLING_MODE_AUTO)).strip().lower()
        return mode if mode in UPSTREAM_SAMPLING_MODE_VALUES else UPSTREAM_SAMPLING_MODE_AUTO

    async def get_upstream_sample_interval_ms(self) -> int:
        return await self._get_int_state(
            "upstreamSampleIntervalMs",
            default=UPSTREAM_SAMPLE_INTERVAL_MS_DEFAULT,
            minimum=UPSTREAM_SAMPLE_INTERVAL_MS_MIN,
            maximum=UPSTREAM_SAMPLE_INTERVAL_MS_MAX,
        )
