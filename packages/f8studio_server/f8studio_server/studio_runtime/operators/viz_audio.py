from __future__ import annotations

from ._viz_base import StudioVizRuntimeNodeBase

from ...presentation_models import AudioConfig

import msgspec

import asyncio
import logging
import time
from typing import Any, cast

from f8pysdk.capabilities import RungraphHookBus
from f8pysdk.f8_naming import ensure_token
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    audio_chunk_port,
    boolean_schema,
    integer_schema,
)

from ..identifiers import SERVICE_CLASS
from ..presentation import PresentationOutlet
from .categories import PALETTE_CATEGORY_VIZ


OPERATOR_CLASS = "f8.viz.audio"
RENDERER_CLASS = "viz_audio"


logger = logging.getLogger(__name__)


class VizAudioRuntimeNode(StudioVizRuntimeNodeBase):
    """
    Studio visualization node for latest-audio waveforms.

    This runtime node sends stream configuration to the injected presentation outlet.
    """

    SPEC = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=SERVICE_CLASS,
        paletteCategory=PALETTE_CATEGORY_VIZ,
        operatorClass=OPERATOR_CLASS,
        version="0.0.1",
        label="Audio Viz",
        description="Display waveform from Zenoh latest-audio streams.",
        tags=["ui", "zenoh", "audio", "viewer", "waveform"],
        dataInPorts=[
            audio_chunk_port(
                name="audio",
                description="Input audio chunk stream.",
                definition_protected=True,
            ),
        ],
        dataOutPorts=[],
        rendererClass=RENDERER_CLASS,
        stateFields=[
            F8StateSpec(
                name="uiUpdate",
                label="UI Update",
                description="Pause/resume embedded viewer updates in the editor.",
                valueSchema=boolean_schema(default=True),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="throttleMs",
                label="Refresh (ms)",
                description="UI refresh interval in milliseconds (0 = as fast as possible).",
                valueSchema=integer_schema(default=20, minimum=0, maximum=60000),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="historyMs",
                label="History (ms)",
                description="Waveform window length in milliseconds.",
                valueSchema=integer_schema(default=250, minimum=20, maximum=680),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="channel",
                label="Channel",
                description="Channel to display (0..N-1).",
                valueSchema=integer_schema(default=0, minimum=0, maximum=16),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
        ],
    )

    def __init__(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        initial_state: dict[str, Any] | None = None,
        presentation: PresentationOutlet,
    ) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=["audio"],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
            initial_state=initial_state,
            presentation=presentation,
        )
        self._config_loaded = False
        self._throttle_ms = 20
        self._history_ms = 250
        self._channel = 0
        self._pending_task: asyncio.Task[object] | None = None
        self._init_task: asyncio.Task[None] | None = None
        self._rungraph_bus: RungraphHookBus | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        rungraph_bus = cast(RungraphHookBus, bus)
        rungraph_bus.register_rungraph_hook(self)
        self._rungraph_bus = rungraph_bus
        try:
            loop = asyncio.get_running_loop()
            self._init_task = loop.create_task(self._ensure_config_loaded(), name=f"pystudio:audio:init:{self.node_id}")
        except RuntimeError:
            logger.debug("viz audio initialization deferred; no event loop node_id=%s", self.node_id, exc_info=True)

    async def close(self) -> None:
        rungraph_bus = self._rungraph_bus
        self._rungraph_bus = None
        if rungraph_bus is not None:
            rungraph_bus.unregister_rungraph_hook(self)
        tasks = [task for task in (self._init_task, self._pending_task) if task is not None]
        self._init_task = None
        self._pending_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error("viz audio background task failed node_id=%s", self.node_id, exc_info=result)
        self.presentation.emit(self.node_id, "viz.audio.detach", {}, ts_ms=int(time.time() * 1000))

    async def validate_rungraph(self, graph: F8RuntimeGraph) -> None:
        del graph

    async def on_rungraph(self, graph: F8RuntimeGraph) -> None:
        del graph
        await self._ensure_config_loaded()
        await self._push_config(now_ms=int(time.time() * 1000))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del value
        f = str(field or "").strip()
        if f not in ("throttleMs", "historyMs", "channel"):
            return
        await self._ensure_config_loaded()
        if f == "throttleMs":
            self._throttle_ms = await self._get_int_state(
                "throttleMs", default=self._throttle_ms, minimum=0, maximum=60000
            )
        elif f == "historyMs":
            self._history_ms = await self._get_int_state("historyMs", default=self._history_ms, minimum=20, maximum=680)
        elif f == "channel":
            self._channel = await self._get_int_state("channel", default=self._channel, minimum=0, maximum=16)
        await self._push_config(now_ms=int(ts_ms) if ts_ms is not None else int(time.time() * 1000))

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._throttle_ms = await self._get_int_state("throttleMs", default=20, minimum=0, maximum=60000)
        self._history_ms = await self._get_int_state("historyMs", default=250, minimum=20, maximum=680)
        self._channel = await self._get_int_state("channel", default=0, minimum=0, maximum=16)
        self._config_loaded = True
        await self._push_config(now_ms=int(time.time() * 1000))

    async def _push_config(self, *, now_ms: int) -> None:
        if self._pending_task is not None and not self._pending_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending_task = loop.create_task(
            self._push_config_async(now_ms), name=f"pystudio:audio:cfg:{self.node_id}"
        )

    async def _push_config_async(self, now_ms: int) -> None:
        payload: dict[str, object] = msgspec.to_builtins(
            AudioConfig(
                audioStreamKey=str(self.input_zenoh_key("audio") or "").strip(),
                throttleMs=int(self._throttle_ms),
                historyMs=int(self._history_ms),
                channel=int(self._channel),
            )
        )
        self.presentation.emit(
            self.node_id,
            "viz.audio.set",
            payload,
            ts_ms=int(now_ms),
        )


def register_spec(registry: Registry) -> Registry:
    registry.register_operator_spec(VizAudioRuntimeNode.SPEC, overwrite=True)
    return registry
