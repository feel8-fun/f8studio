from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    integer_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ..ui_bus import emit_ui_command
from ._viz_base import StudioVizRuntimeNodeBase, viz_sampling_state_fields

logger = logging.getLogger(__name__)

OPERATOR_CLASS = "f8.tcode_viewer"
RENDERER_CLASS = "pystudio_tcode_viewer"
MODEL_VALUES = ("OSR2", "SR6", "SSR1")


class PyStudioTCodeViewerRuntimeNode(StudioVizRuntimeNodeBase):
    """
    Studio-side runtime node that forwards TCode lines to detached viewer UI.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
            initial_state=initial_state,
        )
        self._config_loaded = False
        self._model = "SR6"
        self._throttle_ms = 0
        self._max_line_length = 4096

        self._pending_lines: list[str] = []
        self._last_flush_ms: int | None = None
        self._flush_task: asyncio.Task[object] | None = None
        self._warned_signatures: set[str] = set()

    async def close(self) -> None:
        task = self._flush_task
        self._flush_task = None
        if task is not None:
            try:
                task.cancel()
            except Exception:
                logger.exception("failed to cancel tcode viewer flush task nodeId=%s", self.node_id)
            try:
                await asyncio.gather(task, return_exceptions=True)
            except Exception:
                logger.exception("failed to await tcode viewer flush task nodeId=%s", self.node_id)
        emit_ui_command(self.node_id, "tcode_viewer.detach", {}, ts_ms=int(time.time() * 1000))

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if str(port or "").strip() != "tcode":
            return
        await self._ensure_config_loaded()

        if not isinstance(value, str):
            self._log_bad_input_once(value)
            return

        line = self._normalize_line(value)
        if not line:
            return
        self._pending_lines.append(line)
        now_ms = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        await self._schedule_flush(now_ms=now_ms)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        await self._ensure_config_loaded()
        name = str(field or "").strip()
        if not name:
            return

        if name == "model":
            next_model = self._coerce_model(value, default=self._model)
            if next_model == self._model:
                return
            self._model = next_model
            self._emit_set_model()
            self._emit_reset()
            return

        if name == "throttleMs":
            self._throttle_ms = self._coerce_int(value, default=self._throttle_ms, minimum=0, maximum=60000)
            return

        if name == "maxLineLength":
            self._max_line_length = self._coerce_int(value, default=self._max_line_length, minimum=32, maximum=65536)
            return

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._model = self._coerce_model(await self._get_state_or_initial("model", "SR6"), default="SR6")
        self._throttle_ms = self._coerce_int(
            await self._get_state_or_initial("throttleMs", 0), default=0, minimum=0, maximum=60000
        )
        self._max_line_length = self._coerce_int(
            await self._get_state_or_initial("maxLineLength", 4096), default=4096, minimum=32, maximum=65536
        )
        self._config_loaded = True
        self._emit_set_model()

    async def _get_state_or_initial(self, name: str, default: Any) -> Any:
        value: Any = None
        try:
            value = await self.get_state_value(name)
        except Exception:
            value = None
        if value is not None:
            return value
        return self._initial_state.get(name, default)

    async def _schedule_flush(self, *, now_ms: int) -> None:
        throttle_ms = max(0, int(self._throttle_ms))
        if throttle_ms <= 0:
            self._emit_pending(now_ms=now_ms)
            return

        last_flush_ms = self._last_flush_ms
        if last_flush_ms is None or now_ms >= (last_flush_ms + throttle_ms):
            self._emit_pending(now_ms=now_ms)
            return

        if self._flush_task is not None and not self._flush_task.done():
            return

        delay_ms = max(0, (last_flush_ms + throttle_ms) - now_ms)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._flush_task = loop.create_task(self._flush_after(delay_ms=delay_ms), name=f"pystudio:tcode:flush:{self.node_id}")

    async def _flush_after(self, *, delay_ms: int) -> None:
        try:
            await asyncio.sleep(float(max(0, int(delay_ms))) / 1000.0)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("failed in delayed flush sleep nodeId=%s", self.node_id)
            return
        self._emit_pending(now_ms=int(time.time() * 1000))

    def _emit_set_model(self) -> None:
        emit_ui_command(
            self.node_id,
            "tcode_viewer.set_model",
            {"model": self._model},
            ts_ms=int(time.time() * 1000),
        )

    def _emit_reset(self) -> None:
        emit_ui_command(self.node_id, "tcode_viewer.reset", {}, ts_ms=int(time.time() * 1000))

    def _emit_pending(self, *, now_ms: int) -> None:
        if not self._pending_lines:
            self._last_flush_ms = int(now_ms)
            return
        line = "".join(self._pending_lines)
        self._pending_lines.clear()
        emit_ui_command(self.node_id, "tcode_viewer.write", {"line": line, "tsMs": int(now_ms)}, ts_ms=int(now_ms))
        self._last_flush_ms = int(now_ms)

    def _normalize_line(self, value: str) -> str:
        text = value.replace("\r", "")
        if self._max_line_length > 0 and len(text) > self._max_line_length:
            text = text[: self._max_line_length]
        if not text.endswith("\n"):
            text = text + "\n"
        return text

    @staticmethod
    def _coerce_model(value: Any, *, default: str) -> str:
        text = str(value or "").strip().upper()
        if text in MODEL_VALUES:
            return text
        return default

    @staticmethod
    def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            out = int(value) if value is not None else int(default)
        except (TypeError, ValueError):
            out = int(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

    def _log_bad_input_once(self, value: Any) -> None:
        sig = f"{type(value).__name__}"
        if sig in self._warned_signatures:
            return
        self._warned_signatures.add(sig)
        logger.warning("tcode_viewer ignored invalid input type=%s nodeId=%s", type(value).__name__, self.node_id)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PyStudioTCodeViewerRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=SERVICE_CLASS,
            operatorClass=OPERATOR_CLASS,
            version="0.0.1",
            label="TCode Viewer",
            description="Detached OSR emulator viewer for TCode string streams.",
            tags=["viz", "tcode", "osr", "ui"],
            dataInPorts=[
                F8DataPortSpec(
                    name="tcode",
                    description="TCode command string input.",
                    valueSchema=string_schema(),
                )
            ],
            dataOutPorts=[],
            rendererClass=RENDERER_CLASS,
            stateFields=[
                F8StateSpec(
                    name="model",
                    label="Model",
                    description="OSR model type in viewer.",
                    valueSchema=string_schema(default="SR6", enum=list(MODEL_VALUES)),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="throttleMs",
                    label="Push Throttle (ms)",
                    description="UI command throttling interval in milliseconds (0 = no throttle).",
                    valueSchema=integer_schema(default=0, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="maxLineLength",
                    label="Max Line Length",
                    description="Maximum allowed length of each incoming line.",
                    valueSchema=integer_schema(default=4096, minimum=32, maximum=65536),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                *viz_sampling_state_fields(show_on_node=False),
            ],
        ),
        overwrite=True,
    )
    return reg
