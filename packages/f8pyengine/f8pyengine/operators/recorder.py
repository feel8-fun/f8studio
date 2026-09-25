from __future__ import annotations

from f8pysdk.specs import exec_port_specs

from f8pysdk.codec import coerce_flag
from pathlib import Path
import logging
import time
from typing import Any, Final

from f8pysdk.specs import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    editable_collection_edit_policy,
    integer_schema,
    string_schema,
)
from f8pysdk._specs.builtin_fields import OPERATOR_ID_FIELD_NAME, SVC_ID_FIELD_NAME
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ..recording import FORMAT_VERSION, RecordingHeader, RecordingReader, RecordingWriter
from ._runtime_errors import OPERATOR_STATE_PUBLISH_ERRORS

OPERATOR_CLASS: Final[str] = "f8.recorder"

logger = logging.getLogger(__name__)
_RECORDER_WRITE_ERRORS = (Exception,)

_CONTROL_STATE_NAMES = {
    "path",
    "enabled",
    "append",
    "recording",
    "sessionStartTsMs",
    "sampleCount",
    "stateEventCount",
    SVC_ID_FIELD_NAME,
    OPERATOR_ID_FIELD_NAME,
}


class RecorderRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[str(p.name) for p in (node.dataInPorts or [])],
            data_out_ports=[str(p.name) for p in (node.dataOutPorts or [])],
            state_fields=[str(s.name) for s in (node.stateFields or [])],
            exec_in_ports=[str(p) for p in (node.execInPorts or [])],
            exec_out_ports=[str(p) for p in (node.execOutPorts or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._path = str(self._initial_state.get("path") or "").strip()
        self._enabled = coerce_flag(self._initial_state.get("enabled"), default=True)
        self._append = coerce_flag(self._initial_state.get("append"), default=True)
        self._writer: RecordingWriter | None = None
        self._writer_path = ""
        self._session_start_ts_ms: int | None = None
        self._sample_count = 0
        self._state_event_count = 0
        self._published_state_cache: dict[str, Any] = {}
        self._user_state_names = tuple(
            name for name in self.state_fields if str(name).strip() and str(name) not in _CONTROL_STATE_NAMES
        )

    async def close(self) -> None:
        self._close_writer()

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        event_ts_ms = now_ms()
        await self._record_tick(exec_id=exec_id, event_ts_ms=event_ts_ms)
        return []

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "").strip()
        if name == "path":
            self._path = str(value or "").strip()
            self._close_writer()
            await self._safe_set_state_if_changed("recording", False)
            return
        if name == "enabled":
            self._enabled = coerce_flag(value, default=self._enabled)
            if not self._enabled:
                self._close_writer()
                await self._safe_set_state_if_changed("recording", False)
            return
        if name == "append":
            self._append = coerce_flag(value, default=self._append)
            self._close_writer()
            await self._safe_set_state_if_changed("recording", False)
            return
        if name not in self._user_state_names:
            return
        if not self._enabled:
            return
        event_ts_ms = now_ms()
        try:
            writer = self._ensure_writer(start_ts_ms=event_ts_ms)
            relative_offset_ms = max(0, int(event_ts_ms - self._session_start_ts_ms_or_now(event_ts_ms)))
            writer.write_state_change(
                state_ts_ms=event_ts_ms,
                relative_offset_ms=relative_offset_ms,
                field=name,
                value=value,
            )
            self._state_event_count += 1
            await self._publish_recording_state()
        except _RECORDER_WRITE_ERRORS as exc:
            logger.exception("[%s:recorder] failed to record state change: %s", self.node_id, name)
            await self._set_last_error(str(exc))

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name in ("enabled", "append"):
            return coerce_flag(value, default=(name == "enabled"))
        if name == "path":
            return str(value or "").strip()
        return value

    def _header(self, *, start_ts_ms: int) -> RecordingHeader:
        data_ports = tuple(str(name) for name in self.data_in_ports if str(name).strip())
        state_fields = tuple(str(name) for name in self._user_state_names if str(name).strip())
        return RecordingHeader(
            format_version=FORMAT_VERSION,
            created_ts_ms=int(start_ts_ms),
            data_ports=data_ports,
            state_fields=state_fields,
        )

    def _ensure_writer(self, *, start_ts_ms: int) -> RecordingWriter:
        path = str(self._path or "").strip()
        if not path:
            raise ValueError("path is required")
        if self._writer is not None and self._writer_path == path:
            return self._writer

        self._close_writer()
        header = self._header(start_ts_ms=start_ts_ms)
        path_obj = Path(path)
        if self._append and path_obj.exists() and path_obj.stat().st_size > 0:
            existing_header = RecordingReader(path).read_header()
            if (
                int(existing_header.format_version) != int(header.format_version)
                or tuple(existing_header.data_ports) != tuple(header.data_ports)
                or tuple(existing_header.state_fields) != tuple(header.state_fields)
            ):
                raise ValueError("append recording header mismatch")
            self._session_start_ts_ms = int(existing_header.created_ts_ms)
        else:
            self._session_start_ts_ms = int(start_ts_ms)

        writer = RecordingWriter(path, header=header, append=self._append)
        writer.open()
        self._writer = writer
        self._writer_path = path
        return writer

    async def _record_tick(self, *, exec_id: str | int, event_ts_ms: int) -> None:
        if not self._enabled:
            return
        try:
            writer = self._ensure_writer(start_ts_ms=event_ts_ms)
            data: dict[str, Any] = {}
            for port in self.data_in_ports:
                data[str(port)] = await self.pull(str(port), ctx_id=exec_id)
            relative_offset_ms = max(
                0,
                int(event_ts_ms - self._session_start_ts_ms_or_now(event_ts_ms)),
            )
            writer.write_data_sample(
                tick_ts_ms=event_ts_ms,
                relative_offset_ms=relative_offset_ms,
                data=data,
            )
            self._sample_count += 1
            await self._publish_recording_state()
        except _RECORDER_WRITE_ERRORS as exc:
            logger.exception("[%s:recorder] failed to record sample", self.node_id)
            await self._set_last_error(str(exc))

    async def _publish_recording_state(self) -> None:
        session_start_ts_ms = self._session_start_ts_ms
        if session_start_ts_ms is None:
            session_start_ts_ms = now_ms()
        await self._safe_set_state_if_changed("recording", self._writer is not None)
        await self._safe_set_state_if_changed("sessionStartTsMs", int(session_start_ts_ms))
        await self.clear_error()

    async def _set_last_error(self, message: str) -> None:
        self._close_writer()
        await self._safe_set_state_if_changed("recording", False)
        await self.report_error(
            "RECORDER_ERROR",
            str(message),
            severity="error",
            fingerprint=f"recorder:{message}",
        )

    async def _safe_set_state_if_changed(self, field: str, value: Any) -> None:
        prev = self._published_state_cache.get(field)
        if prev == value:
            return
        self._published_state_cache[field] = value
        try:
            await self.set_state(field, value)
        except OPERATOR_STATE_PUBLISH_ERRORS:
            self._published_state_cache.pop(field, None)
            logger.exception("[%s:recorder] failed to publish state: %s", self.node_id, field)

    def _session_start_ts_ms_or_now(self, fallback: int) -> int:
        if self._session_start_ts_ms is not None:
            return int(self._session_start_ts_ms)
        return int(fallback)

    def _close_writer(self) -> None:
        writer = self._writer
        self._writer = None
        self._writer_path = ""
        if writer is not None:
            writer.close()


def now_ms() -> int:
    return int(time.time() * 1000.0)

RecorderRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.debug",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Recorder",
    description="Tick-driven debug recorder that captures data samples and sparse state changes.",
    tags=["record", "replay", "debug", "capture"],
    execInPorts=exec_port_specs(["record"]),
    execOutPorts=exec_port_specs([]),
    dataInPorts=[],
    dataOutPorts=[],
    editPolicy=F8SpecEditPolicy(
        stateFields=editable_collection_edit_policy(),
        dataInPorts=editable_collection_edit_policy(),
    ),
    stateFields=[
        F8StateSpec(
            name="path",
            label="Path",
            description="Recording output path.",
            valueSchema=string_schema(),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="enabled",
            label="Enabled",
            description="When enabled, incoming exec ticks are recorded.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="append",
            label="Append",
            description="Append to an existing compatible recording file.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="recording",
            label="Recording",
            description="Readonly flag indicating whether the file is open and writable.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="sessionStartTsMs",
            label="Session Start",
            description="Readonly session start timestamp in milliseconds.",
            valueSchema=integer_schema(),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)

def register_operator(registry: Registry) -> Registry:
    registry.register_operator(RecorderRuntimeNode.SPEC, RecorderRuntimeNode, overwrite=True)
    return registry
