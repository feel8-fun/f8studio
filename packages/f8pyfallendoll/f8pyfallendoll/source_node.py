from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f8pysdk.capabilities import ClosableNode, NodeBus
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode
from f8pysdk.specs import F8RuntimeNode

from .constants import (
    DEFAULT_FUNCTIONAL_BONES,
    DEFAULT_POLL_INTERVAL_MS,
    DEFAULT_REFERENCE_PARTICIPANTS,
    DEFAULT_STALE_AFTER_MS,
    DEFAULT_TARGET_PARTICIPANTS,
)
from .stream import (
    available_bones,
    available_participants,
    parse_latest_frame,
    read_appended,
    resolve_spool_path,
    select_frame,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceConfig:
    runtime_dir: str
    poll_interval_ms: int
    stale_after_ms: int
    reference_role: str
    target_role: str
    enabled_reference_participants: list[str]
    enabled_target_participants: list[str]
    enabled_reference_bones: list[str]
    enabled_target_bones: list[str]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None or isinstance(value, bool):
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)
    return max(int(minimum), min(int(maximum), parsed))


def _string_list(value: Any, *, default: list[str]) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return list(default)
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


class FallenDollSourceNode(ServiceNode, ClosableNode):
    def __init__(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[port.name for port in (node.dataInPorts or [])],
            data_out_ports=[port.name for port in (node.dataOutPorts or [])],
            state_fields=[field.name for field in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._config = self._config_from_values(self._initial_state)
        self._active = True
        self._task: asyncio.Task[None] | None = None
        self._wakeup = asyncio.Event()
        self._offset = 0
        self._fragment = ""
        self._spool_path: Path | None = None
        self._directory_ready = False
        self._last_valid_ms: int | None = None
        self._stale_emitted = False
        self._published_state: dict[str, Any] = {}
        self._error_signature = ""

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        bus_like = bus if isinstance(bus, NodeBus) else None
        self._active = True if bus_like is None else bool(bus_like.active)
        self._ensure_task()

    async def close(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)
        if self._active:
            self._ensure_task()
            self._wakeup.set()
            return
        await self._emit_inactive_once(reason="inactive")

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "runtimeDir":
            return str(value or "").strip()
        if name == "pollIntervalMs":
            return _coerce_int(value, default=DEFAULT_POLL_INTERVAL_MS, minimum=10, maximum=1000)
        if name == "staleAfterMs":
            return _coerce_int(value, default=DEFAULT_STALE_AFTER_MS, minimum=50, maximum=10000)
        if name in {"referenceRole", "targetRole"}:
            role = str(value or "").strip().lower()
            if not role:
                raise ValueError(f"{name} must not be empty")
            return role
        if name == "enabledReferenceParticipants":
            return _string_list(value, default=DEFAULT_REFERENCE_PARTICIPANTS)
        if name == "enabledTargetParticipants":
            return _string_list(value, default=DEFAULT_TARGET_PARTICIPANTS)
        if name in {"enabledReferenceBones", "enabledTargetBones"}:
            return _string_list(value, default=DEFAULT_FUNCTIONAL_BONES)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        configurable = {
            "runtimeDir",
            "pollIntervalMs",
            "staleAfterMs",
            "referenceRole",
            "targetRole",
            "enabledReferenceParticipants",
            "enabledTargetParticipants",
            "enabledReferenceBones",
            "enabledTargetBones",
        }
        if name not in configurable:
            return
        values = {
            "runtimeDir": self._config.runtime_dir,
            "pollIntervalMs": self._config.poll_interval_ms,
            "staleAfterMs": self._config.stale_after_ms,
            "referenceRole": self._config.reference_role,
            "targetRole": self._config.target_role,
            "enabledReferenceParticipants": self._config.enabled_reference_participants,
            "enabledTargetParticipants": self._config.enabled_target_participants,
            "enabledReferenceBones": self._config.enabled_reference_bones,
            "enabledTargetBones": self._config.enabled_target_bones,
        }
        values[name] = value
        previous_path = self._config.runtime_dir
        self._config = self._config_from_values(values)
        if previous_path != self._config.runtime_dir:
            self._reset_tail()
        self._wakeup.set()

    def _ensure_task(self) -> None:
        if self._task is not None and not self._task.done():
            return
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._run_loop(), name=f"fallendoll-source:{self.node_id}")

    async def _run_loop(self) -> None:
        while True:
            try:
                if self._active:
                    await self._poll_once(_now_ms())
                timeout = max(0.01, self._config.poll_interval_ms / 1000.0)
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=timeout)
                    self._wakeup.clear()
                except asyncio.TimeoutError:
                    continue
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                await self._report_io_error(exc)
                await asyncio.sleep(max(0.05, self._config.poll_interval_ms / 1000.0))

    async def _poll_once(self, now_ms: int) -> None:
        path = resolve_spool_path(self._config.runtime_dir)
        if path != self._spool_path:
            self._reset_tail()
            self._spool_path = path
        await self._publish_state_if_changed("resolvedPath", str(path))
        if not self._directory_ready:
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            self._directory_ready = True

        try:
            tail = await asyncio.to_thread(read_appended, path, self._offset)
        except FileNotFoundError:
            await self._publish_state_if_changed("connected", False)
            await self._emit_stale_once(now_ms, reason="waiting_for_game")
            return

        self._offset = tail.offset
        if tail.truncated or tail.skipped_bytes > 0:
            self._fragment = ""
        text = self._fragment + tail.text
        complete_lines, self._fragment = self._split_complete_lines(text)
        if not complete_lines:
            await self._check_stale(now_ms)
            return

        parsed = parse_latest_frame(complete_lines, arrival_timestamp_ms=now_ms)
        if not parsed.skeletons:
            await self._check_stale(now_ms)
            return

        selected = select_frame(
            parsed.skeletons,
            reference_role=self._config.reference_role,
            target_role=self._config.target_role,
            enabled_reference_participants=self._config.enabled_reference_participants,
            enabled_target_participants=self._config.enabled_target_participants,
            enabled_reference_bones=self._config.enabled_reference_bones,
            enabled_target_bones=self._config.enabled_target_bones,
        )
        self._last_valid_ms = int(now_ms)
        self._stale_emitted = False
        await self._publish_state_if_changed("connected", True)
        await self._publish_state_if_changed("availableParticipants", available_participants(parsed.skeletons))
        await self._publish_state_if_changed("availableReferenceBones", available_bones(selected.reference_skeleton))
        await self._publish_state_if_changed("availableTargetBones", available_bones(selected.target_skeleton))

        await self.emit("skeletons", selected.skeletons, ts_ms=now_ms)
        await self.emit("referenceSkeleton", selected.reference_skeleton, ts_ms=now_ms)
        await self.emit("targetSkeleton", selected.target_skeleton, ts_ms=now_ms)
        await self.emit("referenceBone", selected.reference_bone, ts_ms=now_ms)
        await self.emit("targetBone", selected.target_bone, ts_ms=now_ms)
        await self.emit("status", selected.status, ts_ms=now_ms)
        self.record_monitor_processed(port="skeletons", ts_ms=now_ms)
        await self._clear_io_error()

    async def _check_stale(self, now_ms: int) -> None:
        if self._last_valid_ms is None:
            await self._emit_stale_once(now_ms, reason="waiting_for_hanime")
            return
        if now_ms - self._last_valid_ms >= self._config.stale_after_ms:
            await self._publish_state_if_changed("connected", False)
            await self._emit_stale_once(now_ms, reason="stale")

    async def _emit_stale_once(self, now_ms: int, *, reason: str) -> None:
        if self._stale_emitted:
            return
        self._stale_emitted = True
        status = {
            "valid": False,
            "reason": reason,
            "referenceKey": "",
            "targetKey": "",
            "referenceBone": "",
            "targetBone": "",
            "hanimeId": "",
            "hanimeAsset": "",
            "hanimeCategory": "",
        }
        await self.emit("skeletons", [], ts_ms=now_ms)
        await self.emit("referenceSkeleton", None, ts_ms=now_ms)
        await self.emit("targetSkeleton", None, ts_ms=now_ms)
        await self.emit("referenceBone", None, ts_ms=now_ms)
        await self.emit("targetBone", None, ts_ms=now_ms)
        await self.emit("status", status, ts_ms=now_ms)

    async def _emit_inactive_once(self, *, reason: str) -> None:
        self._stale_emitted = False
        await self._emit_stale_once(_now_ms(), reason=reason)
        await self._publish_state_if_changed("connected", False)

    async def _publish_state_if_changed(self, field: str, value: Any) -> None:
        if self._published_state.get(field) == value:
            return
        self._published_state[field] = value
        await self.set_state(field, value)

    async def _report_io_error(self, exc: BaseException) -> None:
        signature = f"{type(exc).__name__}:{exc}"
        if signature == self._error_signature:
            return
        self._error_signature = signature
        logger.error("Fallen Doll source read failed: %s", signature, exc_info=(type(exc), exc, exc.__traceback__))
        await self.report_error(
            "fallen_doll_source_read_failed",
            signature,
            fingerprint="fallen-doll-source-io",
        )

    async def _clear_io_error(self) -> None:
        if not self._error_signature:
            return
        self._error_signature = ""
        await self.clear_error(fingerprint="fallen-doll-source-io")

    def _reset_tail(self) -> None:
        self._offset = 0
        self._fragment = ""
        self._spool_path = None
        self._directory_ready = False
        self._last_valid_ms = None
        self._stale_emitted = False

    @staticmethod
    def _split_complete_lines(text: str) -> tuple[list[str], str]:
        if not text:
            return [], ""
        parts = text.split("\n")
        if text.endswith("\n"):
            return [part.rstrip("\r") for part in parts[:-1]], ""
        return [part.rstrip("\r") for part in parts[:-1]], parts[-1]

    @staticmethod
    def _config_from_values(values: dict[str, Any]) -> SourceConfig:
        return SourceConfig(
            runtime_dir=str(values.get("runtimeDir") or "").strip(),
            poll_interval_ms=_coerce_int(values.get("pollIntervalMs"), default=20, minimum=10, maximum=1000),
            stale_after_ms=_coerce_int(values.get("staleAfterMs"), default=250, minimum=50, maximum=10000),
            reference_role=str(values.get("referenceRole") or "male").strip().lower() or "male",
            target_role=str(values.get("targetRole") or "female").strip().lower() or "female",
            enabled_reference_participants=_string_list(
                values.get("enabledReferenceParticipants"), default=DEFAULT_REFERENCE_PARTICIPANTS
            ),
            enabled_target_participants=_string_list(
                values.get("enabledTargetParticipants"), default=DEFAULT_TARGET_PARTICIPANTS
            ),
            enabled_reference_bones=_string_list(
                values.get("enabledReferenceBones"), default=DEFAULT_FUNCTIONAL_BONES
            ),
            enabled_target_bones=_string_list(
                values.get("enabledTargetBones"), default=DEFAULT_FUNCTIONAL_BONES
            ),
        )
