from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .local_exec import PyScriptPermissionContext
from .script_runtime_values import PyScriptStatesView


logger = logging.getLogger(__name__)


class VideoLatestSubscribeCallback(Protocol):
    def __call__(self, key: str, *, stream_key: str, decode: str) -> None: ...


class LocalExecCallback(Protocol):
    def __call__(
        self,
        command: str,
        args: list[str] | tuple[str, ...] | None = None,
        *,
        timeout_ms: int | None = None,
        cwd: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> Awaitable[dict[str, Any]]: ...


@dataclass(slots=True)
class PyScriptServiceContext:
    service_id: str
    locals: dict[str, Any]
    state_keys: tuple[str, ...]
    permission: PyScriptPermissionContext
    build_states_view: Callable[[tuple[str, ...]], PyScriptStatesView]
    emit_value: Callable[[str, Any], Awaitable[None]]
    set_state_value: Callable[[str, Any], Awaitable[None]]
    read_state_value: Callable[[str], Awaitable[Any]]
    subscribe_video_latest_value: VideoLatestSubscribeCallback
    get_video_latest_value: Callable[[str], dict[str, Any] | None]
    unsubscribe_video_latest_value: Callable[[str], object]
    list_video_latest_values: Callable[[], list[dict[str, Any]]]
    exec_local_value: LocalExecCallback

    def with_permission(self, permission: PyScriptPermissionContext) -> "PyScriptServiceContext":
        return replace(self, permission=permission)

    @property
    def states(self) -> PyScriptStatesView:
        return self.build_states_view(self.state_keys)

    def log(self, message: object) -> None:
        logger.info("[%s:pyscript] %s", self.service_id, str(message))

    async def emit_async(self, port: str, value: Any) -> None:
        await self.emit_value(str(port), value)

    def emit(self, port: str, value: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            logger.error("[%s:pyscript] emit without running loop", self.service_id, exc_info=exc)
            return
        loop.create_task(self.emit_async(str(port), value), name=f"pyscript:emit:{self.service_id}:{port}")

    async def set_state_async(self, field: str, value: Any) -> None:
        await self.set_state_value(str(field), value)

    def set_state(self, field: str, value: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            logger.error("[%s:pyscript] set_state without running loop", self.service_id, exc_info=exc)
            return
        loop.create_task(
            self.set_state_async(str(field), value),
            name=f"pyscript:set_state:{self.service_id}:{field}",
        )

    async def read_state(self, field: str) -> Any:
        return await self.read_state_value(str(field))

    def subscribe_video_latest(
        self,
        key: str,
        *,
        stream_key: str = "",
        decode: str = "auto",
    ) -> None:
        key_name = str(key or "").strip()
        stream_key_text = str(stream_key or "").strip()
        if not key_name:
            return
        if not stream_key_text:
            return
        self.subscribe_video_latest_value(key_name, stream_key=stream_key_text, decode=decode)

    def get_video_latest(self, key: str) -> dict[str, Any] | None:
        key_name = str(key or "").strip()
        if not key_name:
            return None
        return self.get_video_latest_value(key_name)

    def unsubscribe_video_latest(self, key: str) -> None:
        self.unsubscribe_video_latest_value(str(key or "").strip())

    def list_video_latest_subscriptions(self) -> list[dict[str, Any]]:
        return self.list_video_latest_values()

    async def exec_local(
        self,
        command: str,
        args: list[str] | tuple[str, ...] | None = None,
        *,
        timeout_ms: int | None = None,
        cwd: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.exec_local_value(
            command,
            args,
            timeout_ms=timeout_ms,
            cwd=cwd,
            env=env,
        )


__all__ = [
    "LocalExecCallback",
    "PyScriptServiceContext",
    "VideoLatestSubscribeCallback",
]
