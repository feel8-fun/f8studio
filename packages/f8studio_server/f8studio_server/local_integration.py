from __future__ import annotations

from f8studio_server.errors import InvalidRequestError, NotFoundError

import asyncio
import concurrent.futures
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from contextlib import AbstractContextManager

from .database import StudioDatabase
from threading import RLock
from typing import Literal, cast
from uuid import uuid4

import msgspec

from .native_hotkeys import (
    NativeHotkeyBackend,
    NativeHotkeyBinding,
    NativeHotkeyRegistrationError,
    NativeHotkeyUnsupportedError,
    create_native_hotkey_backend,
    parse_native_hotkey,
)


logger = logging.getLogger(__name__)


class HotkeyBinding(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    binding_id: str
    accelerator: str
    project_id: str
    node_id: str
    field: str
    status: Literal["configured", "registered", "disabled", "error"] = "configured"
    message: str = ""


class RegisterHotkeyRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    accelerator: str
    project_id: str
    node_id: str
    field: str
    binding_id: str | None = None


HotkeyActivation = Callable[[HotkeyBinding], Awaitable[None]]
HotkeyValidator = Callable[[HotkeyBinding], None]


class LocalIntegrationService:
    def __init__(
        self,
        *,
        database_path: Path | None = None,
        hotkey_backend: NativeHotkeyBackend | None = None,
        hotkey_activation: HotkeyActivation | None = None,
        hotkey_validator: HotkeyValidator | None = None,
    ) -> None:
        self._database_path = database_path.resolve() if database_path is not None else None
        self._hotkeys = self._load_hotkeys()
        self._registered_hotkeys: dict[str, HotkeyBinding] = {}
        self._hotkey_status: dict[str, tuple[Literal["registered", "disabled", "error"], str]] = {}
        self._hotkey_backend = hotkey_backend
        self._hotkey_backend_injected = hotkey_backend is not None
        self._hotkey_backend_error = ""
        self._hotkey_activation = hotkey_activation
        self._hotkey_validator = hotkey_validator
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._activation_futures: set[concurrent.futures.Future[None]] = set()
        self._hotkey_lock = RLock()

    async def start(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        if self._hotkey_backend is None and not self._hotkey_backend_injected:
            try:
                self._hotkey_backend = await asyncio.to_thread(
                    create_native_hotkey_backend,
                    self._on_native_hotkey,
                )
            except NativeHotkeyUnsupportedError as exc:
                self._hotkey_backend_error = str(exc)
                logger.info("native global hotkeys unavailable: %s", exc)
            except Exception as exc:
                self._hotkey_backend_error = f"{type(exc).__name__}: {exc}"
                logger.exception("native global hotkey backend initialization failed", exc_info=exc)
        await asyncio.to_thread(self._refresh_hotkey_backend)

    async def close(self) -> None:
        backend = self._hotkey_backend
        self._hotkey_backend = None
        with self._hotkey_lock:
            self._registered_hotkeys.clear()
        if backend is not None:
            try:
                await asyncio.to_thread(backend.close)
            except Exception as exc:
                logger.exception("native global hotkey backend close failed", exc_info=exc)
        for future in tuple(self._activation_futures):
            future.cancel()
        self._activation_futures.clear()
        self._event_loop = None

    def list_hotkeys(self, project_id: str | None = None) -> tuple[HotkeyBinding, ...]:
        with self._hotkey_lock:
            return tuple(
                self._binding_with_status(binding)
                for binding in self._hotkeys.values()
                if project_id is None or binding.project_id == project_id
            )

    def register_hotkey(self, request: RegisterHotkeyRequest) -> HotkeyBinding:
        with self._hotkey_lock:
            accelerator = parse_native_hotkey(request.accelerator).display_text
            binding_id = request.binding_id or uuid4().hex
            conflict = next(
                (
                    binding
                    for binding in self._hotkeys.values()
                    if binding.accelerator == accelerator and binding.binding_id != binding_id
                ),
                None,
            )
            if conflict is not None:
                raise InvalidRequestError(f"hotkey accelerator is already registered: {accelerator}")
            binding = HotkeyBinding(
                binding_id=binding_id,
                accelerator=accelerator,
                project_id=request.project_id,
                node_id=request.node_id,
                field=request.field,
            )
            if self._hotkey_validator is not None:
                self._hotkey_validator(binding)
            self._save_hotkey(binding)
            self._hotkeys[binding.binding_id] = binding
            self._refresh_hotkey_backend()
            return self._binding_with_status(binding)

    def unregister_hotkey(self, binding_id: str) -> None:
        with self._hotkey_lock:
            if binding_id not in self._hotkeys:
                raise NotFoundError(f"hotkey binding not found: {binding_id}")
            self._delete_hotkey(binding_id)
            self._hotkeys.pop(binding_id)
            self._refresh_hotkey_backend()

    def refresh_hotkeys(self) -> None:
        with self._hotkey_lock:
            self._refresh_hotkey_backend()

    def forget_project_hotkeys(self, project_id: str) -> None:
        with self._hotkey_lock:
            self._hotkeys = {
                binding_id: binding
                for binding_id, binding in self._hotkeys.items()
                if binding.project_id != project_id
            }
            self._refresh_hotkey_backend()

    def _refresh_hotkey_backend(self) -> None:
        with self._hotkey_lock:
            self._refresh_hotkey_backend_locked()

    def _refresh_hotkey_backend_locked(self) -> None:
        if self._hotkey_backend is None:
            message = self._hotkey_backend_error or "Native global hotkey backend is not running"
            self._hotkey_status = {key: ("disabled", message) for key in self._hotkeys}
            self._registered_hotkeys.clear()
            return
        # Graph edits that do not change hotkey targets must not ungrab keys.
        valid: dict[str, HotkeyBinding] = {}
        invalid: dict[str, tuple[Literal["registered", "disabled", "error"], str]] = {}
        for binding_id, binding in self._hotkeys.items():
            try:
                if self._hotkey_validator is not None:
                    self._hotkey_validator(binding)
                valid[binding_id] = binding
            except (FileNotFoundError, ValueError) as exc:
                invalid[binding_id] = ("error", str(exc))
                logger.warning("hotkey target is no longer valid binding_id=%s", binding_id, exc_info=exc)
        if valid == self._registered_hotkeys and set(valid) == set(self._hotkeys):
            return
        self._registered_hotkeys = {}
        backend = self._hotkey_backend
        self._hotkey_status.clear()
        try:
            backend.unregister_all()
        except Exception as exc:
            logger.exception("failed to clear native global hotkeys before refresh", exc_info=exc)
            message = f"{type(exc).__name__}: {exc}"
            self._hotkey_status.update((binding_id, ("error", message)) for binding_id in self._hotkeys)
            return
        self._hotkey_status.update(invalid)
        for binding in valid.values():
            try:
                backend.register_hotkey(
                    NativeHotkeyBinding(
                        binding_id=binding.binding_id,
                        spec=parse_native_hotkey(binding.accelerator),
                    )
                )
            except NativeHotkeyRegistrationError as exc:
                self._hotkey_status[binding.binding_id] = ("error", str(exc))
                logger.warning(
                    "native global hotkey registration failed binding_id=%s accelerator=%s",
                    binding.binding_id,
                    binding.accelerator,
                    exc_info=exc,
                )
            else:
                self._hotkey_status[binding.binding_id] = ("registered", "")
                self._registered_hotkeys[binding.binding_id] = binding

    def _binding_with_status(self, binding: HotkeyBinding) -> HotkeyBinding:
        status, message = self._hotkey_status.get(binding.binding_id, ("configured", ""))
        return msgspec.structs.replace(binding, status=status, message=message)

    def _on_native_hotkey(self, binding_id: str) -> None:
        loop = self._event_loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._dispatch_native_hotkey, binding_id)

    def _dispatch_native_hotkey(self, binding_id: str) -> None:
        with self._hotkey_lock:
            binding = self._hotkeys.get(binding_id)
        loop = self._event_loop
        activation = self._hotkey_activation
        if binding is None or loop is None or activation is None or loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(self._run_hotkey_activation(binding), loop)
        self._activation_futures.add(future)
        future.add_done_callback(self._activation_done)

    async def _run_hotkey_activation(self, binding: HotkeyBinding) -> None:
        activation = self._hotkey_activation
        if activation is not None:
            await activation(binding)

    def _activation_done(self, future: concurrent.futures.Future[None]) -> None:
        self._activation_futures.discard(future)
        if future.cancelled():
            return
        try:
            future.result()
        except Exception:
            logger.exception("native global hotkey action failed")

    def _connect_hotkeys(self) -> AbstractContextManager[sqlite3.Connection]:
        if self._database_path is None:
            raise RuntimeError("hotkey persistence is not configured")
        return StudioDatabase(self._database_path).connection()

    def _load_hotkeys(self) -> dict[str, HotkeyBinding]:
        if self._database_path is None:
            return {}
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect_hotkeys() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS global_hotkeys (
                    binding_id TEXT PRIMARY KEY,
                    accelerator TEXT NOT NULL UNIQUE,
                    project_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    field_name TEXT NOT NULL
                )
                """
            )
            rows = connection.execute(
                "SELECT binding_id, accelerator, project_id, node_id, field_name "
                "FROM global_hotkeys ORDER BY binding_id"
            ).fetchall()
        bindings: dict[str, HotkeyBinding] = {}
        for row in rows:
            if not all(isinstance(value, str) for value in row):
                raise TypeError("global_hotkeys row contains a non-text value")
            binding = HotkeyBinding(
                binding_id=cast(str, row[0]),
                accelerator=cast(str, row[1]),
                project_id=cast(str, row[2]),
                node_id=cast(str, row[3]),
                field=cast(str, row[4]),
            )
            bindings[binding.binding_id] = binding
        return bindings

    def _save_hotkey(self, binding: HotkeyBinding) -> None:
        if self._database_path is None:
            return
        try:
            with self._connect_hotkeys() as connection:
                connection.execute(
                    """
                    INSERT INTO global_hotkeys(binding_id, accelerator, project_id, node_id, field_name)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(binding_id) DO UPDATE SET
                        accelerator = excluded.accelerator,
                        project_id = excluded.project_id,
                        node_id = excluded.node_id,
                        field_name = excluded.field_name
                    """,
                    (
                        binding.binding_id,
                        binding.accelerator,
                        binding.project_id,
                        binding.node_id,
                        binding.field,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise InvalidRequestError(f"hotkey accelerator is already registered: {binding.accelerator}") from exc

    def _delete_hotkey(self, binding_id: str) -> None:
        if self._database_path is None:
            return
        with self._connect_hotkeys() as connection:
            connection.execute("DELETE FROM global_hotkeys WHERE binding_id = ?", (binding_id,))

__all__ = [
    "HotkeyBinding",
    "LocalIntegrationService",
    "RegisterHotkeyRequest",
]
