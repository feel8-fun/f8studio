from __future__ import annotations

import logging
import time
from typing import Any

from f8pysdk.bus import ServiceBus
from f8pysdk.capabilities import ClosableNode, CommandableNode
from f8pysdk.codec import unwrap_json_value
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode

from .command_dispatcher import PyScriptCommandDispatcher
from .error_reporter import PyScriptErrorReporter
from .hook_invoker import PyScriptHookInvoker
from .lifecycle_state import PyScriptLifecycleState
from .local_exec import PyScriptLocalExec, PyScriptPermissionContext
from .script_context import PyScriptServiceContext
from .script_hook_runtime import PyScriptHookRuntime
from .script_runtime_values import (
    ScriptOutputPorts,
    build_script_output_ports,
    extract_script_outputs,
)
from .script_runtime import PyScriptRuntimeCompiler
from .script_templates import DEFAULT_CODE
from .state_access import PyScriptStateAccess, collect_readable_state_names
from .tick_scheduler import PyScriptTickScheduler
from .video_latest import VideoLatestConfig, VideoLatestSubscriptions

logger = logging.getLogger(__name__)
_DESTRUCTOR_CLEANUP_ERRORS = (OSError, RuntimeError, TypeError, ValueError)
_SCRIPT_OUTPUT_ERRORS = (LookupError, OSError, RuntimeError, TypeError, ValueError)
_SCRIPT_COMPILE_ERRORS = (Exception,)


class PythonScriptServiceNode(ServiceNode, CommandableNode, ClosableNode):
    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[str(p.name) for p in list(node.dataInPorts or [])],
            data_out_ports=[str(p.name) for p in list(node.dataOutPorts or [])],
            state_fields=[str(s.name) for s in list(node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

        self._code = str(self._initial_state.get("code") or DEFAULT_CODE)
        self._locals: dict[str, Any] = {}

        self._lifecycle = PyScriptLifecycleState()

        self._error_reporter = PyScriptErrorReporter(
            node_id=self.node_id,
            logger=logger,
            report_error=self.report_error,
        )
        self._state_access = PyScriptStateAccess(
            readable_state_names=collect_readable_state_names(list(node.stateFields or [])),
            state_fields=lambda: list(self.state_fields),
            get_cached=self.get_state_cached,
            set_state=self.set_state,
        )
        self._hook_invoker = PyScriptHookInvoker(
            node_id=self.node_id,
            build_context=self._build_invoke_ctx,
            set_error=self._set_error,
        )
        self._hook_runtime = PyScriptHookRuntime.empty(invoker=self._hook_invoker)

        self._tick_scheduler = PyScriptTickScheduler(
            node_id=self.node_id,
            tick_enabled=bool(self._initial_state.get("tickEnabled") or False),
            tick_ms=PyScriptTickScheduler.coerce_tick_ms(self._initial_state.get("tickMs"), default=100),
            now_ms=self._now_ms,
            is_closing=lambda: bool(self._lifecycle.closing),
            is_tick_allowed=self._is_tick_allowed,
            run_tick=self._run_tick,
            log_error=self._log_error_deduped,
        )

        self._video_latest = VideoLatestSubscriptions(
            node_id=self.node_id,
            log_context="pyscript",
            read_enabled=self._is_video_latest_read_enabled,
        )

        self._local_exec = PyScriptLocalExec(node_id=self.node_id, now_ms=self._now_ms)
        self._command_dispatcher = PyScriptCommandDispatcher(
            local_exec=self._local_exec,
            run_script_command=self._run_script_command,
        )
        self._script_runtime = PyScriptRuntimeCompiler(is_local_exec_allowed=self._local_exec.is_allowed)

        self._script_output_ports = ScriptOutputPorts(
            data_out_ports=frozenset(),
            single_data_out_port=None,
            has_out_port=False,
        )
        self._refresh_data_out_port_cache()

        self._ctx: PyScriptServiceContext = self._build_ctx()

        self._compile_and_start()

    def __del__(self) -> None:
        if self._lifecycle.should_stop_in_destructor:
            try:
                self._hook_runtime.invoke_stop_sync()
            except _DESTRUCTOR_CLEANUP_ERRORS as exc:
                logger.error("[%s:pyscript] __del__ onStop failed", self.node_id, exc_info=exc)
        try:
            self._video_latest.shutdown_sync()
        except _DESTRUCTOR_CLEANUP_ERRORS as exc:
            logger.error("[%s:pyscript] __del__ video cleanup failed", self.node_id, exc_info=exc)

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        if isinstance(bus, ServiceBus):
            cfg = bus.config
            self._video_latest.configure(
                VideoLatestConfig(
                    config_path=cfg.zenoh_config_path,
                    connect=cfg.zenoh_connect,
                    listen=cfg.zenoh_listen,
                    shm_pool_bytes=cfg.zenoh_shm_pool_bytes,
                )
            )

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000.0)

    def _set_error(self, stage: str, exc: BaseException) -> None:
        self._error_reporter.set_error(stage, exc)

    def _log_error_deduped(self, key: str, message: str, exc: BaseException) -> None:
        self._error_reporter.log_deduped(key, message, exc)

    def _permission_context(self) -> PyScriptPermissionContext:
        return self._local_exec.permission_context()

    def _build_ctx(self) -> PyScriptServiceContext:
        return PyScriptServiceContext(
            service_id=self.node_id,
            locals=self._locals,
            state_keys=self._state_access.readable_state_names,
            permission=self._permission_context(),
            build_states_view=self._state_access.build_states_view,
            emit_value=self.emit,
            set_state_value=self._state_access.set_runtime_state,
            read_state_value=self.get_state_value,
            subscribe_video_latest_value=self._video_latest.subscribe,
            get_video_latest_value=self._video_latest.get_packet,
            unsubscribe_video_latest_value=self._video_latest.unsubscribe_sync,
            list_video_latest_values=self._video_latest.list_status,
            exec_local_value=self._local_exec.exec_local,
        )

    def _build_invoke_ctx(self) -> PyScriptServiceContext:
        permission = self._permission_context()
        if self._ctx.permission == permission:
            return self._ctx
        self._ctx = self._ctx.with_permission(permission)
        return self._ctx

    def _refresh_data_out_port_cache(self) -> None:
        self._script_output_ports = build_script_output_ports(self.data_out_ports)

    def _compile_script(self, code: str) -> None:
        self._hook_runtime = PyScriptHookRuntime(hooks=self._script_runtime.compile(code), invoker=self._hook_invoker)

    def _compile_and_start(self) -> None:
        if self._lifecycle.started:
            self._hook_runtime.invoke_stop_sync()
        self._video_latest.shutdown_sync()

        self._locals = {}
        self._ctx = self._build_ctx()

        code = str(self._code or "").replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
        self._hook_runtime = PyScriptHookRuntime.empty(invoker=self._hook_invoker)

        try:
            self._compile_script(code)
        except _SCRIPT_COMPILE_ERRORS as exc:
            self._lifecycle.mark_compile_failed()
            self._set_error("compile", exc)
            return

        self._hook_runtime.invoke_start_sync()
        self._lifecycle.mark_started()
        self._tick_scheduler.ensure_task()

    async def _emit_outputs(self, result: Any) -> None:
        try:
            outputs = extract_script_outputs(result, ports=self._script_output_ports)
        except ValueError as exc:
            self._set_error("result", exc)
            return
        for out_port, out_value in outputs.items():
            try:
                await self.emit(str(out_port), out_value)
            except _SCRIPT_OUTPUT_ERRORS as exc:
                self._set_error(f"result:emit:{out_port}", exc)
                return

    def _is_video_latest_read_enabled(self) -> bool:
        return self._lifecycle.can_read_video_latest

    def _is_tick_allowed(self) -> bool:
        return self._lifecycle.can_tick

    async def _run_tick(self, tick_payload: dict[str, int]) -> None:
        result = await self._hook_runtime.invoke_tick(tick_payload)
        await self._emit_outputs(result)

    async def close(self) -> None:
        if not self._lifecycle.begin_close():
            return
        try:
            await self._hook_runtime.invoke_stop()
            self._lifecycle.mark_stopped()
            await self._tick_scheduler.shutdown()
        finally:
            await self._video_latest.shutdown_async()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        self._lifecycle.set_active(active)
        if not self._lifecycle.started:
            return

        if self._lifecycle.active:
            if self._lifecycle.paused:
                self._lifecycle.mark_resumed()
                result = await self._hook_runtime.invoke_resume(dict(meta or {}))
                await self._emit_outputs(result)
            return

        if not self._lifecycle.paused:
            self._lifecycle.mark_paused()
            result = await self._hook_runtime.invoke_pause(dict(meta or {}))
            await self._emit_outputs(result)

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        value_unwrapped = unwrap_json_value(value)
        if name == "code":
            return str(value_unwrapped or "")
        if name == "tickEnabled":
            return bool(value_unwrapped)
        if name == "tickMs":
            return PyScriptTickScheduler.coerce_tick_ms(value_unwrapped, default=self._tick_scheduler.tick_ms)
        return value_unwrapped

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "").strip()
        value_unwrapped = unwrap_json_value(value)

        if name == "code":
            self._code = str(value_unwrapped or "")
            self._compile_and_start()
            return
        if name == "tickEnabled":
            self._tick_scheduler.set_enabled(bool(value_unwrapped))
            return
        if name == "tickMs":
            self._tick_scheduler.set_tick_ms(value_unwrapped)
            return

        if self._state_access.is_self_state_write(name, value_unwrapped):
            return

        result = await self._hook_runtime.invoke_state(name, value_unwrapped, ts_ms)
        await self._emit_outputs(result)

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if not self._lifecycle.can_handle_data:
            return
        in_port = str(port or "")
        result = await self._hook_runtime.invoke_data(in_port, value, ts_ms)
        await self._emit_outputs(result)

    async def on_command(self, name: str, args: dict[str, Any] | None = None, *, meta: dict[str, Any] | None = None) -> Any:
        return await self._command_dispatcher.dispatch(name, args, meta=meta)

    async def _run_script_command(self, call: str, call_args: dict[str, Any], call_meta: dict[str, Any]) -> Any:
        if not self._hook_runtime.has_command_hook:
            raise ValueError(f"unknown command: {call}")

        return await self._hook_runtime.invoke_command(call, call_args, call_meta)
