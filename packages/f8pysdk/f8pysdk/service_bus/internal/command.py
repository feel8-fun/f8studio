"""Internal command execution owner boundary for `ServiceBus`."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

from ...capabilities import CommandableNode
from ...command import CommandExecutionErrorKind, CommandExecutionResult, CommandOutputPolicy
from ...command import (
    command_input_state_field,
    command_key_for_name,
    command_output_state_field,
)
from ...generated import F8Command, F8OperatorSpec, F8ServiceSpec
from ...state import StateWriteOrigin, StateWriteSource
from .logging import log_error_once

if TYPE_CHECKING:
    from ..runtime import ServiceBus


log = logging.getLogger(__name__)


_HIDDEN_COMMAND_FORWARD_EXCLUDED_KEYS = frozenset(("value", "actor", "ts", "origin", "seq"))


@dataclass(frozen=True)
class CommandBinding:
    node_id: str
    command_name: str
    command_key: str
    input_field: str
    output_field: str
    param_names: tuple[str, ...]


@dataclass(frozen=True)
class CommandInvocation:
    node_id: str
    call: str
    args: Any = None


@dataclass(frozen=True)
class CommandInvokeOptions:
    call_meta: dict[str, Any] = field(default_factory=dict)
    output_policy: CommandOutputPolicy = CommandOutputPolicy.none
    output_ts_ms: int | None = None
    output_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _CommandDispatchState:
    running: bool = False
    pending: bool = False
    latest_value: Any = None
    latest_ts_ms: int | None = None
    latest_meta: dict[str, Any] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class _NormalizedCommandArgs:
    args: dict[str, Any]
    error_message: str | None = None


def _command_specs_for_node(node: Any) -> list[F8Command]:
    try:
        spec = node.spec
    except (AttributeError, RuntimeError, TypeError):
        spec = None
    if isinstance(spec, (F8ServiceSpec, F8OperatorSpec)):
        return list(spec.commands or [])
    return []


def build_command_bindings(nodes: Mapping[str, Any]) -> tuple[
    dict[tuple[str, str], CommandBinding],
    dict[tuple[str, str], CommandBinding],
    set[tuple[str, str]],
]:
    input_bindings: dict[tuple[str, str], CommandBinding] = {}
    output_bindings: dict[tuple[str, str], CommandBinding] = {}
    hidden_fields: set[tuple[str, str]] = set()

    for node_id, node in list(nodes.items()):
        for command in _command_specs_for_node(node):
            command_name = str(command.name or "").strip()
            if not command_name:
                continue
            command_key = command_key_for_name(command_name)
            input_field = command_input_state_field(command_name)
            output_field = command_output_state_field(command_name)
            binding = CommandBinding(
                node_id=str(node_id),
                command_name=command_name,
                command_key=command_key,
                input_field=input_field,
                output_field=output_field,
                param_names=tuple(str(param.name or "").strip() for param in list(command.params or []) if str(param.name or "").strip()),
            )
            input_bindings[(binding.node_id, binding.input_field)] = binding
            output_bindings[(binding.node_id, binding.command_name)] = binding
            hidden_fields.add((binding.node_id, binding.input_field))
            hidden_fields.add((binding.node_id, binding.output_field))
    return input_bindings, output_bindings, hidden_fields


def map_command_args(value: Any, param_names: tuple[str, ...]) -> dict[str, Any]:
    if value is None:
        return {}
    if not param_names:
        return {}
    if isinstance(value, dict):
        args: dict[str, Any] = {}
        for name in param_names:
            if name in value:
                args[name] = value[name]
        return args
    if isinstance(value, (list, tuple)):
        args = {}
        for index, item in enumerate(value):
            if index >= len(param_names):
                break
            args[param_names[index]] = item
        return args
    return {param_names[0]: value}


def build_hidden_command_call_meta(*, command_input_field: str, meta: dict[str, Any]) -> dict[str, Any]:
    """
    Metadata forwarded to `node.on_command(...)` for hidden-state command inputs.
    """
    call_meta: dict[str, Any] = {}
    for key, value in dict(meta).items():
        if key in _HIDDEN_COMMAND_FORWARD_EXCLUDED_KEYS:
            continue
        call_meta[str(key)] = value
    call_meta.setdefault("source", StateWriteSource.state_edge_intra.value)
    call_meta.setdefault("commandInputField", str(command_input_field))
    return call_meta


def build_command_output_meta(*, command_name: str, command_input_field: str) -> dict[str, Any]:
    """
    Metadata persisted on hidden command output state writeback.
    """
    return {
        "command": str(command_name),
        "commandInputField": str(command_input_field),
        "source": StateWriteSource.cmd.value,
    }


class CommandGateway:
    def __init__(self, *, bus: "ServiceBus", nodes: dict[str, Any]) -> None:
        self._bus = bus
        self._nodes = nodes
        self._input_bindings: dict[tuple[str, str], CommandBinding] = {}
        self._output_bindings: dict[tuple[str, str], CommandBinding] = {}
        self._hidden_fields: set[tuple[str, str]] = set()
        self._dispatch_states: dict[tuple[str, str], _CommandDispatchState] = {}

    def refresh_bindings(self) -> None:
        input_bindings, output_bindings, hidden_fields = build_command_bindings(self._nodes)
        self._input_bindings = input_bindings
        self._output_bindings = output_bindings
        self._hidden_fields = hidden_fields
        active_dispatch_keys = set(output_bindings.keys())
        stale_dispatch_keys = [key for key in self._dispatch_states if key not in active_dispatch_keys]
        for key in stale_dispatch_keys:
            del self._dispatch_states[key]

    def input_binding(self, *, node_id: str, field: str) -> CommandBinding | None:
        return self._input_bindings.get((str(node_id), str(field)))

    def output_binding(self, *, node_id: str, call: str) -> CommandBinding | None:
        binding = self._output_bindings.get((str(node_id), str(call)))
        if binding is not None:
            return binding
        return self._binding_from_node_spec(node_id=str(node_id), call=str(call))

    def is_hidden_field(self, *, node_id: str, field: str) -> bool:
        return (str(node_id), str(field)) in self._hidden_fields

    async def write_output(
        self,
        *,
        node_id: str,
        call: str,
        result: Any,
        ts_ms: int | None,
        meta: dict[str, Any] | None,
    ) -> None:
        from ..state.pipeline import publish_state

        binding = self.output_binding(node_id=str(node_id), call=str(call))
        if binding is None:
            return
        payload_meta = dict(meta or {})
        payload_meta.setdefault("command", str(call))
        await publish_state(
            self._bus,
            binding.node_id,
            binding.output_field,
            result,
            origin=StateWriteOrigin.runtime,
            source=StateWriteSource.cmd,
            ts_ms=ts_ms,
            meta=payload_meta,
        )

    async def invoke(
        self,
        *,
        invocation: CommandInvocation,
        options: CommandInvokeOptions | None = None,
    ) -> CommandExecutionResult:
        invoke_options = options if options is not None else CommandInvokeOptions()
        node_id = str(invocation.node_id)
        call = str(invocation.call or "").strip()
        service_node = self._bus.get_node(node_id)
        if service_node is None or not isinstance(service_node, CommandableNode):
            log_error_once(
                self._bus,
                key=f"command_execute_missing_target:{node_id}:{call}",
                message=f"command target missing or not commandable: {node_id}.{call}",
            )
            return CommandExecutionResult(
                error_kind=CommandExecutionErrorKind.missing_target,
                error_message=f"unknown call: {call}",
            )

        normalized = self._normalize_args(node_id=node_id, call=call, value=invocation.args)
        if normalized.error_message is not None:
            return CommandExecutionResult(
                error_kind=CommandExecutionErrorKind.invalid_args,
                error_message=normalized.error_message,
            )

        try:
            result = await service_node.on_command(call, normalized.args, meta=dict(invoke_options.call_meta))  # type: ignore[misc]
        except Exception as exc:
            log_error_once(
                self._bus,
                key=f"command_execute_failed:{node_id}:{call}:{type(exc).__name__}:{exc}",
                message=f"command dispatch failed for {node_id}.{call}",
                exc=exc,
            )
            return CommandExecutionResult(
                error_kind=CommandExecutionErrorKind.handler_failed,
                error_message=str(exc),
            )

        if invoke_options.output_policy == CommandOutputPolicy.hidden_state:
            try:
                await self.write_output(
                    node_id=node_id,
                    call=call,
                    result=result,
                    ts_ms=invoke_options.output_ts_ms,
                    meta=dict(invoke_options.output_meta),
                )
            except Exception as exc:
                log_error_once(
                    self._bus,
                    key=f"command_output_writeback_failed:{node_id}:{call}:{type(exc).__name__}:{exc}",
                    message=f"command output writeback failed for {node_id}.{call}",
                    exc=exc,
                )

        return CommandExecutionResult(value=result)

    async def dispatch_hidden_input(
        self,
        *,
        node_id: str,
        field: str,
        value: Any,
        ts_ms: int,
        meta: dict[str, Any],
    ) -> None:
        binding = self.input_binding(node_id=str(node_id), field=str(field))
        if binding is None:
            return
        dispatch_state = self._dispatch_states.setdefault(
            (binding.node_id, binding.command_name),
            _CommandDispatchState(),
        )

        async with dispatch_state.lock:
            dispatch_state.latest_value = value
            dispatch_state.latest_ts_ms = int(ts_ms)
            dispatch_state.latest_meta = dict(meta)
            if dispatch_state.running:
                dispatch_state.pending = True
                return
            dispatch_state.running = True

        try:
            while True:
                async with dispatch_state.lock:
                    dispatch_state.pending = False
                    latest_value = dispatch_state.latest_value
                    latest_ts_ms = dispatch_state.latest_ts_ms
                    latest_meta = dict(dispatch_state.latest_meta)
                await self._dispatch_hidden_once(
                    binding=binding,
                    value=latest_value,
                    ts_ms=latest_ts_ms,
                    meta=latest_meta,
                )
                async with dispatch_state.lock:
                    if not dispatch_state.pending:
                        dispatch_state.running = False
                        return
        except Exception:
            async with dispatch_state.lock:
                dispatch_state.running = False
            raise

    async def _dispatch_hidden_once(
        self,
        *,
        binding: CommandBinding,
        value: Any,
        ts_ms: int | None,
        meta: dict[str, Any],
    ) -> None:
        result = await self.invoke(
            invocation=CommandInvocation(
                node_id=binding.node_id,
                call=binding.command_name,
                args=value,
            ),
            options=CommandInvokeOptions(
                call_meta=_hidden_state_command_meta(binding=binding, meta=meta),
                output_policy=CommandOutputPolicy.hidden_state,
                output_ts_ms=ts_ms,
                output_meta=_hidden_state_output_meta(binding=binding),
            ),
        )
        if not result.ok:
            return

    def _normalize_args(self, *, node_id: str, call: str, value: Any) -> _NormalizedCommandArgs:
        binding = self.output_binding(node_id=node_id, call=call)
        if binding is not None:
            return _NormalizedCommandArgs(args=map_command_args(value, binding.param_names))

        if value is None:
            return _NormalizedCommandArgs(args={})
        if isinstance(value, dict):
            return _NormalizedCommandArgs(args={str(key): item for key, item in value.items()})
        return _NormalizedCommandArgs(
            args={},
            error_message="command args must be an object when the command has no declared params",
        )

    def _binding_from_node_spec(self, *, node_id: str, call: str) -> CommandBinding | None:
        node = self._nodes.get(str(node_id))
        if node is None:
            return None
        for command in _command_specs_for_node(node):
            command_name = str(command.name or "").strip()
            if command_name != str(call or "").strip():
                continue
            command_key = command_key_for_name(command_name)
            return CommandBinding(
                node_id=str(node_id),
                command_name=command_name,
                command_key=command_key,
                input_field=command_input_state_field(command_name),
                output_field=command_output_state_field(command_name),
                param_names=tuple(
                    str(param.name or "").strip()
                    for param in list(command.params or [])
                    if str(param.name or "").strip()
                ),
            )
        return None


def _hidden_state_command_meta(*, binding: CommandBinding, meta: dict[str, Any]) -> dict[str, Any]:
    return build_hidden_command_call_meta(command_input_field=binding.input_field, meta=meta)


def _hidden_state_output_meta(*, binding: CommandBinding) -> dict[str, Any]:
    return build_command_output_meta(
        command_name=binding.command_name,
        command_input_field=binding.input_field,
    )
async def write_command_output(
    bus: "ServiceBus",
    *,
    node_id: str,
    call: str,
    result: Any,
    ts_ms: int | None,
    meta: dict[str, Any] | None,
) -> None:
    await bus.command_gateway.write_output(
        node_id=node_id,
        call=call,
        result=result,
        ts_ms=ts_ms,
        meta=meta,
    )


async def execute_command(
    bus: "ServiceBus",
    *,
    invocation: CommandInvocation,
    options: CommandInvokeOptions | None = None,
) -> CommandExecutionResult:
    return await bus.command_gateway.invoke(invocation=invocation, options=options)


async def dispatch_command_input(
    bus: "ServiceBus",
    *,
    node_id: str,
    field: str,
    value: Any,
    ts_ms: int,
    meta: dict[str, Any],
) -> None:
    await bus.command_gateway.dispatch_hidden_input(
        node_id=node_id,
        field=field,
        value=value,
        ts_ms=ts_ms,
        meta=meta,
    )
