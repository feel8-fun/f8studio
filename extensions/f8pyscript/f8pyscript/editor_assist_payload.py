from __future__ import annotations

from f8pysdk.editor_assist_protocol import validate_editor_assist_spec
from f8pysdk.specs import F8EditorAssistSpec

_PYSCRIPT_STUB = """from __future__ import annotations

from typing import Any, Protocol

from f8_dynamic_states import F8States as F8States

class F8Tick:
    \"\"\"Payload for `onTick(ctx, tick)`.\"\"\"
    seq: int
    tsMs: int
    deltaMs: int

class F8Permission:
    \"\"\"Local-exec permission snapshot available on `ctx.permission`.\"\"\"
    local_exec_granted: bool
    \"\"\"Whether local process execution is currently allowed.\"\"\"
    expires_ts_ms: int | None
    \"\"\"Grant expiration timestamp (ms), or None for no expiry.\"\"\"
    grant_ts_ms: int
    \"\"\"Grant issue timestamp (ms).\"\"\"
    session_id: str
    \"\"\"Grant/session identifier.\"\"\"

class F8PyScriptContext:
    \"\"\"Runtime context passed to f8.pyscript hooks.\"\"\"
    service_id: str
    \"\"\"Current service id.\"\"\"
    locals: dict[str, Any]
    \"\"\"Script-local mutable memory persisted between hook calls.\"\"\"
    states: F8States
    \"\"\"Readonly cached state snapshot view (rw/ro/wo fields).\"\"\"
    permission: F8Permission
    \"\"\"Permission snapshot for local execution and imports.\"\"\"
    def log(self, message: object) -> None:
        \"\"\"Write an info log line from script context.\"\"\"
        ...
    def emit(self, port: str, value: Any) -> None:
        \"\"\"Emit one value to a data output port (fire-and-forget).\"\"\"
        ...
    async def emit_async(self, port: str, value: Any) -> None:
        \"\"\"Emit one value to a data output port and await completion.\"\"\"
        ...
    def set_state(self, field: str, value: Any) -> None:
        \"\"\"Set a state value asynchronously (fire-and-forget).\"\"\"
        ...
    async def set_state_async(self, field: str, value: Any) -> None:
        \"\"\"Set a state value and await the write completion.\"\"\"
        ...
    async def read_state(self, field: str) -> Any:
        \"\"\"Read a fresh state value via runtime/state service path.\"\"\"
        ...
    def subscribe_video_latest(self, key: str, *, stream_key: str = '', decode: str = 'auto') -> None:
        \"\"\"Subscribe to a latest-frame video stream by key.\"\"\"
        ...
    def get_video_latest(self, key: str) -> dict[str, Any] | None:
        \"\"\"Get latest cached video packet for a subscription key.\"\"\"
        ...
    def unsubscribe_video_latest(self, key: str) -> None:
        \"\"\"Cancel one latest-frame video subscription by key.\"\"\"
        ...
    def list_video_latest_subscriptions(self) -> list[dict[str, Any]]:
        \"\"\"List active latest-frame video subscription metadata.\"\"\"
        ...
    async def exec_local(self, command: str, args: list[str] | tuple[str, ...] | None = None, *, timeout_ms: int | None = None, cwd: str | None = None, env: dict[str, Any] | None = None) -> dict[str, Any]:
        \"\"\"Execute a local process when `permission.local_exec_granted` is True.\"\"\"
        ...

class PyScriptOnStartHook(Protocol):
    \"\"\"Type signature for `onStart(ctx)`.\"\"\"
    def __call__(self, ctx: F8PyScriptContext) -> Any: ...

class PyScriptOnStopHook(Protocol):
    \"\"\"Type signature for `onStop(ctx)`.\"\"\"
    def __call__(self, ctx: F8PyScriptContext) -> Any: ...

class PyScriptOnPauseHook(Protocol):
    \"\"\"Type signature for `onPause(ctx, meta=None)`.\"\"\"
    def __call__(self, ctx: F8PyScriptContext, meta: dict[str, Any] | None = None) -> Any: ...

class PyScriptOnResumeHook(Protocol):
    \"\"\"Type signature for `onResume(ctx, meta=None)`.\"\"\"
    def __call__(self, ctx: F8PyScriptContext, meta: dict[str, Any] | None = None) -> Any: ...

class PyScriptOnStateHook(Protocol):
    \"\"\"Type signature for `onState(ctx, field, value, ts_ms=None)`.\"\"\"
    def __call__(self, ctx: F8PyScriptContext, field: str, value: Any, ts_ms: int | None = None) -> Any: ...

class PyScriptOnDataHook(Protocol):
    \"\"\"Type signature for `onData(ctx, port, value, ts_ms=None)`.\"\"\"
    def __call__(self, ctx: F8PyScriptContext, port: str, value: Any, ts_ms: int | None = None) -> Any: ...

class PyScriptOnTickHook(Protocol):
    \"\"\"Type signature for `onTick(ctx, tick)`.\"\"\"
    def __call__(self, ctx: F8PyScriptContext, tick: F8Tick) -> Any: ...

class PyScriptOnCommandHook(Protocol):
    \"\"\"Type signature for `onCommand(ctx, name, args, meta=None)`.\"\"\"
    def __call__(self, ctx: F8PyScriptContext, name: str, args: dict[str, Any], meta: dict[str, Any] | None = None) -> Any: ...
"""

_PYSCRIPT_OVERLAY = """from __future__ import annotations
from f8_script_api import (
    PyScriptOnCommandHook as _F8OnCommandHook,
    PyScriptOnDataHook as _F8OnDataHook,
    PyScriptOnPauseHook as _F8OnPauseHook,
    PyScriptOnResumeHook as _F8OnResumeHook,
    PyScriptOnStartHook as _F8OnStartHook,
    PyScriptOnStateHook as _F8OnStateHook,
    PyScriptOnStopHook as _F8OnStopHook,
    PyScriptOnTickHook as _F8OnTickHook,
)
onStart: _F8OnStartHook
onStop: _F8OnStopHook
onPause: _F8OnPauseHook
onResume: _F8OnResumeHook
onState: _F8OnStateHook
onData: _F8OnDataHook
onTick: _F8OnTickHook
onCommand: _F8OnCommandHook
"""


def pyscript_code_field_editor_assist_payload() -> F8EditorAssistSpec:
    return validate_editor_assist_spec(
        {
            "version": 1,
            "language": "python",
            "python": {
                "support_files": {"f8_script_api.pyi": _PYSCRIPT_STUB},
                "overlay_prefix": _PYSCRIPT_OVERLAY,
                "dynamic_bindings": {
                    "inputs": {
                        "enabled": True,
                        "source": "data_in_ports",
                        "type_name": "F8Inputs",
                        "module_name": "f8_dynamic_inputs",
                        "schema_mode": "basic_recursive",
                        "access_mode": "object_and_mapping",
                    },
                    "states": {
                        "enabled": True,
                        "source": "state_fields",
                        "type_name": "F8States",
                        "module_name": "f8_dynamic_states",
                        "schema_mode": "basic_recursive",
                        "access_mode": "object_and_mapping",
                    }
                },
            },
        }
    )
