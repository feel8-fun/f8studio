from __future__ import annotations

from f8pysdk.editor_assist_protocol import validate_editor_assist_spec
from f8pysdk.specs import F8EditorAssistSpec

_PYENGINE_STUB = """from __future__ import annotations

from typing import Any, Mapping, Protocol, TypeAlias, Literal

from f8_dynamic_inputs import F8Inputs as F8Inputs
from f8_dynamic_states import F8States as F8States
InputMode: TypeAlias = Literal['input_view', 'raw_dict', 'msgspec_struct']
F8InputsMapping: TypeAlias = Mapping[str, Any]

class F8PyEngineContext:
    \"\"\"Runtime context passed to f8.python_script hooks.\"\"\"
    node_id: str
    \"\"\"Current runtime node id.\"\"\"
    locals: dict[str, Any]
    \"\"\"Script-local mutable memory persisted between hook calls.\"\"\"
    states: F8States
    \"\"\"Readonly cached state snapshot view (rw/ro/wo fields).\"\"\"
    exec_in: str | None
    \"\"\"Current exec trigger input name for onExec, else None.\"\"\"
    input_mode: InputMode
    \"\"\"Current input binding mode selected by node state `inputMode`.\"\"\"
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

class PyEngineOnStartHook(Protocol):
    \"\"\"Type signature for `onStart(ctx)`.\"\"\"
    def __call__(self, ctx: F8PyEngineContext) -> Any: ...

class PyEngineOnStateHook(Protocol):
    \"\"\"Type signature for `onState(ctx, field, value, ts_ms=None)`.\"\"\"
    def __call__(self, ctx: F8PyEngineContext, field: str, value: Any, ts_ms: int | None = None) -> Any: ...

class PyEngineOnMsgHook(Protocol):
    \"\"\"Type signature for `onMsg(ctx, inputs)`.\"\"\"
    def __call__(self, ctx: F8PyEngineContext, inputs: F8Inputs | F8InputsMapping) -> Any: ...

class PyEngineOnExecHook(Protocol):
    \"\"\"Type signature for `onExec(ctx, exec_in, inputs)`.\"\"\"
    def __call__(self, ctx: F8PyEngineContext, exec_in: str, inputs: F8Inputs | F8InputsMapping) -> Any: ...

class PyEngineOnStopHook(Protocol):
    \"\"\"Type signature for `onStop(ctx)`.\"\"\"
    def __call__(self, ctx: F8PyEngineContext) -> Any: ...
"""

_PYENGINE_OVERLAY = """from __future__ import annotations
from f8_script_api import (
    PyEngineOnExecHook as _F8OnExecHook,
    PyEngineOnMsgHook as _F8OnMsgHook,
    PyEngineOnStartHook as _F8OnStartHook,
    PyEngineOnStateHook as _F8OnStateHook,
    PyEngineOnStopHook as _F8OnStopHook,
)
onStart: _F8OnStartHook
onState: _F8OnStateHook
onMsg: _F8OnMsgHook
onExec: _F8OnExecHook
onStop: _F8OnStopHook
"""


def python_script_field_editor_assist_payload() -> F8EditorAssistSpec:
    return validate_editor_assist_spec(
        {
            "version": 1,
            "language": "python",
            "python": {
                "support_files": {"f8_script_api.pyi": _PYENGINE_STUB},
                "overlay_prefix": _PYENGINE_OVERLAY,
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
