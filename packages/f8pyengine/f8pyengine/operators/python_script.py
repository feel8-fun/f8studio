from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    string_schema,
)
from f8pysdk.capabilities import ClosableNode
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

OPERATOR_CLASS = "f8.python_script"
logger = logging.getLogger(__name__)


DEFAULT_CODE = (
    "# Define hooks (any subset is ok):\n"
    "# - onStart(ctx)\n"
    "# - onState(ctx, field, value, tsMs=None)      # called on state updates (except 'code')\n"
    "# - onMsg(ctx, inputs)                        # called on data arrival (push mode) or as fallback for exec\n"
    "# - onExec(ctx, execIn, inputs)               # called on exec trigger (pull mode)\n"
    "# - onStop(ctx)\n"
    "# - ctx['state'] is preserved between calls\n"
    "# - ctx['execIn'] is set for exec-triggered calls\n"
    "# - State helpers:\n"
    "#   - await ctx['get_state'](field)           # read state value\n"
    "#   - ctx['set_state'](field, value)          # write state (fire-and-forget)\n"
    "#   - await ctx['set_state_async'](field, value)\n"
    "#   Note: for best UI/graph support, add the target state fields on the node (editableStateFields=True).\n"
    "# - inputs is a dict keyed by input port names\n"
    "# Return value protocol:\n"
    "# - onMsg: return {'outputs': {...}} or any value (emits to 'out' if present)\n"
    "# - onExec: return {'exec': ['exec','exec2'], 'outputs': {...}}\n"
    "#   - 'exec' selects exec out port(s) to trigger (defaults to pass-through)\n"
    "#   - 'outputs' is a dict mapping dataOutPort -> value\n\n"
    "def onStart(ctx):\n"
    "    ctx['log']('python_script started')\n\n"
    "def onState(ctx, field, value, tsMs=None):\n"
    "    # Example: react to a state change.\n"
    "    # ctx['log'](f'state {field}={value} tsMs={tsMs}')\n"
    "    return\n\n"
    "def onMsg(ctx, inputs):\n"
    "    msg = inputs.get('msg')\n"
    "    return {'outputs': {'out': msg}}\n\n"
    "def onExec(ctx, execIn, inputs):\n"
    "    # Example: choose different exec outputs by execIn port name.\n"
    "    if execIn == 'exec2':\n"
    "        return {'exec': ['exec2'], 'outputs': {'out': inputs.get('msg')}}\n"
    "    return {'exec': ['exec'], 'outputs': {'out': inputs.get('msg')}}\n\n"
    "def onStop(ctx):\n"
    "    ctx['log']('python_script stopped')\n"
)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


class PythonScriptRuntimeNode(OperatorNode, ClosableNode):
    """
    Execute user-provided python code with lifecycle hooks:

    - onStart(ctx): invoked on construction (best-effort) and after recompiles
    - onState(ctx, field, value, tsMs=None): invoked on state updates (except 'code')
    - onMsg(ctx, inputs): invoked on exec or data arrival
    - onStop(ctx): invoked on close() (best-effort) and before recompiles
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._exec_out_ports = exec_out_ports(node, default=["exec"])
        self._state: dict[str, Any] = {}

        self._code = str(self._initial_state.get("code") or DEFAULT_CODE)
        self._runtime: dict[str, Callable[..., Any]] = {}
        self._ctx: dict[str, Any] = {}
        self._started = False
        self._closing = False
        self._last_error: str | None = None
        self._self_state_writes: dict[str, Any] = {}
        self._allow_unsafe_exec = _coerce_bool(self._initial_state.get("allowUnsafeExec"), default=False)

        self._compile_and_start()

    def __del__(self) -> None:
        # Best-effort fallback: close() is awaited by ServiceBus when nodes are unregistered,
        # but __del__ provides an additional safety net for ad-hoc use.
        try:
            if self._started and not self._closing:
                self._invoke_hook_sync("onStop")
        except Exception:
            pass

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            await self._invoke_hook_async("onStop")
        finally:
            self._started = False

    def _log(self, message: str) -> None:
        logger.info("[%s:python_script] %s", self.node_id, message)

    def _set_error(self, stage: str, exc: BaseException) -> None:
        msg = f"{stage}: {exc}"
        self._last_error = msg
        logger.error("[%s:python_script] error %s", self.node_id, msg, exc_info=exc)
        try:
            loop = asyncio.get_running_loop()
            async def _set_last_error() -> None:
                try:
                    await self.set_state("lastError", msg)
                except Exception:
                    return

            loop.create_task(_set_last_error(), name=f"python_script:lastError:{self.node_id}")
        except Exception:
            pass

    def _build_ctx(self) -> dict[str, Any]:
        async def _emit_async(port: str, value: Any) -> None:
            await self.emit(str(port), value)

        def _emit(port: str, value: Any) -> None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_emit_async(str(port), value), name=f"python_script:emit:{self.node_id}:{port}")
            except Exception:
                pass

        async def _set_state_async(field: str, value: Any) -> None:
            self._self_state_writes[str(field)] = value
            await self.set_state(str(field), value)

        def _set_state(field: str, value: Any) -> None:
            try:
                self._self_state_writes[str(field)] = value
                loop = asyncio.get_running_loop()
                loop.create_task(
                    _set_state_async(str(field), value), name=f"python_script:set_state:{self.node_id}:{field}"
                )
            except Exception:
                pass

        async def _get_state(field: str) -> Any:
            return await self.get_state_value(str(field))

        return {
            "nodeId": self.node_id,
            "state": self._state,
            "execIn": None,
            "log": self._log,
            "emit": _emit,
            "emit_async": _emit_async,
            "set_state": _set_state,
            "set_state_async": _set_state_async,
            "get_state": _get_state,
        }

    def _compile_script(self, code: str) -> dict[str, Callable[..., Any]]:
        if not self._allow_unsafe_exec:
            self._set_error(
                "compile",
                RuntimeError("unsafe python exec is disabled (set allowUnsafeExec=true to enable)"),
            )
            return {}
        env: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(code, env, env)
        except Exception as exc:
            self._set_error("compile", exc)
            return {}
        runtime: dict[str, Callable[..., Any]] = {}
        for hook in ("onStart", "onState", "onMsg", "onExec", "onStop"):
            fn = env.get(hook)
            if callable(fn):
                runtime[hook] = fn
        return runtime

    def _compile_and_start(self) -> None:
        if self._started:
            self._invoke_hook_sync("onStop")
        self._state = {}
        self._ctx = self._build_ctx()
        # Normalize line endings and tabs to avoid TabError on mixed indentation.
        code = str(self._code or "")
        code = code.replace("\r\n", "\n").replace("\r", "\n")
        code = code.expandtabs(4)
        self._runtime = self._compile_script(code)
        if not self._runtime:
            self._started = False
            return
        self._invoke_hook_sync("onStart")

    def _invoke_hook_sync(self, name: str, *args: Any) -> None:
        fn = self._runtime.get(name)
        if not callable(fn):
            if name == "onStart":
                self._started = True
            elif name == "onStop":
                self._started = False
            return
        try:
            r = fn(self._ctx, *args)
            if inspect.isawaitable(r):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(r, name=f"python_script:{name}:{self.node_id}")
                except Exception:
                    pass
        except Exception as exc:
            self._set_error(name, exc)
        finally:
            if name == "onStart":
                self._started = True
            elif name == "onStop":
                self._started = False

    async def _invoke_hook_async(self, name: str, *args: Any) -> None:
        fn = self._runtime.get(name)
        if not callable(fn):
            if name == "onStart":
                self._started = True
            elif name == "onStop":
                self._started = False
            return
        try:
            r = fn(self._ctx, *args)
            if inspect.isawaitable(r):
                await r
        except Exception as exc:
            self._set_error(name, exc)
        finally:
            if name == "onStart":
                self._started = True
            elif name == "onStop":
                self._started = False

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field)
        if name == "allowUnsafeExec":
            self._allow_unsafe_exec = _coerce_bool(value, default=False)
            self._compile_and_start()
            return
        if name == "code":
            self._code = str(value or "")
            self._compile_and_start()
            return

        # Best-effort loop prevention for state writes originating from this node (via ctx['set_state']).
        if name in self._self_state_writes and self._self_state_writes.get(name) == value:
            return

        fn = self._runtime.get("onState")
        if not callable(fn):
            return
        try:
            r = fn(self._ctx, name, value, ts_ms)
            if inspect.isawaitable(r):
                await r
        except Exception as exc:
            self._set_error("onState", exc)

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "allowUnsafeExec":
            return _coerce_bool(value, default=False)
        if name == "code":
            return str(value or "")
        return value

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        # Push-mode: treat incoming data as a message.
        if not self._runtime:
            return
        await self._run_on_msg({str(port): value}, exec_in=None)

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        # Exec-driven: pull current values for all inputs.
        if not self._runtime:
            return list(self._exec_out_ports)
        inputs: dict[str, Any] = {}
        for p in list(self.data_in_ports):
            try:
                inputs[str(p)] = await self.pull(str(p), ctx_id=exec_id)
            except Exception:
                continue
        exec_in = str(in_port or "").strip() or None
        return await self._run_on_exec(inputs, exec_in=exec_in)

    async def _run_on_exec(self, inputs: dict[str, Any], *, exec_in: str | None) -> list[str]:
        fn = self._runtime.get("onExec")
        if callable(fn):
            try:
                if self._ctx is not None:
                    self._ctx["execIn"] = exec_in
                r = fn(self._ctx, str(exec_in or ""), dict(inputs))
                if inspect.isawaitable(r):
                    r = await r
            except Exception as exc:
                self._set_error("onExec", exc)
                return list(self._exec_out_ports)
            out_ports = await self._apply_result(r)
            return out_ports if out_ports is not None else list(self._exec_out_ports)

        await self._run_on_msg(inputs, exec_in=exec_in)
        return list(self._exec_out_ports)

    async def _run_on_msg(self, inputs: dict[str, Any], *, exec_in: str | None) -> None:
        fn = self._runtime.get("onMsg")
        if not callable(fn):
            return
        try:
            if self._ctx is not None:
                self._ctx["execIn"] = exec_in
            r = fn(self._ctx, dict(inputs))
            if inspect.isawaitable(r):
                r = await r
        except Exception as exc:
            self._set_error("onMsg", exc)
            return
        await self._apply_result(r)

    async def _apply_result(self, r: Any) -> list[str] | None:
        """
        Apply a script return value:
        - dict:
          - exec routing: r.get("exec") -> str | list[str]
          - outputs: r.get("outputs") -> dict[dataOutPort,value]
          - backward compat: if "outputs" missing, treat remaining keys (excluding "exec") as outputs
        - non-dict: emit to 'out' if present
        Returns selected exec out ports if provided, else None.
        """
        if r is None:
            return None

        if isinstance(r, dict):
            exec_sel = r.get("exec") if "exec" in r else None
            outputs = r.get("outputs") if isinstance(r.get("outputs"), dict) else None

            if outputs is not None:
                # Preferred protocol: only emit from `outputs` so data ports can be named freely
                # (including "exec") without colliding with exec routing.
                for k, v in outputs.items():
                    if str(k) in self.data_out_ports:
                        try:
                            await self.emit(str(k), v)
                        except Exception:
                            continue
            else:
                # Backward compat: emit direct port keys (except 'exec').
                for k, v in r.items():
                    if str(k) in ("exec", "outputs"):
                        continue
                    if str(k) in self.data_out_ports:
                        try:
                            await self.emit(str(k), v)
                        except Exception:
                            continue

            if exec_sel is None:
                return None
            if isinstance(exec_sel, str):
                return [exec_sel]
            if isinstance(exec_sel, (list, tuple)):
                return [str(x) for x in exec_sel if str(x)]
            return None

        # Non-dict: send to default data output if present.
        if "out" in self.data_out_ports:
            try:
                await self.emit("out", r)
            except Exception:
                pass
        return None


PythonScriptRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Python Script",
    description="Execute Python code with onStart/onState/onMsg/onExec/onStop hooks.",
    tags=["script", "python", "programmable"],
    execInPorts=["exec"],
    execOutPorts=["exec"],
    editableExecInPorts=True,
    editableExecOutPorts=True,
    dataInPorts=[F8DataPortSpec(name="msg", description="Message input", valueSchema=any_schema())],
    dataOutPorts=[F8DataPortSpec(name="out", description="Script output", valueSchema=any_schema())],
    editableDataInPorts=True,
    editableDataOutPorts=True,
    stateFields=[
        F8StateSpec(
            name="allowUnsafeExec",
            label="Allow Unsafe Exec",
            description="Enable execution of user-provided Python code in this node.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="code",
            label="Code",
            description="Python source code defining onStart(ctx), onMsg(ctx, inputs), onStop(ctx).",
            uiControl="code",
            uiLanguage="python",
            valueSchema=string_schema(default=DEFAULT_CODE),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="lastError",
            label="Last Error",
            description="Last script error (compile/runtime).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.wo,
            showOnNode=False,
        ),
    ],
    editableStateFields=True,
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return PythonScriptRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(PythonScriptRuntimeNode.SPEC, overwrite=True)
    return reg
