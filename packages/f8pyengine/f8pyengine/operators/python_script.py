from __future__ import annotations

import asyncio
import inspect
import traceback
from typing import Any, Callable

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    string_schema,
)
from f8pysdk.capabilities import ClosableNode
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.python_script"


DEFAULT_CODE = (
    "# Define hooks: onStart(ctx), onMsg(ctx, inputs), onExec(ctx, execIn, inputs), onStop(ctx)\n"
    "# - ctx['state'] is preserved between calls\n"
    "# - ctx['execIn'] is set for exec-triggered calls\n"
    "# - inputs is a dict keyed by input port names\n"
    "# Return value protocol:\n"
    "# - onMsg: return {'outputs': {...}} or any value (emits to 'out' if present)\n"
    "# - onExec: return {'exec': ['exec','exec2'], 'outputs': {...}}\n"
    "#   - 'exec' selects exec out port(s) to trigger (defaults to pass-through)\n"
    "#   - 'outputs' is a dict mapping dataOutPort -> value\n\n"
    "def onStart(ctx):\n"
    "    ctx['log']('python_script started')\n\n"
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


class PythonScriptRuntimeNode(RuntimeNode, ClosableNode):
    """
    Execute user-provided python code with lifecycle hooks:

    - onStart(ctx): invoked on construction (best-effort) and after recompiles
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
        self._exec_out_ports = list(getattr(node, "execOutPorts", None) or []) or ["exec"]
        self._state: dict[str, Any] = {}

        self._code = str(self._initial_state.get("code") or DEFAULT_CODE)
        self._runtime: dict[str, Callable[..., Any]] = {}
        self._ctx: dict[str, Any] = {}
        self._started = False
        self._closing = False
        self._last_error: str | None = None

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
        print(f"[{self.node_id}:python_script] {message}")

    def _set_error(self, stage: str, exc: BaseException) -> None:
        msg = f"{stage}: {exc}"
        self._last_error = msg
        self._log(f"error {msg}")
        try:
            traceback.print_exc()
        except Exception:
            pass
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
            await self.set_state(str(field), value)

        def _set_state(field: str, value: Any) -> None:
            try:
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
        env: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(code, env, env)
        except Exception as exc:
            self._set_error("compile", exc)
            return {}
        runtime: dict[str, Callable[..., Any]] = {}
        for hook in ("onStart", "onMsg", "onExec", "onStop"):
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
        if str(field) != "code":
            return
        self._code = str(value or "")
        self._compile_and_start()

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
    description="Execute Python code with onStart/onMsg/onExec/onStop hooks.",
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
            name="code",
            label="Code",
            description="Python source code defining onStart(ctx), onMsg(ctx, inputs), onStop(ctx).",
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

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PythonScriptRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(PythonScriptRuntimeNode.SPEC, overwrite=True)
    return reg
