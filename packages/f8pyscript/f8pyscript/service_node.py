from __future__ import annotations

import asyncio
import builtins
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from f8pysdk.capabilities import ClosableNode, CommandableNode
from f8pysdk.json_unwrap import unwrap_json_value
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import ServiceNode
from f8pysdk.shm.video import VIDEO_FORMAT_BGRA32, VIDEO_FORMAT_FLOW2_F16, VideoShmHeader, VideoShmReader

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


DEFAULT_CODE = (
    "# Hooks (all optional):\n"
    "# - onStart(ctx)\n"
    "# - onStop(ctx)\n"
    "# - onPause(ctx, meta=None)\n"
    "# - onResume(ctx, meta=None)\n"
    "# - onState(ctx, field, value, tsMs=None)\n"
    "# - onData(ctx, port, value, tsMs=None)\n"
    "# - onTick(ctx, tick)\n"
    "# - onCommand(ctx, name, args, meta=None)\n"
    "#\n"
    "# Tick payload: {'seq': int, 'tsMs': int, 'deltaMs': int}\n"
    "# State read:\n"
    "# - await ctx['get_state'](field)                # freshest path\n"
    "# - ctx['get_state_cached'](field, default=None) # sync cached snapshot, may be stale\n"
    "# Permission: ctx['permission'] -> {'localExecGranted', 'expiresTsMs'}\n"
    "#\n"
    "def onStart(ctx):\n"
    "    ctx['log']('pyscript started')\n"
    "\n"
    "def onStop(ctx):\n"
    "    ctx['log']('pyscript stopped')\n"
)

_SAFE_MODULES: set[str] = {
    "asyncio",
    "collections",
    "datetime",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
    "time",
}

@dataclass
class _VideoShmSubscription:
    key: str
    shm_name: str
    decode_mode: str
    use_event: bool
    reader: VideoShmReader | None = None
    task: asyncio.Task[object] | None = None
    latest_packet: dict[str, Any] | None = None
    last_frame_id: int = 0
    last_error_sig: str | None = None
    last_error_ts_ms: int = 0
    error_count: int = 0


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
        self._ctx: dict[str, Any] = {}

        self._hook_on_start: Callable[..., Any] | None = None
        self._hook_on_stop: Callable[..., Any] | None = None
        self._hook_on_pause: Callable[..., Any] | None = None
        self._hook_on_resume: Callable[..., Any] | None = None
        self._hook_on_state: Callable[..., Any] | None = None
        self._hook_on_data: Callable[..., Any] | None = None
        self._hook_on_tick: Callable[..., Any] | None = None
        self._hook_on_command: Callable[..., Any] | None = None

        self._started = False
        self._paused = False
        self._active = True
        self._closing = False

        self._last_error: str | None = None
        self._error_dedupe: dict[str, int] = {}
        self._self_state_writes: dict[str, Any] = {}

        self._tick_enabled = bool(self._initial_state.get("tickEnabled") or False)
        self._tick_ms = self._coerce_tick_ms(self._initial_state.get("tickMs"), default=100)
        self._tick_task: asyncio.Task[object] | None = None
        self._tick_seq = 0

        self._video_subscriptions: dict[str, _VideoShmSubscription] = {}

        self._local_exec_granted = False
        self._grant_session_id = ""
        self._grant_meta: dict[str, Any] = {}
        self._grant_ts_ms = 0
        self._grant_expires_ts_ms: int | None = None
        self._exec_count = 0
        self._last_exec_ts_ms = 0

        self._declared_commands: list[dict[str, Any]] = []

        self._compile_and_start()

    def __del__(self) -> None:
        if self._started and not self._closing:
            try:
                self._invoke_sync(self._hook_on_stop, "onStop")
            except Exception as exc:
                logger.error("[%s:pyscript] __del__ onStop failed", self.node_id, exc_info=exc)
        try:
            self._shutdown_video_subscriptions_sync()
        except Exception as exc:
            logger.error("[%s:pyscript] __del__ video cleanup failed", self.node_id, exc_info=exc)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000.0)

    @staticmethod
    def _coerce_tick_ms(value: Any, *, default: int) -> int:
        try:
            out = int(value)
        except (TypeError, ValueError):
            out = int(default)
        return max(1, out)

    @staticmethod
    def _normalize_decode_mode(decode: Any) -> str:
        mode = str(decode or "auto").strip().lower()
        if mode in ("none", "auto"):
            return mode
        return "auto"

    @staticmethod
    def _header_to_dict(header: VideoShmHeader) -> dict[str, int]:
        return {
            "frameId": int(header.frame_id),
            "tsMs": int(header.ts_ms),
            "width": int(header.width),
            "height": int(header.height),
            "pitch": int(header.pitch),
            "fmt": int(header.fmt),
            "notifySeq": int(header.notify_seq),
        }

    @staticmethod
    def _compact_rows(raw: bytes, *, width: int, height: int, pitch: int, row_bytes: int) -> bytes | None:
        if width <= 0 or height <= 0 or pitch < row_bytes or row_bytes <= 0:
            return None
        if pitch == row_bytes:
            return raw
        compact = bytearray(row_bytes * height)
        for y in range(height):
            src_off = y * pitch
            dst_off = y * row_bytes
            compact[dst_off : dst_off + row_bytes] = raw[src_off : src_off + row_bytes]
        return bytes(compact)

    def _decode_video_payload(self, *, header: dict[str, int], raw: bytes, decode_mode: str) -> dict[str, Any] | None:
        if decode_mode != "auto":
            return None
        width = int(header.get("width") or 0)
        height = int(header.get("height") or 0)
        pitch = int(header.get("pitch") or 0)
        fmt = int(header.get("fmt") or 0)
        if width <= 0 or height <= 0 or pitch <= 0:
            return None

        if fmt == VIDEO_FORMAT_BGRA32:
            row_bytes = width * 4
            compact = self._compact_rows(raw, width=width, height=height, pitch=pitch, row_bytes=row_bytes)
            if compact is None:
                return {"kind": "bgra32", "shape": [height, width, 4], "data": None}
            data = None
            if np is not None:
                try:
                    arr = np.frombuffer(compact, dtype=np.uint8)
                    if int(arr.size) == (height * width * 4):
                        data = arr.reshape(height, width, 4)
                except Exception as exc:
                    logger.warning("[%s:pyscript] decode bgra failed", self.node_id, exc_info=exc)
            return {"kind": "bgra32", "shape": [height, width, 4], "data": data}

        if fmt == VIDEO_FORMAT_FLOW2_F16:
            row_bytes = width * 4
            compact = self._compact_rows(raw, width=width, height=height, pitch=pitch, row_bytes=row_bytes)
            if compact is None:
                return {"kind": "flow2_f16", "shape": [height, width, 2], "data": None}
            data = None
            if np is not None:
                try:
                    arr = np.frombuffer(compact, dtype="<f2")
                    if int(arr.size) == (height * width * 2):
                        data = arr.reshape(height, width, 2)
                except Exception as exc:
                    logger.warning("[%s:pyscript] decode flow failed", self.node_id, exc_info=exc)
            return {"kind": "flow2_f16", "shape": [height, width, 2], "data": data}

        return None

    def _copy_packet_for_script(self, packet: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(packet, dict):
            return None
        header_src = packet.get("header")
        meta_src = packet.get("meta")
        decoded_src = packet.get("decoded")
        out: dict[str, Any] = {
            "header": dict(header_src) if isinstance(header_src, dict) else {},
            "meta": dict(meta_src) if isinstance(meta_src, dict) else {},
            "raw": packet.get("raw"),
            "decoded": None,
        }
        if isinstance(decoded_src, dict):
            decoded_out: dict[str, Any] = {}
            if "kind" in decoded_src:
                decoded_out["kind"] = decoded_src.get("kind")
            if "shape" in decoded_src:
                decoded_out["shape"] = list(decoded_src.get("shape") or [])
            if "data" in decoded_src:
                decoded_out["data"] = decoded_src.get("data")
            out["decoded"] = decoded_out
        return out

    def _set_error(self, stage: str, exc: BaseException) -> None:
        msg = f"{stage}: {exc}"
        self._last_error = msg
        logger.error("[%s:pyscript] error %s", self.node_id, msg, exc_info=exc)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _set_last_error() -> None:
            try:
                await self.set_state("lastError", msg)
            except Exception as set_exc:
                logger.error("[%s:pyscript] set_state(lastError) failed", self.node_id, exc_info=set_exc)

        loop.create_task(_set_last_error(), name=f"pyscript:lastError:{self.node_id}")

    def _log_error_deduped(self, key: str, message: str, exc: BaseException) -> None:
        now_ms = self._now_ms()
        last_ts = int(self._error_dedupe.get(key) or 0)
        if (now_ms - last_ts) < 2000:
            return
        self._error_dedupe[key] = now_ms
        logger.error("[%s:pyscript] %s", self.node_id, message, exc_info=exc)

    async def _set_runtime_state(self, field: str, value: Any) -> None:
        self._self_state_writes[str(field)] = value
        await self.set_state(str(field), value)

    def _permission_view(self) -> dict[str, Any]:
        allowed = self._is_local_exec_allowed()
        return {
            "localExecGranted": bool(allowed),
            "expiresTsMs": int(self._grant_expires_ts_ms) if self._grant_expires_ts_ms is not None else None,
            "grantTsMs": int(self._grant_ts_ms or 0),
            "sessionId": str(self._grant_session_id or ""),
        }

    def _is_local_exec_allowed(self) -> bool:
        if not self._local_exec_granted:
            return False
        expiry = self._grant_expires_ts_ms
        if expiry is not None and self._now_ms() > int(expiry):
            self._local_exec_granted = False
            self._grant_expires_ts_ms = None
            return False
        return True

    async def _publish_permission_state(self) -> None:
        await self._set_runtime_state("localExecGranted", bool(self._is_local_exec_allowed()))
        await self._set_runtime_state("localExecGrantTsMs", int(self._grant_ts_ms))
        await self._set_runtime_state("execCount", int(self._exec_count))
        await self._set_runtime_state("lastExecTsMs", int(self._last_exec_ts_ms))

    def _build_script_builtins(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "abs": builtins.abs,
            "all": builtins.all,
            "any": builtins.any,
            "bool": builtins.bool,
            "bytes": builtins.bytes,
            "callable": builtins.callable,
            "dict": builtins.dict,
            "enumerate": builtins.enumerate,
            "float": builtins.float,
            "int": builtins.int,
            "isinstance": builtins.isinstance,
            "len": builtins.len,
            "list": builtins.list,
            "max": builtins.max,
            "min": builtins.min,
            "pow": builtins.pow,
            "print": builtins.print,
            "range": builtins.range,
            "round": builtins.round,
            "set": builtins.set,
            "slice": builtins.slice,
            "sorted": builtins.sorted,
            "str": builtins.str,
            "sum": builtins.sum,
            "tuple": builtins.tuple,
            "zip": builtins.zip,
            "Exception": builtins.Exception,
            "ValueError": builtins.ValueError,
            "TypeError": builtins.TypeError,
            "RuntimeError": builtins.RuntimeError,
            "KeyError": builtins.KeyError,
            "IndexError": builtins.IndexError,
            "PermissionError": builtins.PermissionError,
        }

        def _guarded_import(name: str, globals_obj: Any = None, locals_obj: Any = None, fromlist: Any = (), level: int = 0) -> Any:
            module_name = str(name or "").strip()
            if not module_name:
                raise ImportError("empty module name")
            root_name = module_name.split(".")[0]
            if not self._is_local_exec_allowed() and root_name not in _SAFE_MODULES:
                raise PermissionError(f"import blocked without local exec grant: {module_name}")
            return builtins.__import__(module_name, globals_obj, locals_obj, fromlist, int(level))

        out["__import__"] = _guarded_import
        return out

    async def _exec_local(
        self,
        command: str,
        args: list[str] | tuple[str, ...] | None = None,
        *,
        timeoutMs: int | None = None,
        cwd: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._is_local_exec_allowed():
            raise PermissionError("local execution is not granted")

        cmd = str(command or "").strip()
        if not cmd:
            raise ValueError("exec_local command is empty")

        argv = [cmd]
        if args is not None:
            for item in list(args):
                argv.append(str(item))

        run_cwd = str(cwd).strip() if cwd is not None else None
        proc_env: dict[str, str] | None = None
        if env is not None:
            proc_env = dict(os.environ)
            for key, value in dict(env).items():
                proc_env[str(key)] = str(value)

        logger.info("[%s:pyscript] exec_local command=%s args=%s", self.node_id, cmd, argv[1:])
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=run_cwd,
            env=proc_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        timeout_s: float | None
        if timeoutMs is None:
            timeout_s = None
        else:
            timeout_s = max(0.001, float(timeoutMs) / 1000.0)

        try:
            if timeout_s is None:
                stdout_raw, stderr_raw = await proc.communicate()
            else:
                stdout_raw, stderr_raw = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            self._last_exec_ts_ms = self._now_ms()
            await self._set_runtime_state("lastExecTsMs", int(self._last_exec_ts_ms))
            raise TimeoutError(f"exec_local timeout command={cmd}") from exc

        stdout_text = (stdout_raw or b"").decode("utf-8", errors="replace")
        stderr_text = (stderr_raw or b"").decode("utf-8", errors="replace")

        self._exec_count += 1
        self._last_exec_ts_ms = self._now_ms()
        await self._set_runtime_state("execCount", int(self._exec_count))
        await self._set_runtime_state("lastExecTsMs", int(self._last_exec_ts_ms))
        return {
            "ok": bool(proc.returncode == 0),
            "returncode": int(proc.returncode or 0),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "command": cmd,
            "args": argv[1:],
        }

    def _build_ctx(self) -> dict[str, Any]:
        async def _emit_async(port: str, value: Any) -> None:
            await self.emit(str(port), value)

        def _emit(port: str, value: Any) -> None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError as exc:
                logger.error("[%s:pyscript] emit without running loop", self.node_id, exc_info=exc)
                return
            loop.create_task(_emit_async(str(port), value), name=f"pyscript:emit:{self.node_id}:{port}")

        async def _set_state_async(field: str, value: Any) -> None:
            await self._set_runtime_state(str(field), value)

        def _set_state(field: str, value: Any) -> None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError as exc:
                logger.error("[%s:pyscript] set_state without running loop", self.node_id, exc_info=exc)
                return
            loop.create_task(
                _set_state_async(str(field), value),
                name=f"pyscript:set_state:{self.node_id}:{field}",
            )

        async def _get_state(field: str) -> Any:
            return await self.get_state_value(str(field))

        def _get_state_cached(field: str, default: Any = None) -> Any:
            return self.get_state_cached(str(field), default)

        def _subscribe_video_shm(key: str, shm_name: str, *, decode: str = "auto", use_event: bool = False) -> None:
            key_name = str(key or "").strip()
            shm = str(shm_name or "").strip()
            if not key_name or not shm:
                return
            self._unsubscribe_video_shm_sync(key_name)
            sub = _VideoShmSubscription(
                key=key_name,
                shm_name=shm,
                decode_mode=self._normalize_decode_mode(decode),
                use_event=bool(use_event),
            )
            self._video_subscriptions[key_name] = sub
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError as exc:
                logger.error("[%s:pyscript] subscribe_video_shm without loop", self.node_id, exc_info=exc)
                return
            sub.task = loop.create_task(
                self._run_video_shm_subscription(key_name),
                name=f"pyscript:video_sub:{self.node_id}:{key_name}",
            )

        def _get_video_shm(key: str) -> dict[str, Any] | None:
            key_name = str(key or "").strip()
            if not key_name:
                return None
            sub = self._video_subscriptions.get(key_name)
            if sub is None:
                return None
            return self._copy_packet_for_script(sub.latest_packet)

        def _unsubscribe_video_shm(key: str) -> None:
            self._unsubscribe_video_shm_sync(str(key or "").strip())

        def _list_video_shm_subscriptions() -> list[dict[str, Any]]:
            items: list[dict[str, Any]] = []
            for key_name in sorted(self._video_subscriptions.keys()):
                sub = self._video_subscriptions.get(key_name)
                if sub is None:
                    continue
                items.append(
                    {
                        "key": sub.key,
                        "shmName": sub.shm_name,
                        "decodeMode": sub.decode_mode,
                        "hasPacket": sub.latest_packet is not None,
                        "lastFrameId": int(sub.last_frame_id),
                        "errorCount": int(sub.error_count),
                    }
                )
            return items

        return {
            "serviceId": self.node_id,
            "locals": self._locals,
            "log": lambda msg: logger.info("[%s:pyscript] %s", self.node_id, str(msg)),
            "emit": _emit,
            "emit_async": _emit_async,
            "set_state": _set_state,
            "set_state_async": _set_state_async,
            "get_state": _get_state,
            "get_state_cached": _get_state_cached,
            "subscribe_video_shm": _subscribe_video_shm,
            "get_video_shm": _get_video_shm,
            "unsubscribe_video_shm": _unsubscribe_video_shm,
            "list_video_shm_subscriptions": _list_video_shm_subscriptions,
            "exec_local": self._exec_local,
        }

    def _build_invoke_ctx(self) -> dict[str, Any]:
        invoke_ctx = dict(self._ctx)
        invoke_ctx["permission"] = self._permission_view()
        return invoke_ctx

    def _compile_script(self, code: str) -> None:
        env: dict[str, Any] = {"__builtins__": self._build_script_builtins()}
        exec(code, env, env)

        on_start = env.get("onStart")
        on_stop = env.get("onStop")
        on_pause = env.get("onPause")
        on_resume = env.get("onResume")
        on_state = env.get("onState")
        on_data = env.get("onData")
        on_tick = env.get("onTick")
        on_command = env.get("onCommand")

        self._hook_on_start = on_start if callable(on_start) else None
        self._hook_on_stop = on_stop if callable(on_stop) else None
        self._hook_on_pause = on_pause if callable(on_pause) else None
        self._hook_on_resume = on_resume if callable(on_resume) else None
        self._hook_on_state = on_state if callable(on_state) else None
        self._hook_on_data = on_data if callable(on_data) else None
        self._hook_on_tick = on_tick if callable(on_tick) else None
        self._hook_on_command = on_command if callable(on_command) else None

    def _compile_and_start(self) -> None:
        if self._started:
            self._invoke_sync(self._hook_on_stop, "onStop")
        self._shutdown_video_subscriptions_sync()

        self._locals = {}
        self._ctx = self._build_ctx()

        code = str(self._code or "").replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
        self._hook_on_start = None
        self._hook_on_stop = None
        self._hook_on_pause = None
        self._hook_on_resume = None
        self._hook_on_state = None
        self._hook_on_data = None
        self._hook_on_tick = None
        self._hook_on_command = None

        try:
            self._compile_script(code)
        except Exception as exc:
            self._started = False
            self._set_error("compile", exc)
            return

        self._invoke_sync(self._hook_on_start, "onStart")
        self._started = True
        self._paused = False
        self._ensure_tick_task()

    def _invoke_sync(self, hook: Callable[..., Any] | None, stage: str, *args: Any) -> Any:
        if hook is None:
            return None
        try:
            invoke_ctx = self._build_invoke_ctx()
            result = hook(invoke_ctx, *args)
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError as exc:
                    self._set_error(stage, exc)
                    return None
                loop.create_task(result, name=f"pyscript:{stage}:{self.node_id}")
                return None
            return result
        except Exception as exc:
            self._set_error(stage, exc)
            return None

    async def _invoke_async(self, hook: Callable[..., Any] | None, stage: str, *args: Any) -> Any:
        if hook is None:
            return None
        try:
            invoke_ctx = self._build_invoke_ctx()
            result = hook(invoke_ctx, *args)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as exc:
            self._set_error(stage, exc)
            raise

    def _extract_outputs(self, result: Any) -> dict[str, Any]:
        if result is None:
            return {}
        if isinstance(result, dict):
            raw_outputs = result.get("outputs")
            if isinstance(raw_outputs, dict):
                return {str(k): v for k, v in raw_outputs.items()}
            return {
                str(k): v
                for k, v in result.items()
                if str(k) not in ("ok", "result", "exec", "error", "outputs")
            }
        return {"out": result}

    async def _emit_outputs(self, result: Any) -> None:
        outputs = self._extract_outputs(result)
        for out_port, out_value in outputs.items():
            try:
                await self.emit(str(out_port), out_value)
            except Exception as exc:
                self._log_error_deduped("emit_output", f"emit failed port={out_port}", exc)

    def _normalize_commands_state(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []

        payload = value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            payload = json.loads(text)

        if payload is None:
            return []

        if isinstance(payload, dict):
            if not payload:
                return []
            commands_any = payload.get("commands")
            if isinstance(commands_any, list):
                payload = commands_any
            elif "name" in payload:
                payload = [payload]
            else:
                raise ValueError("commands must be a list")

        if not isinstance(payload, list):
            raise ValueError("commands must be a list")

        out: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"commands[{index}] must be an object")
            name = str(item.get("name") or "").strip()
            if not name:
                raise ValueError(f"commands[{index}].name is required")
            out.append(dict(item))
        return out

    def _unsubscribe_video_shm_sync(self, key: str) -> bool:
        sub = self._video_subscriptions.pop(str(key), None)
        if sub is None:
            return False
        task = sub.task
        sub.task = None
        if task is not None and not task.done():
            task.cancel()
            return True
        reader = sub.reader
        sub.reader = None
        if reader is not None:
            try:
                reader.close()
            except Exception as exc:
                self._log_error_deduped("video_reader_close", "video reader close failed", exc)
        return True

    def _shutdown_video_subscriptions_sync(self) -> None:
        keys = list(self._video_subscriptions.keys())
        for key in keys:
            self._unsubscribe_video_shm_sync(key)

    async def _shutdown_video_subscriptions_async(self) -> None:
        keys = list(self._video_subscriptions.keys())
        tasks: list[asyncio.Task[object]] = []
        for key in keys:
            sub = self._video_subscriptions.pop(key, None)
            if sub is None:
                continue
            task = sub.task
            sub.task = None
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
            reader = sub.reader
            sub.reader = None
            if reader is not None:
                try:
                    reader.close()
                except Exception as exc:
                    self._log_error_deduped("video_reader_close", "video reader close failed", exc)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_video_shm_subscription(self, key: str) -> None:
        sub_ref: _VideoShmSubscription | None = None
        try:
            while True:
                sub = self._video_subscriptions.get(key)
                if sub is None:
                    return
                sub_ref = sub

                if sub.reader is None:
                    try:
                        reader = VideoShmReader(sub.shm_name)
                        reader.open(use_event=bool(sub.use_event))
                        sub.reader = reader
                    except Exception as exc:
                        sub.error_count += 1
                        self._log_error_deduped(
                            f"video_open:{sub.key}",
                            f"video shm open failed key={sub.key} shm={sub.shm_name}",
                            exc,
                        )
                        await asyncio.sleep(0.2)
                        continue

                assert sub.reader is not None
                try:
                    has_new = bool(sub.reader.wait_new_frame(timeout_ms=20))
                    if not has_new:
                        await asyncio.sleep(0)
                        continue

                    header, payload = sub.reader.read_latest_frame()
                    if header is None or payload is None:
                        await asyncio.sleep(0)
                        continue

                    frame_id = int(header.frame_id)
                    if frame_id <= 0:
                        await asyncio.sleep(0)
                        continue
                    if frame_id == int(sub.last_frame_id) and sub.latest_packet is not None:
                        await asyncio.sleep(0)
                        continue

                    frame_bytes = int(header.frame_bytes)
                    if frame_bytes <= 0 or frame_bytes > int(header.payload_capacity):
                        await asyncio.sleep(0)
                        continue
                    if frame_bytes > len(payload):
                        await asyncio.sleep(0)
                        continue

                    raw = bytes(payload[:frame_bytes])
                    header_dict = self._header_to_dict(header)
                    decoded = self._decode_video_payload(header=header_dict, raw=raw, decode_mode=sub.decode_mode)
                    sub.latest_packet = {
                        "header": header_dict,
                        "raw": raw,
                        "decoded": decoded,
                        "meta": {
                            "key": sub.key,
                            "shmName": sub.shm_name,
                            "decodeMode": sub.decode_mode,
                            "lastUpdateMs": self._now_ms(),
                        },
                    }
                    sub.last_frame_id = frame_id
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    sub.error_count += 1
                    self._log_error_deduped(
                        f"video_read:{sub.key}",
                        f"video shm read failed key={sub.key} shm={sub.shm_name}",
                        exc,
                    )
                    reader = sub.reader
                    sub.reader = None
                    if reader is not None:
                        try:
                            reader.close()
                        except Exception as close_exc:
                            self._log_error_deduped("video_reader_close", "video reader close failed", close_exc)
                    await asyncio.sleep(0.2)
        finally:
            if sub_ref is not None and sub_ref.reader is not None:
                reader = sub_ref.reader
                sub_ref.reader = None
                try:
                    reader.close()
                except Exception as exc:
                    self._log_error_deduped("video_reader_close", "video reader close failed", exc)

    def _ensure_tick_task(self) -> None:
        if self._tick_task is not None and not self._tick_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._tick_task = loop.create_task(self._tick_loop(), name=f"pyscript:tick:{self.node_id}")

    async def _tick_loop(self) -> None:
        last_tick_ts = self._now_ms()
        next_deadline = time.monotonic()
        while not self._closing:
            try:
                if not self._started or self._paused or not self._tick_enabled:
                    next_deadline = time.monotonic()
                    await asyncio.sleep(0.05)
                    continue

                now_mono = time.monotonic()
                wait_s = next_deadline - now_mono
                if wait_s > 0:
                    await asyncio.sleep(wait_s)

                current_ts_ms = self._now_ms()
                delta_ms = max(0, current_ts_ms - last_tick_ts)
                last_tick_ts = current_ts_ms
                self._tick_seq += 1
                tick_payload = {
                    "seq": int(self._tick_seq),
                    "tsMs": int(current_ts_ms),
                    "deltaMs": int(delta_ms),
                }

                if self._hook_on_tick is not None:
                    result = await self._invoke_async(self._hook_on_tick, "onTick", tick_payload)
                    await self._emit_outputs(result)

                interval_s = max(0.001, float(self._tick_ms) / 1000.0)
                now_after = time.monotonic()
                if next_deadline <= now_after:
                    next_deadline = now_after + interval_s
                else:
                    next_deadline += interval_s
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_error_deduped("tick_loop", "tick loop failed", exc)
                await asyncio.sleep(0.05)

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            await self._invoke_async(self._hook_on_stop, "onStop")
            self._started = False
            self._paused = False
            tick_task = self._tick_task
            self._tick_task = None
            if tick_task is not None and not tick_task.done():
                tick_task.cancel()
                await asyncio.gather(tick_task, return_exceptions=True)
        finally:
            await self._shutdown_video_subscriptions_async()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        self._active = bool(active)
        if not self._started:
            return

        if self._active:
            if self._paused:
                self._paused = False
                result = await self._invoke_async(self._hook_on_resume, "onResume", dict(meta or {}))
                await self._emit_outputs(result)
            return

        if not self._paused:
            self._paused = True
            result = await self._invoke_async(self._hook_on_pause, "onPause", dict(meta or {}))
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
            return self._coerce_tick_ms(value_unwrapped, default=self._tick_ms)
        if name == "commands":
            return self._normalize_commands_state(value_unwrapped)
        return value_unwrapped

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "").strip()
        value_unwrapped = unwrap_json_value(value)

        if name == "code":
            self._code = str(value_unwrapped or "")
            self._compile_and_start()
            return
        if name == "tickEnabled":
            self._tick_enabled = bool(value_unwrapped)
            self._ensure_tick_task()
            return
        if name == "tickMs":
            self._tick_ms = self._coerce_tick_ms(value_unwrapped, default=self._tick_ms)
            return
        if name == "commands":
            try:
                self._declared_commands = self._normalize_commands_state(value_unwrapped)
            except Exception as exc:
                self._set_error("commands", exc)
            return

        if name in self._self_state_writes and self._self_state_writes.get(name) == value_unwrapped:
            return

        if self._hook_on_state is None:
            return
        result = await self._invoke_async(self._hook_on_state, "onState", name, value_unwrapped, ts_ms)
        await self._emit_outputs(result)

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if self._hook_on_data is None:
            return
        result = await self._invoke_async(self._hook_on_data, "onData", str(port), value, ts_ms)
        await self._emit_outputs(result)

    async def on_command(self, name: str, args: dict[str, Any] | None = None, *, meta: dict[str, Any] | None = None) -> Any:
        call = str(name or "").strip()
        call_args = dict(args or {})
        call_meta = dict(meta or {})
        if not call:
            raise ValueError("empty command name")

        if call == "grant_local_exec":
            ttl_ms_raw = call_args.get("ttlMs")
            ttl_ms: int | None
            if ttl_ms_raw is None:
                ttl_ms = None
            else:
                try:
                    ttl_ms = max(1, int(ttl_ms_raw))
                except (TypeError, ValueError) as exc:
                    raise ValueError("ttlMs must be an integer") from exc

            self._local_exec_granted = True
            self._grant_ts_ms = self._now_ms()
            self._grant_session_id = str(call_meta.get("reqId") or call_meta.get("sessionId") or self._grant_ts_ms)
            self._grant_meta = dict(call_meta)
            self._grant_expires_ts_ms = (self._grant_ts_ms + ttl_ms) if ttl_ms is not None else None
            await self._publish_permission_state()
            return {"ok": True, "result": self._permission_view()}

        if call == "revoke_local_exec":
            self._local_exec_granted = False
            self._grant_expires_ts_ms = None
            await self._publish_permission_state()
            return {"ok": True, "result": self._permission_view()}

        if call == "restart_script":
            self._compile_and_start()
            return {"ok": True, "result": {"restarted": True}}

        if call == "status":
            return {
                "ok": True,
                "result": {
                    "started": bool(self._started),
                    "paused": bool(self._paused),
                    "active": bool(self._active),
                    "tickEnabled": bool(self._tick_enabled),
                    "tickMs": int(self._tick_ms),
                    "tickSeq": int(self._tick_seq),
                    "lastError": str(self._last_error or ""),
                    "permission": self._permission_view(),
                    "videoSubscriptions": [
                        {
                            "key": item["key"],
                            "shmName": item["shmName"],
                            "hasPacket": item["hasPacket"],
                            "errorCount": item["errorCount"],
                        }
                        for item in self._ctx["list_video_shm_subscriptions"]()
                    ],
                    "declaredCommands": list(self._declared_commands),
                },
            }

        if self._hook_on_command is None:
            raise ValueError(f"unknown command: {call}")

        result = await self._invoke_async(self._hook_on_command, "onCommand", call, call_args, call_meta)
        return {"ok": True, "result": result}
