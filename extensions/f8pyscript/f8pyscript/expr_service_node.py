from __future__ import annotations

import keyword
import logging
import time
from types import CodeType
from typing import Any

from f8pysdk.capabilities import ClosableNode
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode

from .expr_error_reporter import PyExprErrorReporter
from .expr_evaluator import PyExprEvaluator
from .expr_json_ref import wrap_pyexpr_value
from .expr_templates import DEFAULT_CODE
from .expr_validator import compile_pyexpr

logger = logging.getLogger(__name__)


def _is_identifier(name: str) -> bool:
    text = str(name or "")
    return bool(text) and text.isidentifier() and not keyword.iskeyword(text)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{field_name} expects a boolean")


def _normalize_code(value: Any) -> str:
    text = str("" if value is None else value)
    if "\n" not in text and "\r" not in text:
        return text.strip()
    parts = [part.strip() for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return " ".join([part for part in parts if part]).strip()


class PythonExprServiceNode(ServiceNode, ClosableNode):
    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None = None) -> None:
        data_in_ports = [str(p.name) for p in list(node.dataInPorts or [])] or ["in"]
        data_out_ports = [str(p.name) for p in list(node.dataOutPorts or [])] or ["out"]
        state_fields = [str(s.name) for s in list(node.stateFields or [])] or [
            "code",
            "allowNumpy",
            "unpackDictOutputs",
            "active",
        ]
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=data_in_ports,
            data_out_ports=data_out_ports,
            state_fields=state_fields,
        )
        self._initial_state = dict(initial_state or {})
        self._code = _normalize_code(self._initial_state.get("code") or DEFAULT_CODE)
        self._allow_numpy = _coerce_bool(self._initial_state.get("allowNumpy"), field_name="allowNumpy")
        self._unpack_dict_outputs = _coerce_bool(
            self._initial_state.get("unpackDictOutputs"), field_name="unpackDictOutputs"
        )
        self._compiled: CodeType | None = None
        self._compile_error: str | None = None
        self._evaluator = PyExprEvaluator()
        self._error_reporter = PyExprErrorReporter(
            report_error=self.report_error,
            clear_error=self.clear_error,
        )
        self._latest_inputs: dict[str, Any] = {}
        self._active = True
        self._recompile()

    async def close(self) -> None:
        return

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "code":
            return _normalize_code(value)
        if name == "allowNumpy":
            return _coerce_bool(value, field_name="allowNumpy")
        if name == "unpackDictOutputs":
            return _coerce_bool(value, field_name="unpackDictOutputs")
        if name == "active":
            return _coerce_bool(value, field_name="active")
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "code":
            self._code = _normalize_code(value)
            self._recompile()
            return
        if name == "allowNumpy":
            self._allow_numpy = _coerce_bool(value, field_name="allowNumpy")
            self._recompile()
            return
        if name == "unpackDictOutputs":
            self._unpack_dict_outputs = _coerce_bool(value, field_name="unpackDictOutputs")
            return
        if name == "active":
            self._active = _coerce_bool(value, field_name="active")

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if not self._active:
            return
        in_port = str(port or "").strip()
        if not in_port:
            return
        self._latest_inputs[in_port] = value
        result = await self._eval_latest()
        if result is None and self._compiled is None:
            return
        await self._emit_result(result)

    def _recompile(self) -> None:
        compiled, error = compile_pyexpr(self._code, allow_numpy=self._allow_numpy)
        self._compiled = compiled
        self._compile_error = error

    def _build_eval_names(self) -> dict[str, Any]:
        wrapped_inputs = {key: wrap_pyexpr_value(item) for key, item in self._latest_inputs.items()}
        env: dict[str, Any] = {"inputs": wrapped_inputs}
        for key, item in wrapped_inputs.items():
            if _is_identifier(key):
                env[key] = item
        return env

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000.0)

    async def _eval_latest(self) -> Any:
        if self._compiled is None:
            await self._error_reporter.set_error(self._compile_error or "invalid expression")
            return None
        eval_result = self._evaluator.evaluate(
            self._compiled,
            names=self._build_eval_names(),
            allow_numpy=self._allow_numpy,
        )
        if eval_result.error is not None:
            exc = eval_result.error
            now_ms = self._now_ms()
            sig = f"{type(exc).__name__}:{exc}"
            if self._error_reporter.should_log_eval_error(sig, now_ms=now_ms):
                logger.warning("[%s:pyexpr] eval failed: %s", self.node_id, exc,
                               exc_info=(type(exc), exc, exc.__traceback__))
            await self._error_reporter.set_error(f"eval: {exc}")
            return None
        await self._error_reporter.clear_error()
        return eval_result.value

    def _default_output_port(self) -> str | None:
        if "out" in self.data_out_ports:
            return "out"
        if self.data_out_ports:
            return str(self.data_out_ports[0])
        return None

    async def _emit_result(self, result: Any) -> None:
        if isinstance(result, dict) and self._unpack_dict_outputs:
            matched = False
            for raw_key, raw_value in result.items():
                out_port = str(raw_key)
                if out_port not in self.data_out_ports:
                    now_ms = self._now_ms()
                    sig = f"unmatched:{out_port}"
                    if self._error_reporter.should_log_unmatched_output(sig, now_ms=now_ms):
                        logger.warning("[%s:pyexpr] unpack output key has no port: %s", self.node_id, out_port)
                    continue
                matched = True
                await self.emit(out_port, raw_value)
            if matched:
                return
        default_out = self._default_output_port()
        if default_out is None:
            return
        await self.emit(default_out, result)
