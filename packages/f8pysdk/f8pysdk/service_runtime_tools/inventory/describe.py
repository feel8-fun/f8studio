from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import msgspec

from f8pysdk._specs.builtin_fields import normalize_describe_payload_dict
from f8pysdk.codec import dump_json, validate_as
from f8pysdk.monitoring import MonitorContractError, validate_describe_monitor_contract
from f8pysdk.specs import F8ServiceDescribe, F8ServiceEntry

from .entry import _read_yaml


logger = logging.getLogger(__name__)
_LAST_DISCOVERY_TIMING_LINES: list[str] = []
_DISCOVERY_ERROR_LOCK = threading.Lock()
_LAST_DISCOVERY_ERROR_LINES: list[str] = []
_STATIC_DESCRIBE_JSON_ERRORS = (OSError, UnicodeError, json.JSONDecodeError)
_STATIC_DESCRIBE_YAML_ERRORS = (ValueError,)
_DESCRIBE_ENTRY_READ_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_DESCRIBE_PATH_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError)
_DESCRIBE_SUBPROCESS_ERRORS = (OSError, RuntimeError, ValueError, subprocess.SubprocessError)
_DESCRIBE_VALIDATION_ERRORS = (msgspec.ValidationError, TypeError, ValueError)
_ENV_PARSE_ERRORS = (TypeError, ValueError)


def last_discovery_timing_lines() -> list[str]:
    return list(_LAST_DISCOVERY_TIMING_LINES)


def last_discovery_error_lines() -> list[str]:
    with _DISCOVERY_ERROR_LOCK:
        return list(_LAST_DISCOVERY_ERROR_LINES)


def _truncate_text(text: str, *, max_chars: int) -> str:
    raw = str(text or "")
    if len(raw) <= max_chars:
        return raw
    head = max_chars // 2
    tail = max_chars - head - 20
    return raw[:head] + "\n... <truncated> ...\n" + raw[-tail:]


def clear_discovery_errors() -> None:
    with _DISCOVERY_ERROR_LOCK:
        _LAST_DISCOVERY_ERROR_LINES.clear()


def _add_discovery_error(line: str) -> None:
    text = str(line or "").strip()
    if not text:
        return
    with _DISCOVERY_ERROR_LOCK:
        _LAST_DISCOVERY_ERROR_LINES.append(text)


def _read_static_describe_file(service_dir: Path) -> dict[str, Any] | None:
    if (os.environ.get("F8_DISCOVERY_DISABLE_STATIC_DESCRIBE") or "").strip():
        return None

    service_dir = Path(service_dir).resolve()
    json_path = service_dir / "describe.json"
    if json_path.is_file():
        try:
            raw = json_path.read_text("utf-8")
            obj = json.loads(raw) if raw.strip() else None
            return obj if isinstance(obj, dict) else None
        except _STATIC_DESCRIBE_JSON_ERRORS as exc:
            logger.debug("Failed to read static describe JSON path=%s", json_path, exc_info=exc)
            return None

    for name in ("describe.yml", "describe.yaml"):
        yaml_path = service_dir / name
        if not yaml_path.is_file():
            continue
        try:
            obj = _read_yaml(yaml_path)
            return obj if isinstance(obj, dict) else None
        except _STATIC_DESCRIBE_YAML_ERRORS as exc:
            logger.debug("Failed to read static describe YAML path=%s", yaml_path, exc_info=exc)
            return None
    return None


def read_static_describe_payload(service_dir: Path, entry: F8ServiceEntry) -> tuple[dict[str, Any] | None, str | None]:
    inline_payload = _read_inline_describe(entry)
    if inline_payload is not None:
        return inline_payload, "inline"

    static_payload = _read_static_describe_file(service_dir)
    if static_payload is not None:
        return static_payload, "file"

    return None, None


def _read_inline_describe(entry: F8ServiceEntry) -> dict[str, Any] | None:
    if (os.environ.get("F8_DISCOVERY_DISABLE_STATIC_DESCRIBE") or "").strip():
        return None
    entry_payload = dump_json(entry, mode="json")
    if not isinstance(entry_payload, dict):
        return None
    describe_obj = entry_payload.get("describe")
    return describe_obj if isinstance(describe_obj, dict) else None


def _filter_benign_stderr(text: str) -> str:
    if not text:
        return ""
    lines: list[str] = []
    for line in str(text).splitlines():
        stripped = str(line).strip()
        if not stripped:
            continue
        if stripped.startswith("Pixi task ("):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_last_json_obj(text: str) -> Any | None:
    raw = (text or "").strip()
    if not raw:
        return None
    single_json_error: json.JSONDecodeError | None = None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        single_json_error = exc

    decoder = json.JSONDecoder()
    index = 0
    last: Any | None = None
    last_decode_error: json.JSONDecodeError | None = None
    while index < len(raw):
        match = re.search(r"[\{\[]", raw[index:])
        if match is None:
            break
        start = index + match.start()
        try:
            obj, end = decoder.raw_decode(raw[start:])
            last = obj
            index = start + end
        except json.JSONDecodeError as exc:
            last_decode_error = exc
            index = start + 1
    if last is None:
        decode_error = last_decode_error or single_json_error
        if decode_error is not None:
            logger.debug("Failed to decode JSON from describe output", exc_info=decode_error)
    return last


def _is_pixi_command(command: str) -> bool:
    normalized = str(command or "").strip().lower()
    if not normalized:
        return False
    if normalized in ("pixi", "pixi.exe", "pixi.bat", "pixi.cmd"):
        return True
    return Path(normalized).name in ("pixi", "pixi.exe", "pixi.bat", "pixi.cmd")


def describe_entry(
    service_dir: Path,
    entry: F8ServiceEntry,
    *,
    initial_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    service_dir = Path(service_dir).resolve()

    if initial_payload is not None:
        initial_data: dict[str, Any] = dict(initial_payload)
    else:
        inline_payload = _read_inline_describe(entry)
        if inline_payload is not None:
            initial_data = dict(inline_payload)
        else:
            static_payload = _read_static_describe_file(service_dir)
            initial_data = static_payload if static_payload is not None else {}

    try:
        launch = entry.launch
        describe_args_value = entry.describeArgs
        describe_args = list(describe_args_value) if isinstance(describe_args_value, list) else ["--describe"]
        timeout_ms = int(entry.timeoutMs or 4000)
        if _is_pixi_command(str(launch.command)):
            timeout_ms = max(timeout_ms, 15000)
    except _DESCRIBE_ENTRY_READ_ERRORS as exc:
        logger.debug("Failed to prepare describe command for %s", service_dir, exc_info=exc)
        return None

    payload_obj: Any = initial_data if initial_data else None

    def _describe_command_text() -> str:
        return " ".join([str(launch.command), *[str(arg) for arg in (launch.args or [])], *[str(arg) for arg in describe_args]])

    if payload_obj is None:
        cmd = [str(launch.command), *[str(arg) for arg in (launch.args or [])], *[str(arg) for arg in describe_args]]

        env = os.environ.copy()
        launch_env = launch.env
        if isinstance(launch_env, dict):
            try:
                env.update({str(k): str(v) for k, v in launch_env.items()})
            except _DESCRIBE_ENTRY_READ_ERRORS as exc:
                logger.debug("Failed to apply describe launch environment for %s", service_dir, exc_info=exc)

        cwd = service_dir
        try:
            workdir_value = launch.workdir
            workdir_raw = "./" if workdir_value is None or isinstance(workdir_value, msgspec.UnsetType) else str(workdir_value)
            workdir_path = Path(workdir_raw).expanduser()
            if not workdir_path.is_absolute():
                workdir_path = (service_dir / workdir_path).resolve()
            else:
                workdir_path = workdir_path.resolve()
            cwd = workdir_path
        except _DESCRIBE_PATH_ERRORS as exc:
            logger.debug("Failed to resolve describe workdir for %s; using service dir", service_dir, exc_info=exc)
            cwd = service_dir

        started_at = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(0.1, timeout_ms / 1000.0),
                check=False,
            )
        except _DESCRIBE_SUBPROCESS_ERRORS as exc:
            message = f"describe subprocess failed for {service_dir}: {exc} (cwd={cwd}, cmd={' '.join(cmd)})"
            _add_discovery_error(message)
            logger.error(message, exc_info=exc)
            return None
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                dt_ms = (time.perf_counter() - started_at) * 1000.0
                logger.debug("describe took %.1fms: %s", dt_ms, " ".join(cmd))

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        filtered_stderr = _filter_benign_stderr(stderr)
        if filtered_stderr and not stdout:
            message = (
                f"describe stderr (no stdout) for {service_dir}: {' '.join(cmd)}\n"
                f"{_truncate_text(filtered_stderr, max_chars=800)}"
            )
            _add_discovery_error(message)
            logger.error(message)
            return None
        if filtered_stderr:
            logger.warning(
                "Error output from describe command %s:\n%s",
                " ".join(cmd),
                _truncate_text(filtered_stderr, max_chars=800),
            )

        payload_obj = _extract_last_json_obj(stdout)
        if not isinstance(payload_obj, dict):
            message = (
                f"describe produced no JSON object for {service_dir}: {' '.join(cmd)}\n"
                f"stdout:\n{_truncate_text(stdout, max_chars=600)}\n"
                f"stderr:\n{_truncate_text(filtered_stderr or stderr, max_chars=600)}"
            )
            _add_discovery_error(message)
            logger.error(message)
            return None

    data = payload_obj
    if not isinstance(data, dict):
        return None
    data = normalize_describe_payload_dict(data)
    try:
        validate_describe_monitor_contract(data)
    except MonitorContractError as exc:
        message = f"describe monitor contract invalid for {service_dir}: {exc}"
        _add_discovery_error(message)
        logger.error(message)
        return None

    try:
        payload = validate_as(F8ServiceDescribe, data)
        data = msgspec.to_builtins(payload)
    except _DESCRIBE_VALIDATION_ERRORS as exc:
        logger.debug("Describe payload validation failed for %s; using compatibility fallback", service_dir, exc_info=exc)
        if "service" not in data:
            message = f"describe JSON missing required key 'service' for {service_dir}: {_describe_command_text()}"
            _add_discovery_error(message)
            logger.error(message)
            return None
        if "operators" not in data:
            data["operators"] = []

    try:
        entry_service_class = str(entry.serviceClass or "").strip()
        described_service_class = str((data.get("service") or {}).get("serviceClass") or "").strip()
        if entry_service_class and described_service_class and entry_service_class != described_service_class:
            message = (
                f"Service class mismatch for {service_dir}: entry has '{entry_service_class}', "
                f"described has '{described_service_class}'"
            )
            _add_discovery_error(message)
            logger.error(message)
            return None
    except _DESCRIBE_ENTRY_READ_ERRORS as exc:
        logger.debug("Failed to compare service class from describe payload for %s", service_dir, exc_info=exc)

    return data


def describe_entry_timed(
    service_dir: Path,
    entry: F8ServiceEntry,
    *,
    initial_payload: dict[str, Any] | None = None,
    source: str | None = None,
) -> tuple[dict[str, Any] | None, float, str]:
    started_at = time.perf_counter()

    if initial_payload is not None and source is not None:
        payload = describe_entry(service_dir, entry, initial_payload=initial_payload)
        return payload, (time.perf_counter() - started_at) * 1000.0, source

    inline_payload = _read_inline_describe(entry)
    if inline_payload is not None:
        payload = describe_entry(service_dir, entry, initial_payload=inline_payload)
        return payload, (time.perf_counter() - started_at) * 1000.0, "inline"

    static_payload = _read_static_describe_file(service_dir)
    if static_payload is not None:
        payload = describe_entry(service_dir, entry, initial_payload=static_payload)
        return payload, (time.perf_counter() - started_at) * 1000.0, "file"

    payload = describe_entry(service_dir, entry)
    return payload, (time.perf_counter() - started_at) * 1000.0, "subprocess"


def discovery_parallelism(service_count: int) -> int:
    raw = (os.environ.get("F8_DESCRIBE_JOBS") or os.environ.get("F8_DISCOVERY_JOBS") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except _ENV_PARSE_ERRORS as exc:
            logger.debug("Invalid discovery parallelism env value=%r; using 1", raw, exc_info=exc)
            return 1

    cpu_count = os.cpu_count() or 4
    return max(1, min(service_count, min(6, cpu_count)))


def discovery_log_timings_enabled() -> bool:
    raw = (os.environ.get("F8_DISCOVERY_LOG_TIMINGS") or "").strip().lower()
    if raw in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    if raw in ("0", "false", "no", "off", "disable", "disabled", ""):
        return False
    return True


def discovery_slow_ms_default() -> float:
    raw = (os.environ.get("F8_DISCOVERY_SLOW_MS") or "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except _ENV_PARSE_ERRORS as exc:
        logger.debug("Invalid discovery slow threshold env value=%r; using 0.0", raw, exc_info=exc)
        return 0.0


def set_discovery_timing_lines(lines: list[str]) -> None:
    global _LAST_DISCOVERY_TIMING_LINES
    _LAST_DISCOVERY_TIMING_LINES = list(lines)


__all__ = [
    "clear_discovery_errors",
    "describe_entry",
    "describe_entry_timed",
    "discovery_log_timings_enabled",
    "discovery_parallelism",
    "discovery_slow_ms_default",
    "last_discovery_error_lines",
    "last_discovery_timing_lines",
    "read_static_describe_payload",
    "set_discovery_timing_lines",
]
