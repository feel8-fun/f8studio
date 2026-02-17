from __future__ import annotations
from typing import Any

import concurrent.futures
import json
import logging
import os
import subprocess
import re
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
import yaml

from f8pysdk import F8ServiceDescribe, F8ServiceEntry
from f8pysdk.builtin_state_fields import normalize_describe_payload_dict

from .service_catalog import ServiceCatalog


logger = logging.getLogger(__name__)
_LAST_DISCOVERY_TIMING_LINES: list[str] = []
_DISCOVERY_ERROR_LOCK = threading.Lock()
_LAST_DISCOVERY_ERROR_LINES: list[str] = []


def last_discovery_timing_lines() -> list[str]:
    """
    Returns the last discovery timing lines (best-effort), suitable for UI display.

    This is mainly for the Studio GUI where stdout/stderr may not be visible.
    """
    return list(_LAST_DISCOVERY_TIMING_LINES)


def last_discovery_error_lines() -> list[str]:
    """
    Returns discovery errors from the last run (best-effort), suitable for UI display.
    """
    with _DISCOVERY_ERROR_LOCK:
        return list(_LAST_DISCOVERY_ERROR_LINES)


def _truncate_text(text: str, *, max_chars: int) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max_chars - head - 20
    return s[:head] + "\n... <truncated> ...\n" + s[-tail:]


def _discovery_clear_errors() -> None:
    with _DISCOVERY_ERROR_LOCK:
        _LAST_DISCOVERY_ERROR_LINES.clear()


def _discovery_add_error(line: str) -> None:
    s = str(line or "").strip()
    if not s:
        return
    with _DISCOVERY_ERROR_LOCK:
        _LAST_DISCOVERY_ERROR_LINES.append(s)


def _default_roots() -> list[Path]:
    """
    Default discovery roots.

    - `F8_SERVICE_DISCOVERY_DIRS` (os.pathsep-separated) overrides.
    - otherwise, use `<repo>/services` if it exists (when running from source).
    """
    env = (os.environ.get("F8_SERVICE_DISCOVERY_DIRS") or "").strip()
    if env:
        return [Path(p).expanduser().resolve() for p in env.split(os.pathsep) if p.strip()]

    try:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "services"
            if candidate.is_dir():
                return [candidate.resolve()]
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    return []


def default_discovery_roots() -> list[Path]:
    return _default_roots()


def _read_yaml(path: Path) -> Any:
    try:
        raw = path.read_text("utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to read {path}: {exc}") from exc
    try:
        return yaml.safe_load(raw) if raw.strip() else None
    except Exception as exc:
        raise ValueError(f"Failed to parse YAML {path}: {exc}") from exc


def _platform_service_yml_names() -> list[str]:
    """
    Platform-specific service entry filenames (in priority order).

    Supported:
    - service.win.yml
    - service.linux.yml
    - service.mac.yml (darwin)

    Always falls back to `service.yml` if no platform-specific file exists.
    """
    if os.name == "nt" or sys.platform.startswith("win"):
        return ["service.win.yml"]
    if sys.platform.startswith("darwin"):
        return ["service.mac.yml"]
    return ["service.linux.yml"]


def _service_yml_candidates(service_dir: Path) -> list[Path]:
    service_dir = Path(service_dir).resolve()
    names = _platform_service_yml_names()
    return [service_dir / n for n in names] + [service_dir / "service.yml"]


def find_service_dirs(roots: Iterable[Path]) -> list[Path]:
    """
    Find directories containing a service entry file.

    Discovery structure:
      <root>/<serviceClass>/service.yml
      <root>/<serviceClass>/service.win.yml   (optional)
      <root>/<serviceClass>/service.linux.yml (optional)
      <root>/<serviceClass>/service.mac.yml   (optional)
      <root>/<serviceClass>/operators.yml   (optional)

    Note:
    - `serviceClass` is slash-separated (eg. `f8/engine`), so services may live
      in nested directories like `<root>/f8/engine/service.yml`.
    """
    found: set[Path] = set()
    for root in roots:
        r = Path(root).expanduser()
        if not r.is_absolute():
            r = (Path.cwd() / r).resolve()
        else:
            r = r.resolve()

        if not r.exists() or not r.is_dir():
            continue
        try:
            patterns = ["service.yml", "service.win.yml", "service.linux.yml", "service.mac.yml"]
            for pat in patterns:
                for svc_file in r.rglob(pat):
                    if svc_file.is_file():
                        found.add(svc_file.parent.resolve())
        except Exception:
            # Fallback to the immediate children layout.
            for child in sorted(r.iterdir()):
                if not child.is_dir():
                    continue
                if any((child / name).is_file() for name in ("service.yml", "service.win.yml", "service.linux.yml", "service.mac.yml")):
                    found.add(child.resolve())
    return sorted(found)


def _absolutize_entry_paths(entry: F8ServiceEntry, *, service_dir: Path) -> F8ServiceEntry:
    service_dir = Path(service_dir).resolve()
    payload = entry.model_dump(mode="json")

    launch = dict(payload.get("launch") or {})
    wd = str(launch.get("workdir") or "./")
    wd_path = Path(wd).expanduser()
    if not wd_path.is_absolute():
        wd_path = (service_dir / wd_path).resolve()
    else:
        wd_path = wd_path.resolve()
    launch["workdir"] = str(wd_path)

    # Absolutize command when it looks like a path. On Windows, CreateProcess can fail
    # to resolve relative executable paths that include directory components (e.g. `win/foo.exe`),
    # so we normalize them here.
    cmd_raw = str(launch.get("command") or "").strip()
    try:
        cmd_path = Path(cmd_raw).expanduser()
        looks_like_path = bool(cmd_raw) and (
            "/" in cmd_raw or "\\" in cmd_raw or cmd_raw.startswith(".") or bool(cmd_path.suffix)
        )
        if looks_like_path and not cmd_path.is_absolute():
            launch["command"] = str((wd_path / cmd_path).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        pass

    payload["launch"] = launch
    return F8ServiceEntry.model_validate(payload)


def load_service_entry(service_dir: Path) -> F8ServiceEntry:
    """
    Load a minimal discovery entry from a service entry YAML file.

    Supported forms:
    - `schemaVersion: f8serviceEntry/1` + `launch: {...}`
    - shorthand: `command: ...` + optional `args/env/workdir` (will be mapped into `launch`)
    """
    service_dir = Path(service_dir).resolve()

    def _try_load_candidate(candidate: Path) -> dict[str, Any] | None:
        if not candidate.is_file():
            return None
        obj = _read_yaml(candidate)
        if not isinstance(obj, dict):
            raise ValueError(f"{candidate} must be a YAML mapping")

        # Best-effort fallback: if a platform-specific YAML points to a missing relative executable,
        # fall back to `service.yml` so dev workflows still work without a packaging step.
        if candidate.name != "service.yml":
            try:
                launch = obj.get("launch") if isinstance(obj.get("launch"), dict) else {}
                cmd_raw = str((launch or {}).get("command") or "").strip()
                wd_raw = str((launch or {}).get("workdir") or "./").strip() or "./"
                cmd_path = Path(cmd_raw)
                if cmd_raw and not cmd_path.is_absolute() and ("/" in cmd_raw or "\\" in cmd_raw or cmd_path.suffix):
                    wd_path = Path(wd_raw).expanduser()
                    if not wd_path.is_absolute():
                        wd_path = (service_dir / wd_path).resolve()
                    else:
                        wd_path = wd_path.resolve()
                    resolved = (wd_path / cmd_path).resolve()
                    if not resolved.is_file():
                        return None
            except Exception:
                # If we can't validate, keep the candidate (don't hide real errors).
                pass

        return obj

    data: Any | None = None
    used_path: Path | None = None
    for candidate in _service_yml_candidates(service_dir):
        try:
            loaded = _try_load_candidate(candidate)
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        if loaded is None:
            continue
        used_path = candidate
        data = loaded
        break

    if used_path is None or data is None:
        tried = ", ".join(str(p.name) for p in _service_yml_candidates(service_dir))
        raise ValueError(f"{service_dir} is missing a service entry YAML (tried: {tried})")

    # Shorthand mapping.
    if "launch" not in data and "command" in data:
        launch = F8ServiceEntry.model_validate(
            {
                "command": data.get("command"),
                "args": data.get("args") or [],
                "env": data.get("env") or {},
                "workdir": data.get("workdir") or "./",
            }
        )
        data = dict(data)
        data["launch"] = launch.model_dump(mode="json")

    try:
        entry = F8ServiceEntry.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Invalid service entry in {service_dir}: {exc}") from exc

    return _absolutize_entry_paths(entry, service_dir=service_dir)


def _read_static_describe_file(service_dir: Path) -> dict[str, Any] | None:
    """
    Optional fast-path: load describe payload from a local file instead of launching a subprocess.

    Supported filenames (in `service_dir`):
    - describe.json
    - describe.yml / describe.yaml

    File format must match the normal `--describe` payload:
      {"schemaVersion":"f8describe/1","service":{...},"operators":[...]}
    """
    if (os.environ.get("F8_DISCOVERY_DISABLE_STATIC_DESCRIBE") or "").strip():
        return None

    service_dir = Path(service_dir).resolve()

    json_path = service_dir / "describe.json"
    if json_path.is_file():
        try:
            raw = json_path.read_text("utf-8")
            obj = json.loads(raw) if raw.strip() else None
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    for name in ("describe.yml", "describe.yaml"):
        yml_path = service_dir / name
        if not yml_path.is_file():
            continue
        try:
            obj = _read_yaml(yml_path)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def _read_inline_describe(entry: F8ServiceEntry) -> dict[str, Any] | None:
    """
    Optional fast-path: allow `service.yml` to carry an inline `describe:` payload.

    This avoids spawning `pixi run ... --describe` for services whose specs are static.
    """
    if (os.environ.get("F8_DISCOVERY_DISABLE_STATIC_DESCRIBE") or "").strip():
        return None

    try:
        extra = entry.model_extra or {}
    except Exception:
        extra = {}
    if not isinstance(extra, dict):
        return None
    obj = extra.get("describe")
    return obj if isinstance(obj, dict) else None


def _describe_entry(service_dir: Path, entry: F8ServiceEntry) -> dict[str, Any] | None:
    """
    Invoke `{entry.launch} --describe` and parse JSON payload.

    Expected JSON:
      {"schemaVersion":"f8describe/1","service":{...},"operators":[...]}
    """
    service_dir = Path(service_dir).resolve()

    inline = _read_inline_describe(entry)
    if inline is not None:
        data: dict[str, Any] = dict(inline)
    else:
        static_file = _read_static_describe_file(service_dir)
        if static_file is not None:
            data = static_file
        else:
            data = {}

    try:
        launch = entry.launch
        describe_args = list(entry.describeArgs or ["--describe"])
        timeout_ms = int(entry.timeoutMs or 4000)
    except Exception:
        return None

    if data:
        # Validate/normalize the payload just like subprocess output.
        out_obj: Any = data
    else:
        out_obj = None

    cmd = [str(launch.command), *[str(a) for a in (launch.args or [])], *[str(a) for a in describe_args]]
    env = os.environ.copy()
    try:
        env.update({str(k): str(v) for k, v in (launch.env or {}).items()})
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    cwd = service_dir
    try:
        wd = str(launch.workdir or "./")
        wd_path = Path(wd).expanduser()
        if not wd_path.is_absolute():
            wd_path = (service_dir / wd_path).resolve()
        else:
            wd_path = wd_path.resolve()
        cwd = wd_path
    except Exception:
        cwd = service_dir

    if out_obj is None:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(0.1, timeout_ms / 1000.0),
                check=False,
            )
        except Exception as exc:
            msg = f"describe subprocess failed for {service_dir}: {exc} (cwd={cwd}, cmd={' '.join(cmd)})"
            _discovery_add_error(msg)
            logger.error(msg)
            return None
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                dt_ms = (time.perf_counter() - t0) * 1000.0
                logger.debug(f"describe took {dt_ms:.1f}ms: {' '.join(cmd)}")

        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()

        def _filter_benign_stderr(text: str) -> str:
            """
            Some launchers print informational messages to stderr even on success.

            Example: `pixi run` prints "Pixi task (...): <cmd>" to stderr by default.
            Treat these as benign so `--describe` discovery still works.
            """
            if not text:
                return ""
            keep: list[str] = []
            for line in str(text).splitlines():
                s = str(line).strip()
                if not s:
                    continue
                if s.startswith("Pixi task ("):
                    continue
                keep.append(line)
            return "\n".join(keep).strip()

        err2 = _filter_benign_stderr(err)
        if err2 and not out:
            msg = f"describe stderr (no stdout) for {service_dir}: {' '.join(cmd)}\n{_truncate_text(err2, max_chars=800)}"
            _discovery_add_error(msg)
            logger.error(msg)
            return None
        elif err2:
            # Don't fail discovery if stdout contains JSON; stderr might contain harmless noise.
            logger.warning(f"Error output from describe command {' '.join(cmd)}:\n{_truncate_text(err2, max_chars=800)}")

        def _extract_last_json_obj(text: str) -> Any | None:
            """
            Best-effort: extract the last JSON value from noisy output (logs + JSON).

            Handles nested objects by using `raw_decode` instead of naive brace matching.
            """
            s = (text or "").strip()
            if not s:
                return None
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass

            decoder = json.JSONDecoder()
            idx = 0
            last: Any | None = None
            while idx < len(s):
                m = re.search(r"[\{\[]", s[idx:])
                if not m:
                    break
                start = idx + m.start()
                try:
                    obj, end = decoder.raw_decode(s[start:])
                    last = obj
                    idx = start + end
                except Exception:
                    idx = start + 1
            return last

        out_obj = _extract_last_json_obj(out)
        if not isinstance(out_obj, dict):
            msg = (
                f"describe produced no JSON object for {service_dir}: {' '.join(cmd)}\n"
                f"stdout:\n{_truncate_text(out, max_chars=600)}\n"
                f"stderr:\n{_truncate_text(err2 or err, max_chars=600)}"
            )
            _discovery_add_error(msg)
            logger.error(msg)
            return None

    data = out_obj

    if not isinstance(data, dict):
        return None
    data = normalize_describe_payload_dict(data)

    try:
        payload = F8ServiceDescribe.model_validate(data)
        data = payload.model_dump(mode="json")
    except Exception:
        # allow loose payloads as long as required keys exist
        if "service" not in data:
            msg = f"describe JSON missing required key 'service' for {service_dir}: {' '.join(cmd)}"
            _discovery_add_error(msg)
            logger.error(msg)
            return None
        if "operators" not in data:
            data["operators"] = []

    # If the described service omitted launch info, inherit it from the discovery entry.
    try:
        svc = data.get("service") or {}
        if isinstance(svc, dict) and not svc.get("launch"):
            svc["launch"] = entry.launch.model_dump(mode="json")
            data["service"] = svc
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    # Optional stability check: if entry.serviceClass is provided, it must match describe output.
    try:
        entry_service_class = str(entry.serviceClass or "").strip()
        described_service_class = str((data.get("service") or {}).get("serviceClass") or "").strip()
        if entry_service_class and described_service_class and entry_service_class != described_service_class:
            msg = (
                f"Service class mismatch for {service_dir}: entry has '{entry_service_class}', described has '{described_service_class}'"
            )
            _discovery_add_error(msg)
            logger.error(msg)
            return None
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    return data


def _describe_entry_timed(service_dir: Path, entry: F8ServiceEntry) -> tuple[dict[str, Any] | None, float, str]:
    """
    Wrapper around `_describe_entry()` that measures elapsed time and reports the source.

    `source` is one of: inline, file, subprocess, none
    """
    t0 = time.perf_counter()

    inline = _read_inline_describe(entry)
    if inline is not None:
        payload = _describe_entry(service_dir, entry)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return payload, dt_ms, "inline"

    static_file = _read_static_describe_file(service_dir)
    if static_file is not None:
        payload = _describe_entry(service_dir, entry)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return payload, dt_ms, "file"

    payload = _describe_entry(service_dir, entry)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return payload, dt_ms, "subprocess"


def _discovery_parallelism(service_count: int) -> int:
    """
    Determine parallelism for describe subprocess launches.

    Env overrides:
    - `F8_DESCRIBE_JOBS`: int; 1 forces sequential
    - `F8_DISCOVERY_JOBS`: int; legacy alias

    Defaults to a small number to avoid thrashing when `pixi run ...` is heavy.
    """
    raw = (os.environ.get("F8_DESCRIBE_JOBS") or os.environ.get("F8_DISCOVERY_JOBS") or "").strip()
    if raw:
        try:
            v = int(raw)
            return max(1, v)
        except Exception:
            return 1

    cpu = os.cpu_count() or 4
    # Favor modest parallelism; `pixi run` tends to be IO-heavy and can contend on locks/caches.
    return max(1, min(service_count, min(6, cpu)))


def _discovery_log_timings_enabled() -> bool:
    raw = (os.environ.get("F8_DISCOVERY_LOG_TIMINGS") or "").strip().lower()
    if raw in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    if raw in ("0", "false", "no", "off", "disable", "disabled", ""):
        return False
    # Unknown value: be safe and enable (user set something non-empty explicitly).
    return True


def _discovery_slow_ms_default() -> float:
    raw = (os.environ.get("F8_DISCOVERY_SLOW_MS") or "").strip()
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.0


def load_discovery_into_registries(*, roots: list[Path] | None = None, overwrite: bool = True) -> list[str]:
    """
    Load `service.yml` discovery entries into in-process registries.

    Returns the list of discovered serviceClass entries.
    """
    global _LAST_DISCOVERY_TIMING_LINES

    _discovery_clear_errors()

    roots = roots if roots is not None else _default_roots()
    catalog = ServiceCatalog.instance()

    found: list[str] = []

    entries: list[tuple[Path, F8ServiceEntry]] = []
    for service_dir in find_service_dirs(roots):
        try:
            entry = load_service_entry(service_dir)
        except ValueError as exc:
            logger.warning(f"Skipping service in {service_dir}: {exc}")
            continue
        entries.append((Path(service_dir).resolve(), entry))

    payload_by_dir: dict[Path, dict[str, Any] | None] = {}
    timing_by_dir: dict[Path, tuple[float, str]] = {}
    jobs = _discovery_parallelism(len(entries))
    if jobs <= 1 or len(entries) <= 1:
        for service_dir, entry in entries:
            payload, dt_ms, source = _describe_entry_timed(service_dir, entry)
            payload_by_dir[service_dir] = payload
            timing_by_dir[service_dir] = (dt_ms, source)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            futs: dict[concurrent.futures.Future[tuple[dict[str, Any] | None, float, str]], tuple[Path, F8ServiceEntry]] = {}
            for service_dir, entry in entries:
                futs[ex.submit(_describe_entry_timed, service_dir, entry)] = (service_dir, entry)

            for fut in concurrent.futures.as_completed(futs):
                service_dir, entry = futs[fut]
                try:
                    payload, dt_ms, source = fut.result()
                    payload_by_dir[service_dir] = payload
                    timing_by_dir[service_dir] = (dt_ms, source)
                except Exception as exc:
                    logger.warning(f"Describe failed for {service_dir}: {exc}")
                    payload_by_dir[service_dir] = None
                    timing_by_dir[service_dir] = (0.0, "none")

    if _discovery_log_timings_enabled() and timing_by_dir:
        slow_ms = _discovery_slow_ms_default()
        rows: list[tuple[float, str, Path]] = []
        for service_dir, entry in entries:
            dt_ms, source = timing_by_dir.get(service_dir, (0.0, "none"))
            label = str(entry.serviceClass or "").strip() or str(service_dir.name)
            rows.append((dt_ms, f"{label} ({source})", service_dir))
        rows.sort(key=lambda x: x[0], reverse=True)

        total_ms = sum(x[0] for x in rows)
        lines: list[str] = [f"service discovery describe timings: services={len(rows)} jobs={jobs} total={total_ms:.1f}ms"]
        for dt_ms, label, service_dir in rows:
            if slow_ms > 0.0 and dt_ms < slow_ms:
                continue
            lines.append(f"{dt_ms:7.1f}ms  {label}  [{service_dir}]")

        errs = last_discovery_error_lines()
        if errs:
            lines.append("")
            lines.append("discovery errors:")
            for e in errs:
                lines.append(f"ERROR {e}")

        _LAST_DISCOVERY_TIMING_LINES = lines

        logger.info(lines[0])
        for line in lines[1:]:
            logger.info(f"  {line}")
    else:
        _LAST_DISCOVERY_TIMING_LINES = []

    for service_dir, entry in entries:
        payload = payload_by_dir.get(service_dir)
        if payload is None:
            continue
        try:
            svc_payload = payload.get("service")
            if isinstance(svc_payload, dict):
                launch = svc_payload.get("launch")
                if isinstance(launch, dict):
                    wd = str(launch.get("workdir") or "./")
                    wd_path = Path(wd).expanduser()
                    if not wd_path.is_absolute():
                        wd_path = (Path(service_dir).resolve() / wd_path).resolve()
                    else:
                        wd_path = wd_path.resolve()
                    launch = dict(launch)
                    launch["workdir"] = str(wd_path)
                    svc_payload = dict(svc_payload)
                    svc_payload["launch"] = launch
                    payload["service"] = svc_payload
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        try:
            svc = catalog.register_service(
                payload["service"],
                # Keep as directory path; launchers need the service root.
                service_entry_path=Path(service_dir).resolve(),
            )
        except Exception as e:
            logger.warning(f"Failed to register service from {service_dir}: {e}")
            continue
        found.append(str(svc.serviceClass))
        try:
            catalog.register_operators(payload.get("operators") or [])
        except Exception as e:
            logger.warning(f"Failed to register operators from {service_dir}: {e}")
    return found
