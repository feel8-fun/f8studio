from __future__ import annotations
from typing import Any

import json
import logging
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
import yaml

from f8pysdk import F8ServiceDescribe, F8ServiceEntry

from .service_catalog import ServiceCatalog


logger = logging.getLogger(__name__)


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
    except Exception:
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


def find_service_dirs(roots: Iterable[Path]) -> list[Path]:
    """
    Find directories containing a `service.yml`.

    Discovery structure:
      <root>/<serviceClass>/service.yml
      <root>/<serviceClass>/operators.yml   (optional)

    Note:
    - `serviceClass` is slash-separated (eg. `f8/engine`), so services may live
      in nested directories like `<root>/f8/engine/service.yml`.
    """
    found: list[Path] = []
    for root in roots:
        r = Path(root).expanduser()
        if not r.is_absolute():
            r = (Path.cwd() / r).resolve()
        else:
            r = r.resolve()

        if not r.exists() or not r.is_dir():
            continue
        try:
            for svc_file in sorted(r.rglob("service.yml")):
                if svc_file.is_file():
                    found.append(svc_file.parent.resolve())
        except Exception:
            # Fallback to the immediate children layout.
            for child in sorted(r.iterdir()):
                if not child.is_dir():
                    continue
                if (child / "service.yml").is_file():
                    found.append(child.resolve())
    return found


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

    payload["launch"] = launch
    return F8ServiceEntry.model_validate(payload)


def load_service_entry(service_dir: Path) -> F8ServiceEntry:
    """
    Load a minimal discovery entry from `service.yml`.

    Supported forms:
    - `schemaVersion: f8serviceEntry/1` + `launch: {...}`
    - shorthand: `command: ...` + optional `args/env/workdir` (will be mapped into `launch`)
    """
    service_dir = Path(service_dir).resolve()
    data = _read_yaml(service_dir / "service.yml")
    if not isinstance(data, dict):
        raise ValueError(f"{service_dir}/service.yml must be a YAML mapping")

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


def _describe_entry(service_dir: Path, entry: F8ServiceEntry) -> dict[str, Any] | None:
    """
    Invoke `{entry.launch} --describe` and parse JSON payload.

    Expected JSON:
      {"schemaVersion":"f8describe/1","service":{...},"operators":[...]}
    """
    service_dir = Path(service_dir).resolve()

    try:
        launch = entry.launch
        describe_args = list(getattr(entry, "describeArgs", None) or ["--describe"])
        timeout_ms = int(getattr(entry, "timeoutMs", 4000) or 4000)
    except Exception:
        return None

    cmd = [str(launch.command), *[str(a) for a in (launch.args or [])], *[str(a) for a in describe_args]]
    env = os.environ.copy()
    try:
        env.update({str(k): str(v) for k, v in (launch.env or {}).items()})
    except Exception:
        pass

    cwd = service_dir
    try:
        wd = str(getattr(launch, "workdir", "./") or "./")
        wd_path = Path(wd).expanduser()
        if not wd_path.is_absolute():
            wd_path = (service_dir / wd_path).resolve()
        else:
            wd_path = wd_path.resolve()
        cwd = wd_path
    except Exception:
        cwd = service_dir

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
    except Exception:
        return None

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if err and not out:
        logger.error(f"Error output from describe command {' '.join(cmd)}:\n{err}")
        return None
    elif err:
        logger.warning(f"Error output from describe command {' '.join(cmd)}:\n{err}")
        return None

    data: dict[str, Any] | None = None
    try:
        data = json.loads(out)
    except Exception:
        # best-effort: extract the last JSON object from noisy output
        start = out.rfind("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(out[start : end + 1])
            except Exception:
                data = None

    if not isinstance(data, dict):
        return None

    try:
        payload = F8ServiceDescribe.model_validate(data)
        data = payload.model_dump(mode="json")
    except Exception:
        # allow loose payloads as long as required keys exist
        if "service" not in data:
            return None
        if "operators" not in data:
            data["operators"] = []

    # If the described service omitted launch info, inherit it from the discovery entry.
    try:
        svc = data.get("service") or {}
        if isinstance(svc, dict) and not svc.get("launch"):
            svc["launch"] = entry.launch.model_dump(mode="json")
            data["service"] = svc
    except Exception:
        pass

    # Optional stability check: if entry.serviceClass is provided, it must match describe output.
    try:
        entry_service_class = str(getattr(entry, "serviceClass", "") or "").strip()
        described_service_class = str((data.get("service") or {}).get("serviceClass") or "").strip()
        if entry_service_class and described_service_class and entry_service_class != described_service_class:
            logger.error(
                f"Service class mismatch for {service_dir}: entry has '{entry_service_class}', described has '{described_service_class}'"
            )
            return None
    except Exception:
        pass

    return data


def load_discovery_into_registries(*, roots: list[Path] | None = None, overwrite: bool = True) -> list[str]:
    """
    Load `service.yml` discovery entries into in-process registries.

    Returns the list of discovered serviceClass entries.
    """
    roots = roots if roots is not None else _default_roots()
    catalog = ServiceCatalog.instance()

    found: list[str] = []
    for service_dir in find_service_dirs(roots):
        try:
            entry = load_service_entry(service_dir)
        except ValueError as exc:
            logger.warning(f"Skipping service in {service_dir}: {exc}")
            continue

        payload = _describe_entry(service_dir, entry)
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
        except Exception:
            pass
        try:
            svc = catalog.register_service(
                payload["service"],
                service_entry_path=(Path(service_dir) / "service.yml").resolve(),
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
