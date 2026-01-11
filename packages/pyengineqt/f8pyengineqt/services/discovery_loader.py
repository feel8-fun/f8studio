from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any
import logging

logger = logging.getLogger(__name__)

from .discovery import DiscoveryError, find_service_dirs, load_operator_specs, load_service_entry, load_service_spec
from .service_catalog import ServiceCatalog

from f8pysdk import F8ServiceDescribe


def _default_roots() -> list[Path]:
    """
    Default discovery roots.

    - `F8_SERVICE_DISCOVERY_DIRS` (os.pathsep-separated) overrides.
    - otherwise, use `<repo>/services` if it exists (when running from source).
    """
    env = (os.environ.get("F8_SERVICE_DISCOVERY_DIRS") or "").strip()
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p.strip()]

    # Try to locate repo root when running from source.
    try:
        repo_root = Path(__file__).resolve().parents[4]  # .../packages/pyengineqt
        candidate = repo_root / "services"
        if candidate.is_dir():
            return [candidate]
    except Exception:
        pass
    return []


def load_discovery_into_registries(*, roots: list[Path] | None = None, overwrite: bool = True) -> list[str]:
    """
    Load `service.yml` + optional `operators.yml` into in-process registries.

    Returns the list of discovered serviceClass entries.
    """
    roots = roots if roots is not None else _default_roots()
    catalog = ServiceCatalog.instance()

    found: list[str] = []
    for service_dir in find_service_dirs(roots):
        # 1) Backward compatible: full static service spec in YAML.
        try:
            svc = load_service_spec(service_dir)
            try:
                catalog.register_service(svc, overwrite=overwrite)
            except Exception:
                pass
            found.append(str(svc.serviceClass))

            try:
                ops = load_operator_specs(service_dir, service_class=str(svc.serviceClass))
            except DiscoveryError:
                ops = []
            if ops:
                try:
                    catalog.register_operators(str(svc.serviceClass), ops, overwrite=overwrite)
                except Exception:
                    pass
            continue
        except DiscoveryError:
            pass

        # 2) Minimal entry: launch + (optional) serviceClass, with describe-driven specs.
        try:
            entry = load_service_entry(service_dir)
        except DiscoveryError:
            continue

        payload = _describe_entry(service_dir, entry)
        if payload is None:
            continue
        try:
            svc = catalog.register_service(payload["service"], overwrite=overwrite)
        except Exception as e:
            logger.warning(f"Failed to register service from {service_dir}: {e}")
            continue
        found.append(str(svc.serviceClass))
        try:
            catalog.register_operators(str(svc.serviceClass), payload.get("operators") or [], overwrite=overwrite)
        except Exception:
            pass
    return found


def default_discovery_roots() -> list[Path]:
    return _default_roots()


def _describe_entry(service_dir: Path, entry: Any) -> dict[str, Any] | None:
    """
    Invoke `{entry.launch} --describe` and parse JSON payload.

    Expected JSON:
      {"schemaVersion":"f8describe/1","service":{...},"operators":[...]}
    """
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
        wd_path = Path(wd)
        cwd = (service_dir / wd_path) if not wd_path.is_absolute() else wd_path
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
            svc["launch"] = getattr(entry, "launch").model_dump(mode="json")
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
