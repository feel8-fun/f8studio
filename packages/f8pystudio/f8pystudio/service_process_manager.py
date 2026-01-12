from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .service_catalog.discovery import load_service_entry
from .service_catalog.service_catalog import ServiceCatalog


@dataclass(frozen=True)
class ServiceProcessConfig:
    service_class: str
    service_id: str
    nats_url: str = "nats://127.0.0.1:4222"


class ServiceProcessManager:
    """
    Launch/track local service processes based on discovery `service.yml`.

    This is intentionally lightweight (no RPC): Studio deploys graphs via NATS KV.
    """

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen[Any]] = {}  # service_id -> process

    def is_running(self, service_id: str) -> bool:
        p = self._procs.get(str(service_id))
        if p is None:
            return False
        return p.poll() is None

    def stop(self, service_id: str) -> None:
        sid = str(service_id)
        p = self._procs.get(sid)
        if p is None:
            return
        try:
            p.terminate()
        except Exception:
            pass
        self._procs.pop(sid, None)

    def start(self, cfg: ServiceProcessConfig) -> None:
        service_class = str(cfg.service_class).strip()
        service_id = str(cfg.service_id).strip()
        nats_url = str(cfg.nats_url).strip()

        catalog = ServiceCatalog.instance()
        entry_path = catalog.service_entry_path(service_class)
        if entry_path is None:
            raise ValueError(f"Missing discovery entry path for serviceClass={service_class!r}")
        service_dir = Path(entry_path).resolve()
        entry = load_service_entry(service_dir)

        if self.is_running(service_id):
            return

        launch = entry.launch
        cmd = [str(launch.command), *[str(a) for a in (launch.args or [])]]

        # Our python services accept CLI args, but also read env; supply both.
        cmd += ["--service-id", service_id, "--nats-url", nats_url]

        env = os.environ.copy()
        try:
            env.update({str(k): str(v) for k, v in (launch.env or {}).items()})
        except Exception:
            pass
        env["F8_SERVICE_ID"] = service_id
        env["F8_NATS_URL"] = nats_url

        workdir = Path(getattr(launch, "workdir", "./") or "./").expanduser()
        if not workdir.is_absolute():
            workdir = (service_dir / workdir).resolve()
        else:
            workdir = workdir.resolve()

        p = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            env=env,
            stdout=None,
            stderr=None,
        )
        self._procs[service_id] = p

