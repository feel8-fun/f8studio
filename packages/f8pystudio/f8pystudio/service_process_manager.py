from __future__ import annotations

import os
import subprocess
import threading
import time
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
    purge_kv_bucket_on_start: bool = True


class ServiceProcessManager:
    """
    Launch/track local service processes based on discovery `service.yml`.

    This is intentionally lightweight (no RPC): Studio deploys graphs via NATS KV.
    """

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen[Any]] = {}  # service_id -> process
        self._threads: dict[str, threading.Thread] = {}  # service_id -> reader thread

    def _start_reader(self, *, service_id: str, proc: subprocess.Popen[Any], on_output: Any | None) -> None:
        if on_output is None:
            return

        def _run() -> None:
            try:
                stream = proc.stdout
                if stream is None:
                    return
                for line in iter(stream.readline, ""):
                    if line == "" and proc.poll() is not None:
                        break
                    try:
                        on_output(str(service_id), str(line))
                    except Exception:
                        continue
            except Exception:
                return

        t = threading.Thread(target=_run, name=f"svc-log:{service_id}", daemon=True)
        self._threads[service_id] = t
        t.start()

    def is_running(self, service_id: str) -> bool:
        sid = str(service_id)
        p = self._procs.get(sid)
        if p is None:
            return False
        if p.poll() is None:
            return True
        # Process already exited: clean up stale entry so we don't accidentally
        # allow duplicate processes under the same serviceId.
        self._cleanup_entry(sid)
        return False

    def _cleanup_entry(self, service_id: str) -> None:
        sid = str(service_id)
        p = self._procs.pop(sid, None)
        t = self._threads.pop(sid, None)
        if p is None:
            return

        try:
            # Avoid closing stdout while a reader thread is blocked in `readline()`;
            # on Windows this can deadlock the UI thread. Let the pipe close naturally
            # when the process exits; only close if no reader thread is alive.
            if t is None or not t.is_alive():
                if p.stdout:
                    p.stdout.close()
        except Exception:
            pass

    def stop(self, service_id: str) -> bool:
        sid = str(service_id)
        p = self._procs.get(sid)
        if p is None:
            return True
        pid = getattr(p, "pid", None)

        # Best-effort graceful terminate, then ensure the whole process tree is gone.
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=0.8)
        except Exception:
            pass

        # `service.yml` launch commonly uses `pixi run ...` which spawns a child Python process.
        # On Windows, `terminate()` only kills the parent process; explicitly kill the tree.
        if os.name == "nt" and pid:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception:
                pass
        else:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

        # Wait a bit longer for shutdown (especially after taskkill).
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                if p.poll() is not None:
                    break
            except Exception:
                break
            time.sleep(0.05)

        if p.poll() is None:
            # Still alive: keep tracking it so we don't allow duplicates.
            return False

        self._cleanup_entry(sid)
        return True

    def start(self, cfg: ServiceProcessConfig, *, on_output: Any | None = None) -> None:
        service_class = str(cfg.service_class).strip()
        service_id = str(cfg.service_id).strip()
        nats_url = str(cfg.nats_url).strip()

        catalog = ServiceCatalog.instance()
        entry_path = catalog.service_entry_path(service_class)
        if entry_path is None:
            raise ValueError(f"Missing discovery entry path for serviceClass={service_class!r}")
        service_dir = Path(entry_path).resolve()
        # Backwards-compatible: older registries stored `<dir>/service.yml` instead of `<dir>`.
        try:
            if service_dir.is_file() and service_dir.name.lower() == "service.yml":
                service_dir = service_dir.parent.resolve()
        except Exception:
            pass
        entry = load_service_entry(service_dir)

        if self.is_running(service_id):
            return
        # Ensure stale exited procs are removed before starting.
        self._cleanup_entry(service_id)

        # Studio has "root" permissions; clear the service KV bucket before starting a new process
        # to avoid stale `ready=true` from a previous run.
        if cfg.purge_kv_bucket_on_start:
            try:
                from f8pysdk.nats_naming import kv_bucket_for_service
                from f8pysdk.nats_transport import reset_kv_bucket_sync

                reset_kv_bucket_sync(url=nats_url, kv_bucket=kv_bucket_for_service(service_id), timeout_s=2.5)
                try:
                    if on_output is not None:
                        on_output(service_id, "[kv] purged bucket on start\n")
                except Exception:
                    pass
            except Exception as exc:
                try:
                    if on_output is not None:
                        on_output(service_id, f"[kv] purge bucket failed (ignored): {exc}\n")
                except Exception:
                    pass

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
        # Ensure python services flush logs promptly when stdout is piped.
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        workdir = Path(getattr(launch, "workdir", "./") or "./").expanduser()
        if not workdir.is_absolute():
            workdir = (service_dir / workdir).resolve()
        else:
            workdir = workdir.resolve()

        p = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._procs[service_id] = p
        try:
            if on_output is not None:
                on_output(service_id, f"[proc] started pid={getattr(p, 'pid', '?')} cmd={' '.join(cmd)}\n")
        except Exception:
            pass
        self._start_reader(service_id=service_id, proc=p, on_output=on_output)
