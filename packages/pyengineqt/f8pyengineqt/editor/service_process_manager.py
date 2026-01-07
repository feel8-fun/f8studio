from __future__ import annotations

import os
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qtpy import QtCore

from ..engine.nats_naming import cmd_subject, ensure_token, kv_bucket_for_service
from ..runtime.nats_transport import NatsTransport, NatsTransportConfig
from f8pysdk import F8ServiceLaunchSpec


@dataclass(frozen=True)
class ServiceProcessConfig:
    service_id: str
    service_class: str
    nats_url: str
    kv_bucket: str | None = None
    launch: F8ServiceLaunchSpec | None = None
    python_exe: str | None = None
    module: str | None = None
    operator_runtime_modules: str | None = None


class ServiceProcessManager(QtCore.QObject):
    """
    Spawns and monitors service processes (v1).

    For now, this is used to run engine services:
    - one service instance -> one OS process
    - env carries serviceId + NATS config
    """

    statusChanged = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._procs: dict[str, QtCore.QProcess] = {}

    def _safe_emit(self, message: str) -> None:
        try:
            self.statusChanged.emit(str(message))
        except RuntimeError:
            # During shutdown Qt may delete this QObject while QProcess signals are still queued.
            return

    def start_service(self, cfg: ServiceProcessConfig) -> None:
        service_id = ensure_token(cfg.service_id, label="service_id")
        if service_id in self._procs:
            return

        launch = cfg.launch
        if launch is not None and str(getattr(launch, "commandSpec", "") or "").strip():
            program = str(launch.commandSpec)
            args = [str(a) for a in (launch.args or [])]
            workdir = str(getattr(launch, "workdir", "") or "").strip() or "./"
            launch_env = dict(getattr(launch, "env", None) or {})
        else:
            python_exe = cfg.python_exe or sys.executable
            module = cfg.module or "f8pyengineqt.engine.engine_service_process"
            program = str(python_exe)
            args = ["-m", str(module)]
            workdir = "./"
            launch_env = {}
        bucket = (cfg.kv_bucket or "").strip() or kv_bucket_for_service(service_id)
        nats_url = str(cfg.nats_url).strip() or "nats://127.0.0.1:4222"

        proc = QtCore.QProcess(self)
        proc.setProgram(program)
        proc.setArguments(args)
        try:
            proc.setWorkingDirectory(str(Path(workdir).resolve()))
        except Exception:
            pass

        env = QtCore.QProcessEnvironment.systemEnvironment()
        for k, v in launch_env.items():
            try:
                env.insert(str(k), str(v))
            except Exception:
                continue
        env.insert("F8_SERVICE_ID", service_id)
        env.insert("F8_SERVICE_CLASS", str(cfg.service_class))
        env.insert("F8_NATS_URL", nats_url)
        env.insert("F8_NATS_BUCKET", bucket)
        if cfg.operator_runtime_modules and cfg.operator_runtime_modules.strip():
            env.insert("F8_OPERATOR_RUNTIME_MODULES", cfg.operator_runtime_modules.strip())

        # Ensure the repo package folder is importable even when not installed.
        try:
            pkg_root = Path(__file__).resolve().parents[2]  # .../packages/pyengineqt
            existing = env.value("PYTHONPATH") or ""
            add = str(pkg_root)
            if add not in existing.split(os.pathsep):
                env.insert("PYTHONPATH", os.pathsep.join([add, existing]) if existing else add)
        except Exception:
            pass

        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        def _on_ready_read() -> None:
            try:
                data = bytes(proc.readAllStandardOutput() or b"")
            except Exception:
                return
            if not data:
                return
            txt = data.decode("utf-8", errors="replace").rstrip()
            if txt:
                self._safe_emit(f"[svc:{service_id}] {txt}")

        proc.readyReadStandardOutput.connect(_on_ready_read)

        def _on_finished(code: int, status: Any) -> None:
            self._procs.pop(service_id, None)
            self._safe_emit(f"[svc:{service_id}] exited code={code} status={status}")

        proc.finished.connect(_on_finished)
        proc.finished.connect(proc.deleteLater)

        proc.start()
        self._procs[service_id] = proc
        self._safe_emit(f"service: start {cfg.service_class} serviceId={service_id}")

    def stop_service(self, service_id: str) -> None:
        service_id = ensure_token(service_id, label="service_id")
        proc = self._procs.pop(service_id, None)
        if proc is None:
            return
        try:
            proc.terminate()
            if not proc.waitForFinished(1500):
                proc.kill()
                proc.waitForFinished(3000)
        except Exception:
            try:
                proc.kill()
                proc.waitForFinished(3000)
            except Exception:
                pass
        self._safe_emit(f"service: stop serviceId={service_id}")

    def stop_all(self) -> None:
        for sid in list(self._procs.keys()):
            self.stop_service(sid)

    def request_run_once(self, service_id: str, *, nats_url: str | None = None) -> None:
        """
        Publish `svc.<serviceId>.cmd.run` to trigger a single execution pass.
        """
        service_id = ensure_token(service_id, label="service_id")
        url = (nats_url or os.environ.get("F8_NATS_URL") or "nats://127.0.0.1:4222").strip()

        async def _pub() -> None:
            # We use NatsTransport for convenience (shared code, reconnect handling).
            t = NatsTransport(NatsTransportConfig(url=url, kv_bucket=kv_bucket_for_service(service_id)))
            await t.connect()
            try:
                await t.publish(cmd_subject(service_id, "run"), json.dumps({"mode": "once"}).encode("utf-8"))
            finally:
                await t.close()

        try:
            import asyncio

            asyncio.run(_pub())
        except Exception as exc:
            self._safe_emit(f"service: run failed ({exc})")
