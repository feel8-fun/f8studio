from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse


def _log(log_cb: Callable[[str], None] | None, message: str) -> None:
    if log_cb is None:
        return
    log_cb(str(message))


def _parse_nats_host_port(nats_url: str) -> tuple[str, int]:
    raw = str(nats_url or "").strip()
    if not raw:
        raw = "nats://127.0.0.1:4222"
    if "://" not in raw:
        raw = f"nats://{raw}"
    parsed = urlparse(raw)
    host = str(parsed.hostname or "127.0.0.1").strip()
    port = int(parsed.port or 4222)
    return host, port


def _is_local_host(host: str) -> bool:
    h = str(host or "").strip().lower()
    return h in {"127.0.0.1", "localhost", "::1"}


def _is_tcp_reachable(host: str, port: int, timeout_s: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
            return True
    except (ConnectionError, OSError, TimeoutError):
        return False


def _platform_asset_suffix() -> tuple[str, str]:
    system_name = platform.system().lower().strip()
    machine_name = platform.machine().lower().strip()

    if system_name.startswith("win"):
        os_name = "windows"
    elif system_name.startswith("linux"):
        os_name = "linux"
    elif system_name.startswith("darwin"):
        os_name = "darwin"
    else:
        raise RuntimeError(f"Unsupported platform system: {system_name!r}")

    if machine_name in {"amd64", "x86_64"}:
        arch_name = "amd64"
    elif machine_name in {"arm64", "aarch64"}:
        arch_name = "arm64"
    else:
        raise RuntimeError(f"Unsupported platform arch: {machine_name!r}")

    return os_name, arch_name


def _download_latest_release_archive(download_dir: Path, *, log_cb: Callable[[str], None] | None) -> Path:
    api_url = "https://api.github.com/repos/nats-io/nats-server/releases/latest"
    req = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "f8pystudio-nats-bootstrap",
        },
    )
    with urllib.request.urlopen(req, timeout=12.0) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("GitHub release response missing assets")

    os_name, arch_name = _platform_asset_suffix()
    prefix = f"-{os_name}-{arch_name}."
    chosen_url = ""
    chosen_name = ""

    preferred_exts = [".zip", ".tar.gz"] if os_name == "windows" else [".tar.gz", ".zip"]
    for ext in preferred_exts:
        for item in assets:
            if not isinstance(item, dict):
                continue
            name_value = str(item.get("name") or "").strip()
            url_value = str(item.get("browser_download_url") or "").strip()
            if not name_value or not url_value:
                continue
            if prefix in name_value and name_value.endswith(ext):
                chosen_name = name_value
                chosen_url = url_value
                break
        if chosen_url:
            break

    if not chosen_url:
        raise RuntimeError(f"No release asset matched platform {os_name}/{arch_name}")

    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / chosen_name
    _log(log_cb, f"NATS bootstrap: downloading {chosen_name}")
    with urllib.request.urlopen(chosen_url, timeout=30.0) as resp, archive_path.open("wb") as out_f:
        shutil.copyfileobj(resp, out_f)
    return archive_path


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    name = archive_path.name.lower()
    extract_dir.mkdir(parents=True, exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            zf.extractall(path=extract_dir)
        return
    if name.endswith(".tar.gz"):
        with tarfile.open(archive_path, mode="r:gz") as tf:
            tf.extractall(path=extract_dir)
        return
    raise RuntimeError(f"Unsupported archive format: {archive_path.name}")


def _locate_binary(extract_dir: Path) -> Path:
    exe_name = "nats-server.exe" if platform.system().lower().startswith("win") else "nats-server"
    for p in extract_dir.rglob(exe_name):
        if p.is_file():
            return p
    raise RuntimeError(f"Extracted archive did not contain {exe_name}")


def _install_downloaded_binary(*, source_binary: Path, log_cb: Callable[[str], None] | None) -> Path:
    target_root = Path.home() / ".f8" / "nats-server" / "nats-server"
    target_root.mkdir(parents=True, exist_ok=True)
    target_name = "nats-server.exe" if platform.system().lower().startswith("win") else "nats-server"
    target_binary = target_root / target_name
    shutil.copy2(source_binary, target_binary)
    if not platform.system().lower().startswith("win"):
        target_binary.chmod(0o755)
    _log(log_cb, f"NATS bootstrap: installed binary at {target_binary}")
    return target_binary


def _resolve_nats_server_binary(log_cb: Callable[[str], None] | None) -> Path:
    path_hit = shutil.which("nats-server")
    if path_hit:
        return Path(path_hit).resolve()

    installed = Path.home() / ".f8" / "nats-server" / "nats-server"
    installed_exe = installed / ("nats-server.exe" if platform.system().lower().startswith("win") else "nats-server")
    if installed_exe.is_file():
        return installed_exe

    with tempfile.TemporaryDirectory(prefix="f8-nats-") as td:
        temp_root = Path(td)
        archive = _download_latest_release_archive(temp_root, log_cb=log_cb)
        extract_dir = temp_root / "extract"
        _extract_archive(archive, extract_dir)
        source_binary = _locate_binary(extract_dir)
        return _install_downloaded_binary(source_binary=source_binary, log_cb=log_cb)


def _spawn_nats_server(binary: Path, *, log_cb: Callable[[str], None] | None) -> None:
    cmd = [str(binary), "-js"]
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if platform.system().lower().startswith("win"):
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)
    _log(log_cb, f"NATS bootstrap: started {' '.join(cmd)}")


def ensure_nats_server(nats_url: str, *, log_cb: Callable[[str], None] | None = None) -> bool:
    """
    Ensure a NATS server is reachable for the target URL.

    Behavior:
    - If already reachable: no-op.
    - If URL is local and unreachable: start local `nats-server -js`.
    - If `nats-server` executable is missing: download latest release for this platform and install to
      `~/.f8/nats-server/nats-server/`.
    """
    host, port = _parse_nats_host_port(nats_url)

    if _is_tcp_reachable(host, port):
        return True

    if not _is_local_host(host):
        _log(log_cb, f"NATS bootstrap: {host}:{port} is unreachable and not local, skip auto-start")
        return False

    try:
        binary = _resolve_nats_server_binary(log_cb)
        _spawn_nats_server(binary, log_cb=log_cb)
    except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
        _log(log_cb, f"NATS bootstrap failed: {type(exc).__name__}: {exc}")
        return False

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if _is_tcp_reachable(host, port, timeout_s=0.3):
            _log(log_cb, f"NATS bootstrap: {host}:{port} is reachable")
            return True
        time.sleep(0.1)

    _log(log_cb, f"NATS bootstrap: started process but {host}:{port} is still unreachable")
    return False
