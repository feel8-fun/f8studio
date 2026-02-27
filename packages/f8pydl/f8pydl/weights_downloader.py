from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower()


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _acquire_lock(lock_path: Path, *, timeout_s: float) -> int:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    while True:
        try:
            return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting download lock: {lock_path}")
            time.sleep(0.2)


def ensure_onnx_file(
    *,
    onnx_path: Path,
    onnx_url: str,
    onnx_sha256: str,
    timeout_s: float = 300.0,
) -> None:
    path = Path(onnx_path).resolve()
    url = str(onnx_url or "").strip()
    sha256_expected = str(onnx_sha256 or "").strip().lower()
    if not url:
        raise FileNotFoundError(f"ONNX file is missing and onnxUrl is empty: {path}")

    if path.exists():
        if not sha256_expected:
            return
        if _sha256_file(path) == sha256_expected:
            return
        _unlink_if_exists(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".download.lock")
    lock_fd = _acquire_lock(lock_path, timeout_s=timeout_s)
    try:
        if path.exists():
            if not sha256_expected:
                return
            if _sha256_file(path) == sha256_expected:
                return
            _unlink_if_exists(path)

        tmp_path = path.with_suffix(path.suffix + ".download.part")
        _unlink_if_exists(tmp_path)
        hasher = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=float(timeout_s)) as resp, tmp_path.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
        digest = hasher.hexdigest().lower()
        if sha256_expected and digest != sha256_expected:
            _unlink_if_exists(tmp_path)
            raise ValueError(
                f"Downloaded ONNX SHA256 mismatch: expected={sha256_expected}, actual={digest}, file={path}"
            )
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            os.close(lock_fd)
        finally:
            _unlink_if_exists(lock_path)
