from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import BinaryIO, cast


logger = logging.getLogger(__name__)


class LanguageServerError(RuntimeError):
    pass


class PythonLanguageServer:
    def __init__(self, *, workspace_root: Path, request_timeout_s: float = 2.0) -> None:
        self._workspace_root = workspace_root.resolve()
        self._request_timeout_s = request_timeout_s
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, object]]] = {}
        self._next_id = 0
        self._stop = threading.Event()

    @property
    def available(self) -> bool:
        return shutil.which("basedpyright-langserver") is not None

    def start(self, *, document_path: Path, text: str, version: int) -> None:
        if self._process is not None:
            return
        executable = shutil.which("basedpyright-langserver")
        if executable is None:
            raise LanguageServerError("basedpyright-langserver is unavailable")
        try:
            process = subprocess.Popen(
                [executable, "--stdio"],
                cwd=self._workspace_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise LanguageServerError(f"failed to start basedpyright language server: {exc}") from exc
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise LanguageServerError("basedpyright language server did not provide stdio pipes")
        self._process = process
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, name="f8studio-lsp-output", daemon=True)
        self._stderr_reader = threading.Thread(target=self._stderr_loop, name="f8studio-lsp-error", daemon=True)
        self._reader.start()
        self._stderr_reader.start()
        root_uri = self._workspace_root.as_uri()
        _ = self._request(
            "initialize",
            {
                "processId": None,
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self._workspace_root.name}],
                "capabilities": {
                    "textDocument": {
                        "completion": {"completionItem": {"snippetSupport": True}},
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                    }
                },
                "initializationOptions": {"basedpyright": {"analysis": {"diagnosticMode": "openFilesOnly"}}},
            },
            timeout_s=10.0,
        )
        self._notify("initialized", {})
        self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": document_path.resolve().as_uri(),
                    "languageId": "python",
                    "version": version,
                    "text": text,
                }
            },
        )

    def change(self, *, document_path: Path, text: str, version: int) -> None:
        self._notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": document_path.resolve().as_uri(), "version": version},
                "contentChanges": [{"text": text}],
            },
        )

    def completion(self, *, document_path: Path, line: int, column: int) -> object:
        return self._request(
            "textDocument/completion",
            {
                "textDocument": {"uri": document_path.resolve().as_uri()},
                "position": {"line": max(0, line), "character": max(0, column)},
            },
        )

    def hover(self, *, document_path: Path, line: int, column: int) -> object:
        return self._request(
            "textDocument/hover",
            {
                "textDocument": {"uri": document_path.resolve().as_uri()},
                "position": {"line": max(0, line), "character": max(0, column)},
            },
        )

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            _ = self._request("shutdown", {}, timeout_s=1.5)
            self._notify("exit", {})
        except (LanguageServerError, BrokenPipeError, OSError):
            logger.exception("basedpyright language server shutdown failed")
        self._stop.set()
        try:
            process.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        self._process = None

    def _request(self, method: str, params: dict[str, object], *, timeout_s: float | None = None) -> object:
        with self._pending_lock:
            self._next_id += 1
            request_id = self._next_id
            response: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
            self._pending[request_id] = response
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            try:
                message = response.get(timeout=timeout_s or self._request_timeout_s)
            except queue.Empty as exc:
                raise LanguageServerError(f"language server request timed out: {method}") from exc
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
        error = message.get("error")
        if isinstance(error, dict):
            typed_error = cast(dict[str, object], error)
            raise LanguageServerError(f"language server error {typed_error.get('code')}: {typed_error.get('message')}")
        return message.get("result")

    def _notify(self, method: str, params: dict[str, object]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise LanguageServerError("language server is not running")
        body = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        with self._write_lock:
            process.stdin.write(header)
            process.stdin.write(body)
            process.stdin.flush()

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            while not self._stop.is_set():
                message = self._read_message(cast(BinaryIO, process.stdout))
                if message is None:
                    return
                identifier = message.get("id")
                if not isinstance(identifier, int):
                    continue
                with self._pending_lock:
                    pending = self._pending.get(identifier)
                if pending is not None:
                    pending.put(message)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            if not self._stop.is_set():
                logger.exception("basedpyright language server output reader failed")

    def _stderr_loop(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while not self._stop.is_set():
            line = process.stderr.readline()
            if not line:
                return
            logger.warning("basedpyright language server: %s", line.decode("utf-8", errors="replace").rstrip())

    @staticmethod
    def _read_message(stream: BinaryIO) -> dict[str, object] | None:
        content_length: int | None = None
        while True:
            line = stream.readline()
            if not line:
                return None
            if line in {b"\r\n", b"\n"}:
                break
            key, separator, value = line.decode("ascii").partition(":")
            if separator and key.strip().lower() == "content-length":
                content_length = int(value.strip())
        if content_length is None:
            raise ValueError("language server response is missing Content-Length")
        raw = stream.read(content_length)
        if len(raw) != content_length:
            return None
        decoded = cast(object, json.loads(raw.decode("utf-8")))
        if not isinstance(decoded, dict):
            raise ValueError("language server response must be an object")
        return cast(dict[str, object], decoded)


__all__ = ["LanguageServerError", "PythonLanguageServer"]
