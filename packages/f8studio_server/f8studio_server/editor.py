from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from threading import RLock
from typing import Literal, cast
from uuid import uuid4

import msgspec
from f8pysdk.specs import F8JsonValue

from .lsp import LanguageServerError, PythonLanguageServer


class EditorSupportFile(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    path: str
    content: str


class CreateEditorSessionRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    language: Literal["python", "json"]
    text: str
    filename: str = "main.py"
    support_files: tuple[EditorSupportFile, ...] = ()


class UpdateEditorDocumentRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    version: int
    text: str


class EditorPosition(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    line: int
    column: int


class EditorRange(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    start: EditorPosition
    end: EditorPosition


class EditorDiagnostic(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    severity: Literal["error", "warning", "information"]
    message: str
    source: str
    path: str
    range: EditorRange
    rule: str | None = None


class EditorSessionRecord(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    session_id: str
    language: Literal["python", "json"]
    filename: str
    version: int
    text: str


class EditorAnalysis(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    session_id: str
    version: int
    diagnostics: tuple[EditorDiagnostic, ...]
    engine: str


class EditorPositionRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    line: int
    column: int


class EditorLanguageResult(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    session_id: str
    version: int
    result: F8JsonValue


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"editor file path must be relative and contained: {value}")
    return path


class _EditorSession:
    def __init__(self, record: EditorSessionRecord, root: Path, language_server: PythonLanguageServer | None) -> None:
        self.record = record
        self.root = root
        self.language_server = language_server


class EditorSessionService:
    def __init__(self, *, root: Path | None = None, basedpyright_command: str = "basedpyright") -> None:
        self._temporary_root = root is None
        self._root = root or Path(tempfile.mkdtemp(prefix="f8studio-editor-"))
        self._root.mkdir(parents=True, exist_ok=True)
        for stale in self._root.iterdir():
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
        self._basedpyright_command = basedpyright_command
        self._sessions: dict[str, _EditorSession] = {}
        self._lock = RLock()

    @property
    def python_available(self) -> bool:
        return shutil.which(self._basedpyright_command) is not None

    def create(self, request: CreateEditorSessionRequest) -> EditorSessionRecord:
        filename = str(_safe_relative_path(request.filename))
        if request.language == "python" and not filename.endswith((".py", ".pyi")):
            raise ValueError("Python editor filename must end in .py or .pyi")
        if request.language == "json" and not filename.endswith(".json"):
            raise ValueError("JSON editor filename must end in .json")
        session_id = uuid4().hex
        root = self._root / session_id
        root.mkdir(parents=True)
        record = EditorSessionRecord(
            session_id=session_id,
            language=request.language,
            filename=filename,
            version=1,
            text=request.text,
        )
        language_server: PythonLanguageServer | None = None
        try:
            document_path = Path(filename)
            self._write(root, document_path, request.text)
            written_paths = {document_path}
            for support in request.support_files:
                support_path = _safe_relative_path(support.path)
                if support_path in written_paths:
                    raise ValueError(f"duplicate editor file path: {support.path}")
                self._write(root, support_path, support.content)
                written_paths.add(support_path)
            if request.language == "python":
                language_server = PythonLanguageServer(workspace_root=root)
                language_server.start(document_path=root / filename, text=request.text, version=record.version)
        except (LanguageServerError, BrokenPipeError, OSError, ValueError):
            if language_server is not None:
                language_server.close()
            shutil.rmtree(root, ignore_errors=True)
            raise
        with self._lock:
            self._sessions[session_id] = _EditorSession(record, root, language_server)
        return record

    def get(self, session_id: str) -> EditorSessionRecord:
        with self._lock:
            return self._session(session_id).record

    def update(self, session_id: str, request: UpdateEditorDocumentRequest) -> EditorSessionRecord:
        with self._lock:
            session = self._session(session_id)
            expected = session.record.version + 1
            if request.version != expected:
                raise ValueError(f"editor document version must be {expected}, got {request.version}")
            record = EditorSessionRecord(
                session_id=session.record.session_id,
                language=session.record.language,
                filename=session.record.filename,
                version=request.version,
                text=request.text,
            )
            self._write(session.root, Path(record.filename), record.text)
            if session.language_server is not None:
                session.language_server.change(
                    document_path=session.root / record.filename,
                    text=record.text,
                    version=record.version,
                )
            session.record = record
            return record

    def completion(self, session_id: str, request: EditorPositionRequest) -> EditorLanguageResult:
        with self._lock:
            session = self._session(session_id)
            if session.language_server is None:
                raise ValueError("completion is only available for Python editor sessions")
            result = session.language_server.completion(
                document_path=session.root / session.record.filename,
                line=request.line,
                column=request.column,
            )
            return EditorLanguageResult(
                session_id=session_id,
                version=session.record.version,
                result=cast(F8JsonValue, msgspec.to_builtins(result, str_keys=True)),
            )

    def hover(self, session_id: str, request: EditorPositionRequest) -> EditorLanguageResult:
        with self._lock:
            session = self._session(session_id)
            if session.language_server is None:
                raise ValueError("hover is only available for Python editor sessions")
            result = session.language_server.hover(
                document_path=session.root / session.record.filename,
                line=request.line,
                column=request.column,
            )
            return EditorLanguageResult(
                session_id=session_id,
                version=session.record.version,
                result=cast(F8JsonValue, msgspec.to_builtins(result, str_keys=True)),
            )

    def analyze(self, session_id: str) -> EditorAnalysis:
        with self._lock:
            session = self._session(session_id)
            record = session.record
            root = session.root
        if record.language == "json":
            diagnostics = self._analyze_json(record)
            return EditorAnalysis(
                session_id=session_id,
                version=record.version,
                diagnostics=diagnostics,
                engine="json",
            )
        if not self.python_available:
            raise RuntimeError("basedpyright is unavailable in the Web Studio runtime environment")
        completed = subprocess.run(
            [self._basedpyright_command, "--outputjson", record.filename],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20.0,
            check=False,
        )
        if completed.returncode not in {0, 1}:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"basedpyright failed with exit code {completed.returncode}: {detail}")
        try:
            payload = cast(object, json.loads(completed.stdout))
        except json.JSONDecodeError as exc:
            raise RuntimeError("basedpyright returned invalid JSON") from exc
        return EditorAnalysis(
            session_id=session_id,
            version=record.version,
            diagnostics=self._pyright_diagnostics(payload),
            engine="basedpyright",
        )

    def close_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise FileNotFoundError(f"editor session not found: {session_id}")
        if session.language_server is not None:
            session.language_server.close()
        shutil.rmtree(session.root)

    def close(self) -> None:
        with self._lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            if session.language_server is not None:
                session.language_server.close()
            shutil.rmtree(session.root, ignore_errors=True)
        if self._temporary_root:
            shutil.rmtree(self._root, ignore_errors=True)

    def _session(self, session_id: str) -> _EditorSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise FileNotFoundError(f"editor session not found: {session_id}")
        return session

    @staticmethod
    def _write(root: Path, relative: Path, content: str) -> None:
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root.resolve()):
            raise ValueError("editor file escapes session workspace")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    @staticmethod
    def _analyze_json(record: EditorSessionRecord) -> tuple[EditorDiagnostic, ...]:
        try:
            json.loads(record.text)
        except json.JSONDecodeError as exc:
            position = EditorPosition(line=max(0, exc.lineno - 1), column=max(0, exc.colno - 1))
            return (
                EditorDiagnostic(
                    severity="error",
                    message=exc.msg,
                    source="json",
                    path=record.filename,
                    range=EditorRange(start=position, end=position),
                ),
            )
        return ()

    @staticmethod
    def _pyright_diagnostics(payload: object) -> tuple[EditorDiagnostic, ...]:
        if not isinstance(payload, dict):
            raise RuntimeError("basedpyright output must be an object")
        typed_payload = cast(dict[str, object], payload)
        raw_diagnostics = typed_payload.get("generalDiagnostics")
        if not isinstance(raw_diagnostics, list):
            return ()
        diagnostics: list[EditorDiagnostic] = []
        for raw_value in cast(list[object], raw_diagnostics):
            raw = cast(dict[str, object], raw_value) if isinstance(raw_value, dict) else None
            if raw is None:
                continue
            raw_range = raw.get("range")
            if not isinstance(raw_range, dict):
                continue
            typed_range = cast(dict[str, object], raw_range)
            start = typed_range.get("start")
            end = typed_range.get("end")
            if not isinstance(start, dict) or not isinstance(end, dict):
                continue
            typed_start = cast(dict[str, object], start)
            typed_end = cast(dict[str, object], end)
            severity_value = str(raw.get("severity") or "information")
            severity: Literal["error", "warning", "information"]
            if severity_value == "error":
                severity = "error"
            elif severity_value == "warning":
                severity = "warning"
            else:
                severity = "information"
            diagnostics.append(
                EditorDiagnostic(
                    severity=severity,
                    message=str(raw.get("message") or ""),
                    source="basedpyright",
                    path=Path(str(raw.get("file") or "")).name,
                    range=EditorRange(
                        start=EditorPosition(
                            line=EditorSessionService._integer(typed_start.get("line")),
                            column=EditorSessionService._integer(typed_start.get("character")),
                        ),
                        end=EditorPosition(
                            line=EditorSessionService._integer(typed_end.get("line")),
                            column=EditorSessionService._integer(typed_end.get("character")),
                        ),
                    ),
                    rule=str(raw["rule"]) if raw.get("rule") is not None else None,
                )
            )
        return tuple(diagnostics)

    @staticmethod
    def _integer(value: object) -> int:
        return value if isinstance(value, int) else 0


__all__ = [
    "CreateEditorSessionRequest",
    "EditorAnalysis",
    "EditorDiagnostic",
    "EditorLanguageResult",
    "EditorPosition",
    "EditorPositionRequest",
    "EditorRange",
    "EditorSessionRecord",
    "EditorSessionService",
    "EditorSupportFile",
    "UpdateEditorDocumentRequest",
]
