from __future__ import annotations

from f8studio_server.errors import InvalidRequestError, NotFoundError

import re
from collections.abc import Callable, Mapping
from pathlib import Path


_SKILL_ID = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_MAX_SKILL_BYTES = 64 * 1024


class AgentSkillLibrary:
    def __init__(self, *, user_root: Path, extension_files: Callable[[], Mapping[str, Path]] | None = None) -> None:
        self._extension_files: Callable[[], Mapping[str, Path]] = extension_files or self._empty_extension_files
        self._user_root = user_root.resolve()
        self._bundled_root = (Path(__file__).parent / "skills").resolve()
        self._user_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _empty_extension_files() -> Mapping[str, Path]:
        return {}

    def list(self) -> tuple[str, ...]:
        names: set[str] = set()
        for root in (self._bundled_root, self._user_root):
            if not root.is_dir():
                continue
            names.update(
                path.name for path in root.iterdir()
                if path.is_dir() and _SKILL_ID.fullmatch(path.name) and (path / "SKILL.md").is_file()
            )
        names.update(self._extension_files())
        return tuple(sorted(names))

    def read(self, skill_id: str) -> str:
        if ':' in skill_id:
            path = self._extension_files().get(skill_id)
            if path is None:
                raise NotFoundError(f'extension skill is unavailable: {skill_id}')
            if path.stat().st_size > _MAX_SKILL_BYTES:
                raise InvalidRequestError(f'agent skill is too large: {skill_id}')
            return path.read_text(encoding='utf-8')
        if _SKILL_ID.fullmatch(skill_id) is None:
            raise InvalidRequestError(f"invalid agent skill id: {skill_id}")
        for root in (self._user_root, self._bundled_root):
            path = (root / skill_id / "SKILL.md").resolve()
            if not path.is_relative_to(root) or not path.is_file():
                continue
            if path.stat().st_size > _MAX_SKILL_BYTES:
                raise InvalidRequestError(f"agent skill is too large: {skill_id}")
            return path.read_text(encoding="utf-8")
        raise NotFoundError(f"agent skill not found: {skill_id}")


__all__ = ["AgentSkillLibrary"]
