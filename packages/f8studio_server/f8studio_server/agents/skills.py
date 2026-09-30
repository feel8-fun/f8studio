from __future__ import annotations

from f8studio_server.errors import InvalidRequestError, NotFoundError

import re
from pathlib import Path


_SKILL_ID = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_MAX_SKILL_BYTES = 64 * 1024


class AgentSkillLibrary:
    def __init__(self, *, user_root: Path) -> None:
        self._user_root = user_root.resolve()
        self._bundled_root = (Path(__file__).parent / "skills").resolve()
        self._user_root.mkdir(parents=True, exist_ok=True)

    def list(self) -> tuple[str, ...]:
        names: set[str] = set()
        for root in (self._bundled_root, self._user_root):
            if not root.is_dir():
                continue
            names.update(
                path.name for path in root.iterdir()
                if path.is_dir() and _SKILL_ID.fullmatch(path.name) and (path / "SKILL.md").is_file()
            )
        return tuple(sorted(names))

    def read(self, skill_id: str) -> str:
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
