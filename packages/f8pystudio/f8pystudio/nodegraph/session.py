from __future__ import annotations

from pathlib import Path


def last_session_path() -> Path:
    """
    Default session file path.

    Stored in the user home folder so the editor can restore the last session.
    """
    return Path.home() / ".f8" / "studio" / "lastSession.json"

