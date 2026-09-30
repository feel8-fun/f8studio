"""Explicit writable resource locations shared by standalone services."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def model_root() -> Path:
    configured = os.environ.get("F8_MODEL_ROOT")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError("F8_MODEL_ROOT must be absolute")
        return path
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "f8studio" / "models"


def service_config_root() -> Path:
    configured = os.environ.get("F8_CONFIG_ROOT")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError("F8_CONFIG_ROOT must be absolute")
        return path
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "f8studio" / "services"
