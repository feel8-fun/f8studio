from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path


def source_fingerprint(service_dir: Path) -> str | None:
    """Installed bundles ship authoritative static specs; checkouts need evidence.

    Include shared SDK/schema inputs as well as operators so changing an inherited
    field also invalidates the describe cache. No dependency on Git availability.
    """
    root = next((parent for parent in service_dir.resolve().parents
                 if (parent / "pixi.toml").is_file() and (parent / "packages").is_dir()), None)
    if root is None:
        return None
    digest = hashlib.sha256()
    for directory in (root / "packages", root / "schemas", service_dir):
        for parent, directories, files in directory.walk():
            directories[:] = sorted(name for name in directories
                                    if name not in {"tests", "node_modules", "dist", "__pycache__", ".venv", ".git"})
            for name in sorted(files):
                path = parent / name
                if path.suffix not in {".py", ".cpp", ".h", ".hpp", ".yml", ".yaml", ".toml"}:
                    continue
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def write_freshness(service_dir: Path) -> None:
    fingerprint = source_fingerprint(service_dir)
    if fingerprint is not None:
        (service_dir / "describe.fingerprint").write_text(json.dumps({"source": fingerprint}) + "\n", encoding="utf-8")


def static_is_fresh(service_dir: Path) -> bool:
    fingerprint = source_fingerprint(service_dir)
    if fingerprint is None:
        return True
    metadata = service_dir / "describe.fingerprint"
    if not metadata.is_file():
        return False
    # Corrupt metadata must not make a stale cache authoritative.
    try:
        return json.loads(metadata.read_text(encoding="utf-8")) == {"source": fingerprint}
    except (json.JSONDecodeError, UnicodeError):
        logging.getLogger(__name__).debug("Invalid describe cache metadata path=%s", metadata, exc_info=True)
        return False
