from __future__ import annotations

import ast
from importlib import metadata
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    REPO_ROOT / "packages" / "f8studio_core" / "f8studio_core",
    REPO_ROOT / "packages" / "f8media_protocol" / "f8media_protocol",
    REPO_ROOT / "packages" / "f8media_gateway" / "f8media_gateway",
    REPO_ROOT / "packages" / "f8studio_server" / "f8studio_server",
)
FORBIDDEN_IMPORT_ROOTS = frozenset({"NodeGraphQt", "PyQt5", "PyQt6", "PySide2", "PySide6", "pyqtgraph", "qtpy"})
FORBIDDEN_DISTRIBUTIONS = frozenset({"nodegraphqt", "pyqt5", "pyqt6", "pyside2", "pyside6", "pyqtgraph", "qtpy"})


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def main() -> None:
    violations: list[str] = []
    for source_root in SOURCE_ROOTS:
        for path in sorted(source_root.rglob("*.py")):
            forbidden = imported_roots(path) & FORBIDDEN_IMPORT_ROOTS
            if forbidden:
                names = ", ".join(sorted(forbidden))
                violations.append(f"{path.relative_to(REPO_ROOT)} imports {names}")

    installed = {str(distribution.metadata["Name"] or "").lower() for distribution in metadata.distributions()}
    forbidden_installed = installed & FORBIDDEN_DISTRIBUTIONS
    if forbidden_installed:
        violations.append(f"forbidden distributions installed: {', '.join(sorted(forbidden_installed))}")

    loaded = {name.split(".", 1)[0] for name in sys.modules}
    forbidden_loaded = loaded & FORBIDDEN_IMPORT_ROOTS
    if forbidden_loaded:
        violations.append(f"forbidden modules loaded: {', '.join(sorted(forbidden_loaded))}")

    if violations:
        raise RuntimeError("Qt-free boundary check failed:\n" + "\n".join(violations))

    from f8studio_core import API_PROTOCOL_VERSION
    from f8studio_server import create_app

    unexpected_media_modules = {"aiortc", "av", "f8media_gateway.service"} & set(sys.modules)
    if unexpected_media_modules:
        raise RuntimeError(
            "Studio control-plane import loaded media implementation modules: "
            + ", ".join(sorted(unexpected_media_modules))
        )

    app = create_app()
    unexpected_media_modules = {"aiortc", "av", "f8media_gateway.service"} & set(sys.modules)
    if unexpected_media_modules:
        raise RuntimeError(
            "Studio app construction loaded media implementation modules: "
            + ", ".join(sorted(unexpected_media_modules))
        )
    print(f"Qt-free boundary verified: {API_PROTOCOL_VERSION}; app={app.title}")


if __name__ == "__main__":
    main()
