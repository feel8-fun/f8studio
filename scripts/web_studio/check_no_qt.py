from __future__ import annotations

import ast
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
import re
import sys
import tomllib
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    REPO_ROOT / "packages",
    REPO_ROOT / "scripts",
    REPO_ROOT / "tests",
    REPO_ROOT / "external" / "f8unitymods",
)
IGNORED_PARTS = frozenset({".git", ".pixi", "__pycache__", "build", "dist", "node_modules", "site"})
FORBIDDEN_IMPORT_ROOTS = frozenset({"NodeGraphQt", "PyQt5", "PyQt6", "PySide2", "PySide6", "pyqtgraph", "qtpy"})
FORBIDDEN_DISTRIBUTIONS = frozenset({"nodegraphqt", "pyqt5", "pyqt6", "pyside2", "pyside6", "pyqtgraph", "qtpy"})
FORBIDDEN_PACKAGE_PATHS = (
    REPO_ROOT / "packages" / "f8pystudio",
    REPO_ROOT / "packages" / "f8pystudio_ext_template_match",
    REPO_ROOT / "packages" / "f8pystudio_ext_viz_tcode",
)
MANIFEST_PATHS = (
    REPO_ROOT / "pixi.toml",
    *(path for path in (REPO_ROOT / "packages").glob("*/pyproject.toml")),
    REPO_ROOT / "external" / "f8unitymods" / "pyproject.toml",
    REPO_ROOT / "external" / "f8unitymods" / "pixi.toml",
)
DEPENDENCY_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]+")


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_PARTS for part in path.parts)


def _python_sources() -> Iterator[Path]:
    for source_root in SOURCE_ROOTS:
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.py")):
            if not _is_ignored(path):
                yield path


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _manifest_dependency_names(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        for key, nested in mapping.items():
            if key in {"dependencies", "pypi-dependencies", "optional-dependencies"} and isinstance(nested, dict):
                dependency_mapping = cast(dict[str, object], nested)
                yield from (_normalized_distribution_name(name) for name in dependency_mapping)
            elif key == "dependencies" and isinstance(nested, list):
                dependency_list = cast(list[object], nested)
                for item in dependency_list:
                    if not isinstance(item, str):
                        continue
                    match = DEPENDENCY_TOKEN_RE.match(item)
                    if match is not None:
                        yield _normalized_distribution_name(match.group(0))
            yield from _manifest_dependency_names(cast(object, nested))
    elif isinstance(value, list):
        sequence = cast(list[object], value)
        for nested in sequence:
            yield from _manifest_dependency_names(nested)


def _qt_library_paths(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    for pattern in ("libQt*.so*", "Qt*.dll", "Qt*.framework"):
        for path in root.rglob(pattern):
            if path.is_file() or path.is_dir():
                yield path


def main() -> None:
    violations: list[str] = []
    for package_path in FORBIDDEN_PACKAGE_PATHS:
        if package_path.exists():
            violations.append(f"legacy GUI package still exists: {package_path.relative_to(REPO_ROOT)}")

    for path in _python_sources():
        forbidden = imported_roots(path) & FORBIDDEN_IMPORT_ROOTS
        if forbidden:
            names = ", ".join(sorted(forbidden))
            violations.append(f"{path.relative_to(REPO_ROOT)} imports {names}")

    normalized_forbidden = {_normalized_distribution_name(name) for name in FORBIDDEN_DISTRIBUTIONS}
    for manifest_path in MANIFEST_PATHS:
        if not manifest_path.is_file():
            continue
        with manifest_path.open("rb") as manifest_file:
            manifest = tomllib.load(manifest_file)
        forbidden_dependencies = set(_manifest_dependency_names(manifest)) & normalized_forbidden
        if forbidden_dependencies:
            violations.append(
                f"{manifest_path.relative_to(REPO_ROOT)} declares "
                + ", ".join(sorted(forbidden_dependencies))
            )

    installed = {
        _normalized_distribution_name(str(distribution.metadata["Name"] or ""))
        for distribution in metadata.distributions()
    }
    forbidden_installed = installed & normalized_forbidden
    if forbidden_installed:
        violations.append(f"forbidden distributions installed: {', '.join(sorted(forbidden_installed))}")

    loaded = {name.split(".", 1)[0] for name in sys.modules}
    forbidden_loaded = loaded & FORBIDDEN_IMPORT_ROOTS
    if forbidden_loaded:
        violations.append(f"forbidden modules loaded: {', '.join(sorted(forbidden_loaded))}")

    qt_libraries = sorted(_qt_library_paths(Path(sys.prefix)))
    if qt_libraries:
        sample = ", ".join(str(path.relative_to(sys.prefix)) for path in qt_libraries[:10])
        violations.append(f"Qt runtime libraries installed under {sys.prefix}: {sample}")

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
    print(f"Qt-free repository and runtime verified: {API_PROTOCOL_VERSION}; app={app.title}")


if __name__ == "__main__":
    main()
