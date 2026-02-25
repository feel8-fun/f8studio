from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NavTarget:
    title_path: tuple[str, ...]
    raw_target: str


@dataclass(frozen=True)
class NavIssue:
    title_path: tuple[str, ...]
    raw_target: str
    message: str


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ValueError(f"config file does not exist: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid config root type in {config_path}: expected mapping")
    return raw


def _collect_targets(node: Any, trail: tuple[str, ...], targets: list[NavTarget]) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_targets(item, trail, targets)
        return

    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"invalid nav key under {' > '.join(trail) or '<root>'}")
            next_trail = (*trail, key)
            if isinstance(value, str):
                targets.append(NavTarget(title_path=next_trail, raw_target=value))
            elif isinstance(value, list):
                _collect_targets(value, next_trail, targets)
            elif isinstance(value, dict):
                _collect_targets(value, next_trail, targets)
            else:
                raise ValueError(
                    f"invalid nav value type for {' > '.join(next_trail)}: {type(value).__name__}"
                )
        return

    if isinstance(node, str):
        targets.append(NavTarget(title_path=trail, raw_target=node))
        return

    raise ValueError(f"invalid nav node type under {' > '.join(trail) or '<root>'}: {type(node).__name__}")


def _is_external_target(target: str) -> bool:
    lowered = target.lower()
    return (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("mailto:")
        or lowered.startswith("tel:")
        or lowered.startswith("data:")
        or lowered.startswith("#")
    )


def _normalize_target_path(raw_target: str, docs_root: Path) -> Path:
    target_text = raw_target.strip()
    if not target_text:
        raise ValueError("empty nav target")

    target_no_fragment = target_text.split("#", 1)[0].split("?", 1)[0].strip()
    if not target_no_fragment:
        raise ValueError("empty nav target after fragment/query removal")

    if target_no_fragment.startswith("/"):
        target_path = docs_root / target_no_fragment.lstrip("/")
    else:
        target_path = docs_root / target_no_fragment

    return target_path.resolve()


def validate_nav(config_path: Path, docs_root: Path) -> list[NavIssue]:
    config = _load_config(config_path)
    nav_value = config.get("nav")
    if not isinstance(nav_value, list):
        raise ValueError(f"invalid nav in {config_path}: expected list")

    docs_root_resolved = docs_root.resolve()
    if not docs_root_resolved.exists():
        raise ValueError(f"docs root does not exist: {docs_root_resolved}")

    targets: list[NavTarget] = []
    _collect_targets(nav_value, tuple(), targets)

    issues: list[NavIssue] = []
    for target in targets:
        if _is_external_target(target.raw_target):
            continue

        try:
            resolved_target = _normalize_target_path(target.raw_target, docs_root_resolved)
        except ValueError as exc:
            issues.append(
                NavIssue(
                    title_path=target.title_path,
                    raw_target=target.raw_target,
                    message=str(exc),
                )
            )
            continue

        try:
            resolved_target.relative_to(docs_root_resolved)
        except ValueError:
            issues.append(
                NavIssue(
                    title_path=target.title_path,
                    raw_target=target.raw_target,
                    message=f"target escapes docs root: {resolved_target}",
                )
            )
            continue

        if not resolved_target.exists():
            issues.append(
                NavIssue(
                    title_path=target.title_path,
                    raw_target=target.raw_target,
                    message=f"target file does not exist: {resolved_target}",
                )
            )

    return issues


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate mkdocs nav file targets")
    parser.add_argument("--config-path", default="mkdocs.yml", help="Path to mkdocs-compatible config")
    parser.add_argument("--docs-root", default="docs", help="Docs root directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config_path).resolve()
    docs_root = Path(args.docs_root).resolve()

    try:
        issues = validate_nav(config_path, docs_root)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    if issues:
        print("nav validation failed:")
        for issue in issues:
            nav_path = " > ".join(issue.title_path) if issue.title_path else "<root>"
            print(f"- {nav_path}: {issue.raw_target} -> {issue.message}")
        return 1

    print("nav validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
