from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_ATTR_PATTERN = re.compile(r"(?:href|src)=['\"]([^'\"]+)['\"]", re.IGNORECASE)


@dataclass(frozen=True)
class LinkReference:
    source_file: Path
    line_number: int
    raw_target: str


@dataclass(frozen=True)
class LinkIssue:
    source_file: Path
    line_number: int
    raw_target: str
    message: str


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


def _trim_target(target: str) -> str:
    text = target.strip()
    if text.startswith("<") and text.endswith(">") and len(text) > 2:
        text = text[1:-1].strip()

    if not text:
        return ""

    # Handle optional markdown title syntax: path "title"
    if " " in text:
        first_token = text.split(" ", 1)[0].strip()
        if first_token:
            return first_token
    return text


def _normalize_link_target(raw_target: str, source_file: Path, docs_root: Path) -> Path:
    target_text = _trim_target(raw_target)
    target_no_fragment = target_text.split("#", 1)[0].split("?", 1)[0].strip()
    if not target_no_fragment:
        raise ValueError("empty local target")

    if target_no_fragment.startswith("/"):
        target_path = docs_root / target_no_fragment.lstrip("/")
    else:
        target_path = source_file.parent / target_no_fragment

    return target_path.resolve()


def _scan_markdown_targets(md_path: Path) -> list[LinkReference]:
    references: list[LinkReference] = []
    lines = md_path.read_text(encoding="utf-8").splitlines()

    in_fence = False
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        line_parts = line.split("`")
        for part_index, part in enumerate(line_parts):
            # Even indices are outside inline code spans.
            if part_index % 2 != 0:
                continue

            for match in MARKDOWN_LINK_PATTERN.finditer(part):
                references.append(
                    LinkReference(source_file=md_path, line_number=idx, raw_target=match.group(1))
                )

            for match in HTML_ATTR_PATTERN.finditer(part):
                references.append(
                    LinkReference(source_file=md_path, line_number=idx, raw_target=match.group(1))
                )

    return references


def validate_links(docs_root: Path) -> list[LinkIssue]:
    docs_root_resolved = docs_root.resolve()
    if not docs_root_resolved.exists():
        raise ValueError(f"docs root does not exist: {docs_root_resolved}")

    issues: list[LinkIssue] = []
    markdown_files = sorted(docs_root_resolved.rglob("*.md"))

    for md_path in markdown_files:
        references = _scan_markdown_targets(md_path)
        for reference in references:
            target_text = _trim_target(reference.raw_target)
            if not target_text or _is_external_target(target_text):
                continue

            try:
                resolved_target = _normalize_link_target(
                    reference.raw_target,
                    reference.source_file,
                    docs_root_resolved,
                )
            except ValueError as exc:
                issues.append(
                    LinkIssue(
                        source_file=reference.source_file,
                        line_number=reference.line_number,
                        raw_target=reference.raw_target,
                        message=str(exc),
                    )
                )
                continue

            try:
                resolved_target.relative_to(docs_root_resolved)
            except ValueError:
                issues.append(
                    LinkIssue(
                        source_file=reference.source_file,
                        line_number=reference.line_number,
                        raw_target=reference.raw_target,
                        message=f"target escapes docs root: {resolved_target}",
                    )
                )
                continue

            if not resolved_target.exists():
                issues.append(
                    LinkIssue(
                        source_file=reference.source_file,
                        line_number=reference.line_number,
                        raw_target=reference.raw_target,
                        message=f"target file does not exist: {resolved_target}",
                    )
                )

    return issues


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local links in docs markdown files")
    parser.add_argument("--docs-root", default="docs", help="Docs root directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    docs_root = Path(args.docs_root).resolve()

    try:
        issues = validate_links(docs_root)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    if issues:
        print("link validation failed:")
        for issue in issues:
            print(
                f"- {issue.source_file}:{issue.line_number}: {issue.raw_target} -> {issue.message}"
            )
        return 1

    print("link validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
