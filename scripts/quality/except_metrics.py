from __future__ import annotations

import argparse
import fnmatch
import re
from collections import Counter
from pathlib import Path


EXCEPT_EXCEPTION_RE = re.compile(r"^\s*except\s+Exception\b")
SILENT_STATEMENT_RE = re.compile(r"^\s*(pass|return|continue)\s*(#.*)?$")


def _matches_exclude(path: Path, *, root: Path, exclude_globs: tuple[str, ...]) -> bool:
    relative_path = path.relative_to(root).as_posix()
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in exclude_globs)


def iter_py_files(root: Path, *, exclude_globs: tuple[str, ...] = ()) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and not _matches_exclude(path, root=root, exclude_globs=exclude_globs)
    )


def count_metrics(file_path: Path) -> tuple[int, int]:
    total_except_exception = 0
    silent_except_exception = 0
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if not EXCEPT_EXCEPTION_RE.match(line):
            continue
        total_except_exception += 1

        trailing = line.split(":", 1)
        inline_statement = trailing[1].strip() if len(trailing) == 2 else ""
        if inline_statement and SILENT_STATEMENT_RE.match(inline_statement):
            silent_except_exception += 1
            continue
        if index + 1 < len(lines) and SILENT_STATEMENT_RE.match(lines[index + 1]):
            silent_except_exception += 1

    return total_except_exception, silent_except_exception


def main() -> int:
    parser = argparse.ArgumentParser(description="Count broad and silent exception usage in Python files.")
    parser.add_argument("root", type=Path, help="Root folder to scan")
    parser.add_argument("--fail-on-silent", action="store_true", help="Fail when silent broad catches exist")
    parser.add_argument("--exclude-glob", action="append", default=[], help="Relative glob to exclude")
    parser.add_argument("--max-broad", type=int, default=None, help="Maximum broad catch count")
    parser.add_argument("--max-silent", type=int, default=None, help="Maximum silent broad catch count")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"root path does not exist: {root}")

    per_file_total: Counter[str] = Counter()
    per_file_silent: Counter[str] = Counter()
    total_except = 0
    total_silent = 0
    exclude_globs = tuple(str(pattern).replace("\\", "/") for pattern in args.exclude_glob)
    for py_file in iter_py_files(root, exclude_globs=exclude_globs):
        broad_count, silent_count = count_metrics(py_file)
        if broad_count == 0 and silent_count == 0:
            continue
        relative_path = py_file.relative_to(root).as_posix()
        per_file_total[relative_path] = broad_count
        per_file_silent[relative_path] = silent_count
        total_except += broad_count
        total_silent += silent_count

    print(f"[except-metrics] root={root}")
    print(f"[except-metrics] except Exception count={total_except}")
    print(f"[except-metrics] silent except Exception count={total_silent}")
    if per_file_total:
        print("[except-metrics] top broad catches:")
        for relative_path, count in per_file_total.most_common(10):
            print(f"  {relative_path}: {count}")
    if per_file_silent:
        print("[except-metrics] top silent catches:")
        for relative_path, count in per_file_silent.most_common(10):
            if count > 0:
                print(f"  {relative_path}: {count}")

    failed = False
    if args.fail_on_silent and total_silent > 0:
        print("[except-metrics] failed: silent broad catches are not allowed")
        failed = True
    if args.max_broad is not None and total_except > args.max_broad:
        print(f"[except-metrics] failed: broad catches {total_except} > max {args.max_broad}")
        failed = True
    if args.max_silent is not None and total_silent > args.max_silent:
        print(f"[except-metrics] failed: silent catches {total_silent} > max {args.max_silent}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
