from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


EXCEPT_EXCEPTION_RE = re.compile(r"^\s*except\s+Exception\b")
SILENT_STATEMENT_RE = re.compile(r"^\s*(pass|return|continue)\s*(#.*)?$")


def iter_py_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def count_metrics(file_path: Path) -> tuple[int, int]:
    total_except_exception = 0
    silent_except_exception = 0

    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if not EXCEPT_EXCEPTION_RE.match(line):
            continue
        total_except_exception += 1

        trailing = line.split(":", 1)
        inline_stmt = trailing[1].strip() if len(trailing) == 2 else ""
        if inline_stmt and SILENT_STATEMENT_RE.match(inline_stmt):
            silent_except_exception += 1
            continue

        if index + 1 >= len(lines):
            continue
        next_line = lines[index + 1]
        if SILENT_STATEMENT_RE.match(next_line):
            silent_except_exception += 1

    return total_except_exception, silent_except_exception


def main() -> int:
    parser = argparse.ArgumentParser(description="Count broad/silent exception usage in Python files.")
    parser.add_argument("root", type=Path, help="Root folder to scan")
    parser.add_argument("--fail-on-silent", action="store_true", help="Exit with non-zero when silent catches exist")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"root path does not exist: {root}")

    per_file_total = Counter()
    per_file_silent = Counter()
    total_except = 0
    total_silent = 0

    for py_file in iter_py_files(root):
        count_except, count_silent = count_metrics(py_file)
        if count_except == 0 and count_silent == 0:
            continue
        rel = py_file.relative_to(root).as_posix()
        per_file_total[rel] = count_except
        per_file_silent[rel] = count_silent
        total_except += count_except
        total_silent += count_silent

    print(f"[except-metrics] root={root}")
    print(f"[except-metrics] except Exception count={total_except}")
    print(f"[except-metrics] silent except Exception count={total_silent}")

    if per_file_total:
        print("[except-metrics] top broad catches:")
        for rel, count in per_file_total.most_common(10):
            print(f"  {rel}: {count}")

    if per_file_silent:
        print("[except-metrics] top silent catches:")
        for rel, count in per_file_silent.most_common(10):
            if count > 0:
                print(f"  {rel}: {count}")

    if args.fail_on_silent and total_silent > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
