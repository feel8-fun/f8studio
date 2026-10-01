from __future__ import annotations

import argparse
import ast
import fnmatch
from collections import Counter
from pathlib import Path

IGNORED_PARTS = frozenset({".git", ".pixi", ".sdk", "__pycache__", "build", "dist", "node_modules"})

def _matches_exclude(path: Path, *, root: Path, exclude_globs: tuple[str, ...]) -> bool:
    relative_path = path.relative_to(root).as_posix()
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in exclude_globs)


def iter_py_files(root: Path, *, exclude_globs: tuple[str, ...] = ()) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file()
        and not IGNORED_PARTS.intersection(path.relative_to(root).parts)
        and not _matches_exclude(path, root=root, exclude_globs=exclude_globs)
    )


def count_metrics(file_path: Path) -> tuple[int, int]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    # Resolve explicit module-level exception tuples used by runtime boundaries.
    aliases: dict[str, ast.expr] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = statement.value

    def broad(expression: ast.expr | None, seen: frozenset[str] = frozenset()) -> bool:
        if expression is None:
            return True
        if isinstance(expression, ast.Name):
            if expression.id in {"Exception", "BaseException"}:
                return True
            if expression.id in aliases and expression.id not in seen:
                return broad(aliases[expression.id], seen | {expression.id})
        if isinstance(expression, ast.Tuple):
            return any(broad(item, seen) for item in expression.elts)
        if isinstance(expression, ast.Attribute):
            return expression.attr in {"Exception", "BaseException"}
        return False

    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and broad(node.type)]
    def silent_handler(node: ast.ExceptHandler) -> bool:
        # Returning the exception in a typed error result transfers reporting to
        # the caller; it does not discard the exception or its traceback.
        for statement in node.body:
            if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Call):
                if any(keyword.arg == "error" and isinstance(keyword.value, ast.Name)
                       and keyword.value.id == node.name for keyword in statement.value.keywords):
                    return False
        return all(isinstance(statement, (ast.Pass, ast.Return, ast.Continue, ast.Break)) for statement in node.body)

    silent = sum(silent_handler(node) for node in handlers)
    return len(handlers), silent


def main() -> int:
    parser = argparse.ArgumentParser(description="Count broad and silent exception usage in Python files.")
    parser.add_argument("roots", nargs="+", type=Path, help="Root folders to scan")
    parser.add_argument("--fail-on-silent", action="store_true", help="Fail when silent broad catches exist")
    parser.add_argument("--exclude-glob", action="append", default=[], help="Relative glob to exclude")
    parser.add_argument("--max-broad", type=int, default=None, help="Maximum broad catch count")
    parser.add_argument("--max-silent", type=int, default=None, help="Maximum silent broad catch count")
    args = parser.parse_args()

    roots = [root.resolve() for root in args.roots]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"root folder does not exist: {root}")

    per_file_total: Counter[str] = Counter()
    per_file_silent: Counter[str] = Counter()
    total_except = 0
    total_silent = 0
    exclude_globs = tuple(str(pattern).replace("\\", "/") for pattern in args.exclude_glob)
    files = {path for root in roots for path in iter_py_files(root, exclude_globs=exclude_globs)}
    for py_file in sorted(files):
        broad_count, silent_count = count_metrics(py_file)
        if broad_count == 0 and silent_count == 0:
            continue
        root = next(root for root in roots if py_file.is_relative_to(root))
        relative_path = f"{root.name}/{py_file.relative_to(root).as_posix()}"
        per_file_total[relative_path] = broad_count
        per_file_silent[relative_path] = silent_count
        total_except += broad_count
        total_silent += silent_count

    print(f"[except-metrics] roots={', '.join(str(root) for root in roots)}")
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
