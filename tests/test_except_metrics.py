from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCEPT_METRICS_PATH = REPO_ROOT / "scripts" / "quality" / "except_metrics.py"


def _load_except_metrics_module():
    spec = importlib.util.spec_from_file_location("f8_except_metrics", EXCEPT_METRICS_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


def test_except_metrics_counts_silent_and_broad_handlers(tmp_path: Path) -> None:
    module = _load_except_metrics_module()
    target = tmp_path / "sample.py"
    target.write_text(
        "\n".join(
            [
                "def first():",
                "    try:",
                "        risky()",
                "    except Exception:",
                "        pass",
                "",
                "def second():",
                "    try:",
                "        risky()",
                "    except Exception as exc:",
                "        raise RuntimeError(str(exc)) from exc",
            ]
        ),
        encoding="utf-8",
    )

    assert module.count_metrics(target) == (2, 1)


def test_except_metrics_respects_exclude_globs(tmp_path: Path) -> None:
    module = _load_except_metrics_module()
    included = tmp_path / "pkg" / "included.py"
    excluded = tmp_path / "pkg" / "tests" / "excluded.py"
    included.parent.mkdir(parents=True)
    excluded.parent.mkdir(parents=True)
    included.write_text("try:\n    risky()\nexcept Exception:\n    pass\n", encoding="utf-8")
    excluded.write_text("try:\n    risky()\nexcept Exception:\n    pass\n", encoding="utf-8")

    files = module.iter_py_files(tmp_path, exclude_globs=("**/tests/**",))

    assert files == [included]


def test_except_metrics_counts_tuples_aliases_and_bare_handlers(tmp_path: Path) -> None:
    module = _load_except_metrics_module()
    target = tmp_path / "sample.py"
    target.write_text('''ERRORS = (OSError, Exception)
try:
    risky()
except ERRORS:
    # A comment must not hide a silent catch.
    pass
try:
    risky()
except (ValueError, BaseException):
    logger.exception("failed")
    raise
try:
    risky()
except:
    pass
''')
    assert module.count_metrics(target) == (3, 2)


def test_typed_error_result_preserves_exception_for_boundary_reporting(tmp_path: Path) -> None:
    module = _load_except_metrics_module()
    target = tmp_path / "result.py"
    target.write_text("try:\n    evaluate()\nexcept Exception as exc:\n    return Result(error=exc)\n")
    assert module.count_metrics(target) == (1, 0)
