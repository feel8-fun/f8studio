from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    module_path = Path(relative_path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DocsValidatorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nav_module = _load_module("check_docs_nav", "scripts/check_docs_nav.py")
        self.links_module = _load_module("check_docs_links", "scripts/check_docs_links.py")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.docs_root = self.root / "docs"
        self.docs_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_nav_validator_passes_for_existing_targets(self) -> None:
        (self.docs_root / "index.md").write_text("# Home\n", encoding="utf-8")
        (self.docs_root / "guide.md").write_text("# Guide\n", encoding="utf-8")

        config_path = self.root / "mkdocs.yml"
        config_path.write_text(
            "nav:\n"
            "  - Home: index.md\n"
            "  - Guide: guide.md\n",
            encoding="utf-8",
        )

        issues = self.nav_module.validate_nav(config_path, self.docs_root)
        self.assertEqual(len(issues), 0)

    def test_nav_validator_reports_missing_target(self) -> None:
        (self.docs_root / "index.md").write_text("# Home\n", encoding="utf-8")

        config_path = self.root / "mkdocs.yml"
        config_path.write_text(
            "nav:\n"
            "  - Home: index.md\n"
            "  - Missing: missing.md\n",
            encoding="utf-8",
        )

        issues = self.nav_module.validate_nav(config_path, self.docs_root)
        self.assertEqual(len(issues), 1)
        self.assertIn("target file does not exist", issues[0].message)

    def test_link_validator_passes_for_valid_local_links(self) -> None:
        sub = self.docs_root / "sub"
        sub.mkdir(parents=True, exist_ok=True)

        (self.docs_root / "index.md").write_text("[Guide](sub/guide.md)\n", encoding="utf-8")
        (sub / "guide.md").write_text("![Diagram](../img.png)\n", encoding="utf-8")
        (self.docs_root / "img.png").write_bytes(b"png")

        issues = self.links_module.validate_links(self.docs_root)
        self.assertEqual(len(issues), 0)

    def test_link_validator_reports_missing_local_link(self) -> None:
        (self.docs_root / "index.md").write_text("[Missing](missing.md)\n", encoding="utf-8")

        issues = self.links_module.validate_links(self.docs_root)
        self.assertEqual(len(issues), 1)
        self.assertIn("target file does not exist", issues[0].message)


if __name__ == "__main__":
    unittest.main()
