import os
import sys
import tempfile
import unittest
from pathlib import Path


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_STUDIO, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pystudio.service_catalog.discovery import (  # noqa: E402
    _platform_service_yml_names,
    find_service_dirs,
    load_service_entry,
)


class PlatformServiceYmlDiscoveryTests(unittest.TestCase):
    def test_prefers_platform_specific_over_service_yml(self) -> None:
        platform_name = _platform_service_yml_names()[0]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            service_dir = root / "f8" / "svc"
            service_dir.mkdir(parents=True, exist_ok=True)

            (service_dir / "service.yml").write_text(
                "\n".join(
                    [
                        "schemaVersion: f8serviceEntry/1",
                        "serviceClass: f8.tests.fallback",
                        "label: Fallback",
                        "version: 0.0.1",
                        "launch:",
                        "  command: python",
                        "  args: ['-c', 'print(1)']",
                        "  env: {}",
                        "  workdir: './'",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            (service_dir / platform_name).write_text(
                "\n".join(
                    [
                        "schemaVersion: f8serviceEntry/1",
                        "serviceClass: f8.tests.platform",
                        "label: Platform",
                        "version: 0.0.2",
                        "launch:",
                        "  command: python",
                        "  args: ['-c', 'print(2)']",
                        "  env: {}",
                        "  workdir: './'",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            entry = load_service_entry(service_dir)
            self.assertEqual(str(entry.serviceClass), "f8.tests.platform")

    def test_discovers_dir_with_only_platform_service_yml(self) -> None:
        platform_name = _platform_service_yml_names()[0]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            service_dir = root / "f8" / "svc"
            service_dir.mkdir(parents=True, exist_ok=True)

            (service_dir / platform_name).write_text(
                "\n".join(
                    [
                        "schemaVersion: f8serviceEntry/1",
                        "serviceClass: f8.tests.platformonly",
                        "label: PlatformOnly",
                        "version: 0.0.1",
                        "launch:",
                        "  command: python",
                        "  args: ['-c', 'print(3)']",
                        "  env: {}",
                        "  workdir: './'",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            dirs = find_service_dirs([root])
            self.assertIn(service_dir.resolve(), dirs)


if __name__ == "__main__":
    unittest.main()

