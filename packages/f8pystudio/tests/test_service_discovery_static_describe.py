import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_STUDIO, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pystudio.service_catalog import ServiceCatalog  # noqa: E402
from f8pystudio.service_catalog.discovery import load_discovery_into_registries  # noqa: E402


class ServiceDiscoveryStaticDescribeTests(unittest.TestCase):
    def setUp(self) -> None:
        catalog = ServiceCatalog.instance()
        catalog.services.clear()
        catalog.operators.clear()

    def test_uses_describe_json_fast_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            service_dir = root / "f8" / "tests" / "svc"
            service_dir.mkdir(parents=True, exist_ok=True)

            (service_dir / "service.yml").write_text(
                "\n".join(
                    [
                        "schemaVersion: f8serviceEntry/1",
                        "serviceClass: f8.tests.svc",
                        "label: Tests Svc",
                        "version: 0.0.1",
                        "launch:",
                        "  command: python",
                        "  args: ['-c', 'print(123)']",
                        "  env: {}",
                        "  workdir: './'",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            (service_dir / "describe.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": "f8describe/1",
                        "service": {
                            "schemaVersion": "f8service/1",
                            "serviceClass": "f8.tests.svc",
                            "version": "0.0.1",
                            "label": "Tests Svc",
                            "rendererClass": "default_container",
                            "tags": [],
                            "stateFields": [],
                            "commands": [],
                            "dataInPorts": [],
                            "dataOutPorts": [],
                        },
                        "operators": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch("f8pystudio.service_catalog.discovery.subprocess.run") as run_mock:
                run_mock.side_effect = AssertionError("subprocess.run should not be called when describe.json exists")

                discovered = load_discovery_into_registries(roots=[root])

            self.assertIn("f8.tests.svc", discovered)

            catalog = ServiceCatalog.instance()
            spec = catalog.services.get("f8.tests.svc")
            self.assertIsNotNone(spec.launch)
            # Inherited from discovery entry and absolutized.
            if spec.launch is not None:
                self.assertEqual(str(Path(spec.launch.workdir).resolve()), str(service_dir.resolve()))


if __name__ == "__main__":
    unittest.main()

