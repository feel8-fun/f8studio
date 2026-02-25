from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


def _load_generator_module() -> object:
    script_path = Path("scripts/generate_service_docs.py").resolve()
    spec = importlib.util.spec_from_file_location("generate_service_docs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load generate_service_docs module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerateServiceDocsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_generator_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.services_root = self.root / "services"
        self.output_root = self.root / "docs" / "modules" / "services"
        self.manual_root = self.root / "docs" / "modules" / "manual"
        self.index_path = self.root / "docs" / "modules" / "index.md"

        self.services_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manual_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_service_entry(
        self,
        *,
        service_dir: Path,
        service_class: str,
        label: str,
        describe_payload: dict[str, object],
    ) -> None:
        service_dir.mkdir(parents=True, exist_ok=True)
        service_yml = {
            "schemaVersion": "f8serviceEntry/1",
            "serviceClass": service_class,
            "label": label,
            "version": "0.0.1",
            "launch": {
                "command": "pixi",
                "args": ["run", "-e", "default", "service_task"],
                "env": {},
                "workdir": "../../..",
            },
        }
        (service_dir / "service.yml").write_text(yaml.safe_dump(service_yml), encoding="utf-8")
        (service_dir / "describe.json").write_text(json.dumps(describe_payload), encoding="utf-8")

    def _write_manual(self, service_class: str, text: str) -> None:
        slug = service_class.replace(".", "-")
        (self.manual_root / f"{slug}.md").write_text(text, encoding="utf-8")

    def test_generates_service_page_for_non_operator_service(self) -> None:
        service_class = "f8.screencap"
        describe_payload = {
            "service": {
                "serviceClass": service_class,
                "label": "Screen Capture",
                "description": "Capture desktop frames.",
                "tags": ["capture"],
                "stateFields": [
                    {
                        "name": "active",
                        "label": "Active",
                        "description": "Service active flag",
                        "access": "rw",
                        "required": False,
                        "showOnNode": True,
                        "valueSchema": {"type": "boolean", "default": True},
                    }
                ],
                "commands": [],
                "dataInPorts": [],
                "dataOutPorts": [
                    {
                        "name": "frame",
                        "description": "Frame payload",
                        "required": True,
                        "showOnNode": True,
                        "valueSchema": {"type": "object"},
                    }
                ],
            },
            "operators": [],
        }
        service_dir = self.services_root / "f8" / "screencap"
        self._write_service_entry(
            service_dir=service_dir,
            service_class=service_class,
            label="Screen Capture",
            describe_payload=describe_payload,
        )
        self._write_manual(service_class, "### Recommended Use Cases\n\nUse for live desktop capture.")

        result = self.module.build_docs(
            services_root=self.services_root,
            output_root=self.output_root,
            manual_root=self.manual_root,
            index_path=self.index_path,
            check=False,
        )

        self.assertEqual(result, 0)
        service_page = self.output_root / "f8-screencap.md"
        self.assertTrue(service_page.exists())
        text = service_page.read_text(encoding="utf-8")
        self.assertIn("# Screen Capture (`f8.screencap`)", text)
        self.assertIn("## Service State Fields", text)
        self.assertIn("## Operators", text)
        self.assertIn("_None_", text)
        self.assertIn("## Usage Guide (Manual)", text)

    def test_generates_operator_section_for_pyengine_like_service(self) -> None:
        service_class = "f8.pyengine"
        describe_payload = {
            "service": {
                "serviceClass": service_class,
                "label": "PyEngine",
                "description": "Runtime engine",
                "tags": ["engine"],
                "stateFields": [],
                "commands": [],
                "dataInPorts": [],
                "dataOutPorts": [],
            },
            "operators": [
                {
                    "operatorClass": "f8.tick",
                    "label": "Tick",
                    "description": "Clock pulse node",
                    "execInPorts": [],
                    "execOutPorts": ["exec"],
                    "stateFields": [
                        {
                            "name": "intervalMs",
                            "label": "Interval",
                            "description": "Tick interval",
                            "access": "rw",
                            "required": False,
                            "showOnNode": True,
                            "valueSchema": {"type": "integer", "default": 16},
                        }
                    ],
                    "dataInPorts": [],
                    "dataOutPorts": [
                        {
                            "name": "processingMs",
                            "description": "Processing duration",
                            "required": True,
                            "showOnNode": True,
                            "valueSchema": {"type": "integer"},
                        }
                    ],
                }
            ],
        }
        service_dir = self.services_root / "f8" / "engine"
        self._write_service_entry(
            service_dir=service_dir,
            service_class=service_class,
            label="PyEngine",
            describe_payload=describe_payload,
        )
        self._write_manual(service_class, "### Operator Composition Notes\n\nCompose small operator chains.")

        result = self.module.build_docs(
            services_root=self.services_root,
            output_root=self.output_root,
            manual_root=self.manual_root,
            index_path=self.index_path,
            check=False,
        )

        self.assertEqual(result, 0)
        service_page = self.output_root / "f8-pyengine.md"
        text = service_page.read_text(encoding="utf-8")
        self.assertIn("### Tick (`f8.tick`)", text)
        self.assertIn("Exec out ports: `exec`", text)
        self.assertIn("`intervalMs`", text)
        self.assertIn("`processingMs`", text)

    def test_fails_when_describe_missing(self) -> None:
        service_class = "f8.example"
        service_dir = self.services_root / "f8" / "example"
        service_dir.mkdir(parents=True, exist_ok=True)

        service_yml = {
            "schemaVersion": "f8serviceEntry/1",
            "serviceClass": service_class,
            "label": "Example",
            "version": "0.0.1",
            "launch": {
                "command": "pixi",
                "args": ["run"],
                "env": {},
                "workdir": ".",
            },
        }
        (service_dir / "service.yml").write_text(yaml.safe_dump(service_yml), encoding="utf-8")
        self._write_manual(service_class, "### Troubleshooting\n\nNone.")

        with self.assertRaises(ValueError) as ctx:
            self.module.build_docs(
                services_root=self.services_root,
                output_root=self.output_root,
                manual_root=self.manual_root,
                index_path=self.index_path,
                check=False,
            )

        self.assertIn("missing describe.json", str(ctx.exception))

    def test_fails_when_describe_json_invalid(self) -> None:
        service_class = "f8.invalidjson"
        service_dir = self.services_root / "f8" / "invalidjson"
        service_dir.mkdir(parents=True, exist_ok=True)

        service_yml = {
            "schemaVersion": "f8serviceEntry/1",
            "serviceClass": service_class,
            "label": "Invalid JSON",
            "version": "0.0.1",
            "launch": {
                "command": "pixi",
                "args": ["run"],
                "env": {},
                "workdir": ".",
            },
        }
        (service_dir / "service.yml").write_text(yaml.safe_dump(service_yml), encoding="utf-8")
        (service_dir / "describe.json").write_text("{broken json", encoding="utf-8")
        self._write_manual(service_class, "### Troubleshooting\n\nNone.")

        with self.assertRaises(ValueError) as ctx:
            self.module.build_docs(
                services_root=self.services_root,
                output_root=self.output_root,
                manual_root=self.manual_root,
                index_path=self.index_path,
                check=False,
            )

        self.assertIn("invalid JSON", str(ctx.exception))

    def test_fails_when_required_service_field_missing(self) -> None:
        service_class = "f8.missingfield"
        describe_payload = {
            "service": {
                "serviceClass": service_class,
                "description": "Missing label field on purpose",
                "tags": [],
                "stateFields": [],
                "commands": [],
                "dataInPorts": [],
                "dataOutPorts": [],
            },
            "operators": [],
        }
        service_dir = self.services_root / "f8" / "missingfield"
        self._write_service_entry(
            service_dir=service_dir,
            service_class=service_class,
            label="Missing Field",
            describe_payload=describe_payload,
        )
        self._write_manual(service_class, "### Troubleshooting\n\nNone.")

        with self.assertRaises(ValueError) as ctx:
            self.module.build_docs(
                services_root=self.services_root,
                output_root=self.output_root,
                manual_root=self.manual_root,
                index_path=self.index_path,
                check=False,
            )

        self.assertIn("missing or invalid 'label'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
