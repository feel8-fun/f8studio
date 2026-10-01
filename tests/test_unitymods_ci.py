from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_unitymods_ci_module() -> object:
    script_path = Path("scripts/unitymods_ci.py").resolve()
    spec = importlib.util.spec_from_file_location("unitymods_ci", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load unitymods_ci module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UnityModsCiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_unitymods_ci_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.unitymods_root = self.root / "extensions" / "f8unitymods"
        self.unitymods_root.mkdir(parents=True)
        self.manifest_path = self.unitymods_root / "pixi.toml"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _patch_roots(self) -> tuple[mock._patch, mock._patch]:
        return (
            mock.patch.object(self.module, "UNITYMODS_ROOT", self.unitymods_root),
            mock.patch.object(self.module, "UNITYMODS_MANIFEST", self.manifest_path),
        )

    def _create_required_submodule_files(self) -> None:
        (self.unitymods_root / ".git").mkdir()
        self.manifest_path.write_text("[workspace]\nname = \"test\"\n", encoding="utf-8")
        package_init = self.unitymods_root / "f8unitymods_setup" / "__init__.py"
        package_init.parent.mkdir()
        package_init.write_text("", encoding="utf-8")
        project = (
            self.unitymods_root
            / "src"
            / "F8SkeletonStreamer"
            / "F8SkeletonStreamer.csproj"
        )
        project.parent.mkdir(parents=True)
        project.write_text("<Project />\n", encoding="utf-8")

    def test_validate_reports_commit_and_dirty_changes(self) -> None:
        self._create_required_submodule_files()

        def _git_output(command: str, *args: str) -> str:
            if command == "rev-parse":
                return "abc123"
            if command == "status":
                return " M src/Exporter.cs"
            raise AssertionError(f"unexpected git command: {(command, *args)}")

        root_patch, manifest_patch = self._patch_roots()
        with root_patch, manifest_patch, mock.patch.object(
            self.module, "_git_output", side_effect=_git_output
        ):
            result = self.module.validate(allow_dirty=True)

        self.assertEqual(result["commit"], "abc123")
        self.assertIs(result["dirty"], True)
        self.assertEqual(result["changes"], [" M src/Exporter.cs"])

    def test_validate_rejects_dirty_submodule_by_default(self) -> None:
        self._create_required_submodule_files()
        root_patch, manifest_patch = self._patch_roots()
        with root_patch, manifest_patch, mock.patch.object(
            self.module,
            "_git_output",
            side_effect=("abc123", " M src/Exporter.cs"),
        ):
            with self.assertRaisesRegex(RuntimeError, "uncommitted changes"):
                self.module.validate(allow_dirty=False)

    def test_bundle_copies_assets_and_writes_sha256_manifest(self) -> None:
        setup_wheel = self.unitymods_root / "dist" / "setup" / "f8unitymods_setup.whl"
        exporter_zip = self.unitymods_root / "dist" / "release" / "exporters.zip"
        release_manifest = self.unitymods_root / "dist" / "release" / "release-manifest.json"
        for path, content in (
            (setup_wheel, b"wheel"),
            (exporter_zip, b"zip"),
            (release_manifest, b"{}\n"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        output_dir = self.root / "bundle"
        root_patch, manifest_patch = self._patch_roots()
        with root_patch, manifest_patch, mock.patch.object(
            self.module,
            "validate",
            return_value={"commit": "abc123", "dirty": False},
        ), mock.patch.object(self.module, "_run_unitymods_task") as task_mock:
            result = self.module.bundle(
                output_dir=output_dir,
                build_assets=False,
                allow_dirty=False,
            )

        task_mock.assert_not_called()
        self.assertEqual(result["schemaVersion"], "f8unitymodsBundle/1")
        self.assertEqual(result["commit"], "abc123")
        self.assertEqual(len(result["assets"]), 3)
        manifest = json.loads(
            (output_dir / "f8unitymods-bundle.json").read_text(encoding="utf-8")
        )
        assets_by_name = {asset["name"]: asset for asset in manifest["assets"]}
        self.assertEqual(
            assets_by_name[setup_wheel.name]["sha256"],
            hashlib.sha256(b"wheel").hexdigest(),
        )
        self.assertEqual(
            assets_by_name[exporter_zip.name]["source"],
            "dist/release/exporters.zip",
        )

    def test_bundle_builds_assets_in_explicit_order(self) -> None:
        artifact = self.unitymods_root / "dist" / "setup" / "setup.whl"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"wheel")

        root_patch, manifest_patch = self._patch_roots()
        with root_patch, manifest_patch, mock.patch.object(
            self.module,
            "validate",
            return_value={"commit": "abc123", "dirty": True},
        ), mock.patch.object(self.module, "_run_unitymods_task") as task_mock:
            self.module.bundle(
                output_dir=self.root / "bundle",
                build_assets=True,
                allow_dirty=True,
            )

        self.assertEqual(
            task_mock.call_args_list,
            [mock.call("setup-wheel"), mock.call("release-package")],
        )

    def test_bundle_rejects_duplicate_release_asset_names(self) -> None:
        first = self.unitymods_root / "dist" / "setup" / "release-manifest.json"
        second = self.unitymods_root / "dist" / "release" / "release-manifest.json"
        for path in (first, second):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

        root_patch, manifest_patch = self._patch_roots()
        with root_patch, manifest_patch, mock.patch.object(
            self.module,
            "validate",
            return_value={"commit": "abc123", "dirty": False},
        ):
            with self.assertRaisesRegex(ValueError, "unique file names"):
                self.module.bundle(
                    output_dir=self.root / "bundle",
                    build_assets=False,
                    allow_dirty=False,
                )


if __name__ == "__main__":
    unittest.main()
