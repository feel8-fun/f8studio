from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _module_is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _missing_codegen_modules() -> list[str]:
    missing_modules: list[str] = []
    if not _module_is_available("msgspec"):
        missing_modules.append("msgspec")
    if not _module_is_available("datamodel_code_generator"):
        missing_modules.append("datamodel-code-generator")
    return missing_modules


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        project_root = Path(self.root).resolve()
        repo_root = project_root.parent.parent
        protocol_path = repo_root / "schemas" / "protocol.yml"
        output_path = project_root / "f8pysdk" / "generated" / "__init__.py"
        script_path = repo_root / "scripts" / "protocol_codegen_msgspec.py"

        if output_path.is_file():
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)

        missing_modules = _missing_codegen_modules()
        if missing_modules:
            missing_text = ", ".join(missing_modules)
            raise RuntimeError(
                "f8pysdk protocol model generation requires "
                f"{missing_text}, but they are not installed and {output_path} does not exist."
            )

        command = [
            sys.executable,
            str(script_path),
            "--protocol",
            str(protocol_path),
            "--output",
            str(output_path),
        ]
        subprocess.run(command, check=True, cwd=repo_root)
