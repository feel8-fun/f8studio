"""Migrate legacy provider image support into per-model capabilities.

Usage: pixi run python scripts/migrate_provider_settings.py path/to/providers.json ...
Writes each file in place; back it up before running.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def migrate_providers(providers: dict[str, Any]) -> None:
    for provider_id, config in providers.items():
        supported = config.pop("supportsImage", False)
        model = config["model"]
        models = list(dict.fromkeys(([model] if model else []) + config.get("models", [])))
        config["models"] = models
        capabilities = config.setdefault("modelCapabilities", [])
        if not supported or not model or not (provider_id.startswith("connection_") or provider_id == "systemone_local"):
            continue
        capability = next((item for item in capabilities if item["modelId"] == model), None)
        if capability is None:
            capability = {"modelId": model, "source": "legacy"}
            capabilities.append(capability)
        if capability.get("imageInput") is None:
            capability["imageInput"] = True
            capability["imageSource"] = "legacy"
            if capability.get("thinking") is not None:
                capability.setdefault("thinkingSource", capability.get("source", "catalog"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        value = json.loads(path.read_text())
        migrate_providers(value)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
