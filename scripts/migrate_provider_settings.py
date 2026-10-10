"""Migrate legacy provider image support into per-model capabilities.

Usage: pixi run python scripts/migrate_provider_settings.py path/to/providers.json ...
Writes each file in place; back it up before running.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from f8studio_server.agents.migrations import migrate_providers


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
