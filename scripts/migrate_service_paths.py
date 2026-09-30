"""Migrate known legacy model-directory settings in exported Studio documents.

Writes a separate output file; never changes the source or a Studio database.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def migrate_document(document: dict[str, Any]) -> int:
    if document.get("schemaVersion") != "f8studio-document/2":
        raise ValueError("Expected a f8studio-document/2 JSON export")
    count = 0
    for node in document.get("nodes", []):
        service_class = node.get("serviceClass", "")
        values = node.get("stateValues", {})
        if service_class.startswith("f8.dl."):
            field, old_default = "weightsDir", "services/f8/dl/weights"
        elif service_class == "f8.cvkit.tracking":
            field, old_default = "modelDir", "models"
        else:
            continue
        value = values.get(field)
        if isinstance(value, str) and value.replace("\\", "/").removeprefix("./").rstrip("/") == old_default:
            values[field] = ""
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    document = json.loads(args.source.read_text(encoding="utf-8"))
    count = migrate_document(document)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(f"Migrated {count} model directory settings to installation defaults")


if __name__ == "__main__":
    main()
