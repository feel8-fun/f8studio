from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from .variant_models import F8NodeVariantLibraryFile, F8NodeVariantRecord

logger = logging.getLogger(__name__)


def variants_file_path() -> Path:
    return Path.home() / ".f8" / "studio" / "nodeVariants.json"


def load_library() -> F8NodeVariantLibraryFile:
    path = variants_file_path()
    if not path.is_file():
        return F8NodeVariantLibraryFile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return F8NodeVariantLibraryFile.model_validate(data)
    except Exception:
        logger.exception("Failed to load variants library from %s", path)
        return F8NodeVariantLibraryFile()


def save_library(file_model: F8NodeVariantLibraryFile) -> None:
    path = variants_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = file_model.model_dump(mode="json")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def list_variants_for_base(base_node_type: str) -> list[F8NodeVariantRecord]:
    base = str(base_node_type or "").strip()
    if not base:
        return []
    lib = load_library()
    out = [v for v in lib.variants if str(v.baseNodeType or "").strip() == base]
    return sorted(out, key=lambda v: (str(v.name or "").lower(), str(v.variantId or "")))


def upsert_variant(record: F8NodeVariantRecord) -> F8NodeVariantRecord:
    lib = load_library()
    found = False
    out: list[F8NodeVariantRecord] = []
    for v in lib.variants:
        if str(v.variantId) == str(record.variantId):
            found = True
            out.append(record)
        else:
            out.append(v)
    if not found:
        out.append(record)
    lib.variants = out
    save_library(lib)
    return record


def delete_variant(variant_id: str) -> bool:
    vid = str(variant_id or "").strip()
    if not vid:
        return False
    lib = load_library()
    before = len(lib.variants)
    lib.variants = [v for v in lib.variants if str(v.variantId) != vid]
    changed = len(lib.variants) != before
    if changed:
        save_library(lib)
    return changed


def import_from_json(path: str, mode: Literal["merge", "replace"] = "merge") -> F8NodeVariantLibraryFile:
    in_path = Path(str(path or "").strip())
    if not in_path.is_file():
        raise FileNotFoundError(f"Variants file not found: {in_path}")
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    imported = F8NodeVariantLibraryFile.model_validate(raw)

    if mode == "replace":
        save_library(imported)
        return imported

    current = load_library()
    merged: dict[str, F8NodeVariantRecord] = {str(v.variantId): v for v in current.variants}
    for v in imported.variants:
        merged[str(v.variantId)] = v
    current.variants = list(merged.values())
    save_library(current)
    return current


def export_to_json(path: str) -> Path:
    out_path = Path(str(path or "").strip())
    if not str(out_path):
        raise ValueError("Export path is empty")
    if out_path.suffix.lower() != ".json":
        out_path = out_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lib = load_library()
    out_path.write_text(
        json.dumps(lib.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path
