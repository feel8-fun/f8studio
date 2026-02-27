from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from .variant_events import emit_variants_changed
from .variant_models import F8NodeVariantLibraryFile, F8NodeVariantRecord

logger = logging.getLogger(__name__)


def variants_file_path() -> Path:
    return Path.home() / ".f8" / "studio" / "nodeVariants.json"


def normalize_variant_name(name: str) -> str:
    return str(name or "").strip()


def _records_name_conflict(
    records: list[F8NodeVariantRecord],
    *,
    base_node_type: str,
    name: str,
    exclude_variant_id: str | None = None,
) -> bool:
    base = str(base_node_type or "").strip()
    target = normalize_variant_name(name)
    exclude_id = str(exclude_variant_id or "").strip()
    if not base or not target:
        return False
    for variant in records:
        if str(variant.baseNodeType or "").strip() != base:
            continue
        if exclude_id and str(variant.variantId or "").strip() == exclude_id:
            continue
        if normalize_variant_name(variant.name) == target:
            return True
    return False


def is_variant_name_conflict(base_node_type: str, name: str, *, exclude_variant_id: str | None = None) -> bool:
    lib = load_library()
    return _records_name_conflict(
        list(lib.variants),
        base_node_type=base_node_type,
        name=name,
        exclude_variant_id=exclude_variant_id,
    )


def ensure_unique_variant_name(
    base_node_type: str,
    desired_name: str,
    *,
    exclude_variant_id: str | None = None,
    existing_records: list[F8NodeVariantRecord] | None = None,
) -> str:
    base_name = normalize_variant_name(desired_name) or "Variant"
    records = list(existing_records) if existing_records is not None else list(load_library().variants)

    if not _records_name_conflict(
        records,
        base_node_type=base_node_type,
        name=base_name,
        exclude_variant_id=exclude_variant_id,
    ):
        return base_name

    suffix = 2
    while True:
        candidate = f"{base_name} ({suffix})"
        if not _records_name_conflict(
            records,
            base_node_type=base_node_type,
            name=candidate,
            exclude_variant_id=exclude_variant_id,
        ):
            return candidate
        suffix += 1


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
    if _records_name_conflict(
        list(lib.variants),
        base_node_type=record.baseNodeType,
        name=record.name,
        exclude_variant_id=record.variantId,
    ):
        normalized = normalize_variant_name(record.name)
        raise ValueError(
            f'Variant name "{normalized}" already exists for base node type "{record.baseNodeType}".'
        )

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
    emit_variants_changed()
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
        emit_variants_changed()
    return changed


def import_from_json(path: str, mode: Literal["merge", "replace"] = "merge") -> F8NodeVariantLibraryFile:
    in_path = Path(str(path or "").strip())
    if not in_path.is_file():
        raise FileNotFoundError(f"Variants file not found: {in_path}")
    raw = json.loads(in_path.read_text(encoding="utf-8"))
    imported = F8NodeVariantLibraryFile.model_validate(raw)

    if mode == "replace":
        current = F8NodeVariantLibraryFile(schemaVersion=imported.schemaVersion, variants=[])
    else:
        current = load_library()

    target_variants: list[F8NodeVariantRecord] = list(current.variants)
    for variant in imported.variants:
        variant_id = str(variant.variantId or "").strip()
        target_variants = [item for item in target_variants if str(item.variantId or "").strip() != variant_id]

        unique_name = ensure_unique_variant_name(
            variant.baseNodeType,
            variant.name,
            existing_records=target_variants,
        )
        if unique_name != variant.name:
            logger.info(
                "Renamed imported variant name for base=%s variantId=%s from %r to %r",
                variant.baseNodeType,
                variant.variantId,
                variant.name,
                unique_name,
            )
            target_variants.append(variant.model_copy(update={"name": unique_name}))
        else:
            target_variants.append(variant)

    current.variants = target_variants
    save_library(current)
    emit_variants_changed()
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
