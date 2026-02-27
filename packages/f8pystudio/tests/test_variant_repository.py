from __future__ import annotations

import json
from pathlib import Path

import pytest

from f8pystudio.variants.variant_models import F8NodeVariantRecord, F8VariantKind
from f8pystudio.variants.variant_repository import import_from_json, is_variant_name_conflict, list_variants_for_base, upsert_variant


def _make_variant_record(*, variant_id: str, base_node_type: str, name: str) -> F8NodeVariantRecord:
    now = F8NodeVariantRecord.now_iso()
    return F8NodeVariantRecord(
        variantId=variant_id,
        kind=F8VariantKind.operator,
        baseNodeType=base_node_type,
        serviceClass="svc.test",
        operatorClass="op.test",
        name=name,
        description="",
        tags=[],
        spec={"label": "x"},
        createdAt=now,
        updatedAt=now,
    )


def _patch_variants_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    target = tmp_path / "nodeVariants.json"
    monkeypatch.setattr("f8pystudio.variants.variant_repository.variants_file_path", lambda: target)
    return target


def test_variant_name_conflict_strip_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_variants_file(monkeypatch, tmp_path)
    upsert_variant(_make_variant_record(variant_id="v1", base_node_type="svc.a.op", name="foo"))

    assert is_variant_name_conflict("svc.a.op", " foo ") is True
    assert is_variant_name_conflict("svc.a.op", "FOO") is False


def test_upsert_rejects_duplicate_name_same_base(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_variants_file(monkeypatch, tmp_path)
    upsert_variant(_make_variant_record(variant_id="v1", base_node_type="svc.a.op", name="dup"))

    with pytest.raises(ValueError, match="already exists"):
        upsert_variant(_make_variant_record(variant_id="v2", base_node_type="svc.a.op", name=" dup "))


def test_upsert_allows_same_name_different_base(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_variants_file(monkeypatch, tmp_path)
    upsert_variant(_make_variant_record(variant_id="v1", base_node_type="svc.a.op", name="same"))
    upsert_variant(_make_variant_record(variant_id="v2", base_node_type="svc.b.op", name="same"))

    assert len(list_variants_for_base("svc.a.op")) == 1
    assert len(list_variants_for_base("svc.b.op")) == 1


def test_import_auto_renames_duplicates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_variants_file(monkeypatch, tmp_path)
    upsert_variant(_make_variant_record(variant_id="v-existing", base_node_type="svc.a.op", name="name"))

    import_path = tmp_path / "import.json"
    payload = {
        "schemaVersion": "f8variantlib/1",
        "variants": [
            _make_variant_record(variant_id="v2", base_node_type="svc.a.op", name="name").model_dump(mode="json"),
            _make_variant_record(variant_id="v3", base_node_type="svc.a.op", name=" name ").model_dump(mode="json"),
        ],
    }
    import_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import_from_json(str(import_path), mode="merge")
    names = [item.name for item in list_variants_for_base("svc.a.op")]

    assert "name" in names
    assert "name (2)" in names
    assert "name (3)" in names
