from __future__ import annotations

from pathlib import Path

from scripts.web_studio.generate_contracts import generate, ts_type


def test_generated_contracts_are_current_and_facade_has_no_wire_definitions() -> None:
    for path, content in generate().items():
        assert path.read_text() == content, f"Regenerate {path}"
    facade = Path(__file__).resolve().parents[2] / "f8studio_web/src/api/contracts.ts"
    assert "export interface " not in facade.read_text()


def test_tuple_generation_preserves_each_element_type() -> None:
    assert (
        ts_type(
            {"type": "array", "minItems": 2, "maxItems": 2, "prefixItems": [{"type": "string"}, {"type": "integer"}]}
        )
        == "readonly [string, number]"
    )
    assert (
        ts_type({"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}})
        == "readonly [number, number, number]"
    )
