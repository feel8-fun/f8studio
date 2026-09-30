from __future__ import annotations

import pytest

from scripts.gen_cpp_protocol_models import Schema, _gen_header


def test_cpp_generator_rejects_unsupported_shape_instead_of_json_fallback() -> None:
    schemas = {"Model": Schema("Model", {"type": "object", "properties": {"value": {"type": "unsupported"}}})}
    with pytest.raises(ValueError, match="Unsupported typed schema"):
        _gen_header(schemas=schemas, namespace="test")


def test_cpp_generator_rejects_conflicting_enum_titles() -> None:
    schemas = {
        "Model": Schema(
            "Model",
            {
                "type": "object",
                "properties": {
                    "first": {"title": "Mode", "type": "string", "enum": ["a"]},
                    "second": {"title": "Mode", "type": "string", "enum": ["b"]},
                },
            },
        )
    }
    with pytest.raises(ValueError, match="Conflicting enum"):
        _gen_header(schemas=schemas, namespace="test")
