from __future__ import annotations

from f8pysdk import specs
from f8pysdk._specs import edit_policy, metadata, schema
from f8pysdk import generated


def test_specs_public_exports_match_source_modules() -> None:
    expected = sorted(
        {
            *edit_policy.__all__,
            *metadata.__all__,
            *schema.__all__,
            *generated.__all__,
            "operator_state_fields_with_builtins",
            "service_state_fields_with_builtins",
        }
    )

    assert sorted(specs.__all__) == expected
    assert all(hasattr(specs, name) for name in expected)
