import pytest

from f8pyscript.expr_json_ref import PyExprJsonRef, wrap_pyexpr_value


def test_json_ref_supports_attribute_index_iteration_and_unwrap() -> None:
    value = wrap_pyexpr_value({"user": {"name": "Ada"}, "items": [1, {"x": 2}]})

    assert isinstance(value, PyExprJsonRef)
    assert value.user.name == "Ada"
    assert value["items"][1].x == 2
    assert list(value) == ["user", "items"]
    assert value.unwrap() == {"user": {"name": "Ada"}, "items": [1, {"x": 2}]}


def test_json_ref_blocks_private_attribute_and_key_access() -> None:
    value = wrap_pyexpr_value({"_secret": 1})

    with pytest.raises(AttributeError):
        value._secret

    with pytest.raises(KeyError):
        value["_secret"]
