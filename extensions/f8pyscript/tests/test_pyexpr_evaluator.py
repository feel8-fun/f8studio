from f8pyscript.expr_evaluator import PyExprEvaluator
from f8pyscript.expr_json_ref import wrap_pyexpr_value
from f8pyscript.expr_validator import compile_pyexpr


def test_evaluator_returns_value_and_unwraps_json_refs() -> None:
    compiled, error = compile_pyexpr("inputs['msg'].user.name", allow_numpy=False)
    assert compiled is not None
    assert error is None

    result = PyExprEvaluator().evaluate(
        compiled,
        names={"inputs": {"msg": wrap_pyexpr_value({"user": {"name": "Ada"}})}},
        allow_numpy=False,
    )

    assert result.error is None
    assert result.value == "Ada"


def test_evaluator_returns_exception_without_raising() -> None:
    compiled, error = compile_pyexpr("1 / 0", allow_numpy=False)
    assert compiled is not None
    assert error is None

    result = PyExprEvaluator().evaluate(compiled, names={}, allow_numpy=False)

    assert result.value is None
    assert isinstance(result.error, ZeroDivisionError)
