from f8pyscript.expr_validator import PyExprValidator, compile_pyexpr


def test_validator_rejects_disallowed_calls() -> None:
    compiled, error = compile_pyexpr("__import__('os')", allow_numpy=False)

    assert compiled is None
    assert error == "call not allowed: __import__"


def test_validator_rejects_private_attributes() -> None:
    tree, error = PyExprValidator(allow_numpy=False).validate("inputs.msg.__class__")

    assert tree is None
    assert error == "private/dunder attribute access is not allowed"


def test_validator_respects_numpy_toggle() -> None:
    disabled_tree, disabled_error = PyExprValidator(allow_numpy=False).validate("np.clip(x, 0, 1)")
    enabled_tree, enabled_error = PyExprValidator(allow_numpy=True).validate("np.clip(x, 0, 1)")

    assert disabled_tree is None
    assert disabled_error == "numpy calls are disabled"
    assert enabled_tree is not None
    assert enabled_error is None


def test_numpy_io_and_native_library_access_are_rejected_by_both_expression_languages() -> None:
    from f8pyengine.operators._py_expr_eval import compile_expr

    for expression in ("np.load('secret.npy')", "numpy.save('output.npy', [1])",
                       "np.ctypeslib.load_library('library', '.')", "np.lib.npyio", "np.load"):
        for compiler in (compile_expr, compile_pyexpr):
            code, error = compiler(expression, allow_numpy=True)
            assert code is None and error is not None
    for compiler in (compile_expr, compile_pyexpr):
        code, error = compiler('np.linalg.norm(np.array([3, 4]))', allow_numpy=True)
        assert code is not None and error is None
