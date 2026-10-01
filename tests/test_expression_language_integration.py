from f8pyengine.operators._py_expr_eval import compile_expr
from f8pyscript.expr_validator import compile_pyexpr


def test_expression_languages_share_numpy_security_constraints() -> None:
    for expression in ("np.load('secret.npy')", "numpy.save('output.npy', [1])",
                       "np.ctypeslib.load_library('library', '.')", "np.lib.npyio", "np.load"):
        for compiler in (compile_expr, compile_pyexpr):
            code, error = compiler(expression, allow_numpy=True)
            assert code is None and error is not None
    for compiler in (compile_expr, compile_pyexpr):
        code, error = compiler('np.linalg.norm(np.array([3, 4]))', allow_numpy=True)
        assert code is not None and error is None
