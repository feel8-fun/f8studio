from __future__ import annotations

# User-authored expressions and third-party numerical models can raise arbitrary
# Exception subclasses. Keep these broad boundaries named and logged at call sites.
OPERATOR_EVAL_ERRORS = (Exception,)
OPERATOR_MODEL_ERRORS = (Exception,)

OPERATOR_STATE_PUBLISH_ERRORS = (LookupError, OSError, RuntimeError, TypeError, ValueError)
OPERATOR_VALUE_COMPARE_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
