from __future__ import annotations

# User-authored expressions and Studio runtime/UI integration points can raise
# arbitrary ordinary exceptions. Keep broad boundaries named and logged at call sites.
OPERATOR_EVAL_ERRORS = (Exception,)
OPERATOR_PULL_ERRORS = (Exception,)
OPERATOR_STATE_READ_ERRORS = (LookupError, OSError, RuntimeError, TypeError, ValueError)
OPERATOR_STATE_PUBLISH_ERRORS = (LookupError, OSError, RuntimeError, TypeError, ValueError)
OPERATOR_VALUE_COMPARE_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
