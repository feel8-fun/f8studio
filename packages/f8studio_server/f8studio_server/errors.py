from __future__ import annotations

from f8pysdk.specs import F8JsonValue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.responses import JSONResponse


class InvalidRequestError(ValueError):
    """An explicitly rejected domain input, safe to report to an API caller."""


class NotFoundError(FileNotFoundError):
    """A requested domain resource does not exist."""


class ConflictError(RuntimeError):
    """A lifecycle action conflicts with the current project or process state."""


class ServiceUnavailableError(RuntimeError):
    """A managed service could not become available."""


def api_error(status: int, code: str, message: str, *, detail: F8JsonValue = None) -> JSONResponse:
    from starlette.responses import JSONResponse
    # Keep detail during the API/1 transition for existing external clients.
    return JSONResponse(status_code=status, content={
        "code": code, "message": message, "detail": message if detail is None else detail,
    })
