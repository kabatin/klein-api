"""One error envelope for every non-2xx response.

Shape: ``{"error": {code, message, retryable, details}}``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("klein_api.api")


class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool
    details: list[dict[str, Any]] | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


_HTTP_CODES = {
    400: ("bad_request", False),
    401: ("unauthorized", False),
    403: ("forbidden", False),
    404: ("not_found", False),
    405: ("method_not_allowed", False),
    409: ("conflict", False),
    411: ("length_required", False),
    413: ("payload_too_large", False),
    422: ("invalid_params", False),
    429: ("busy", True),
    503: ("unavailable", True),
}


def error_response(
    status: int,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(
        error=ErrorBody(code=code, message=message, retryable=retryable, details=details)
    )
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


BUSY_MESSAGE = "the GPU slot and the waiting queue are full"


def busy_response(retry_after_s: int) -> JSONResponse:
    return error_response(
        429, "busy", BUSY_MESSAGE, retryable=True, headers={"Retry-After": str(retry_after_s)}
    )


def validation_details(errors: list[Any]) -> list[dict[str, Any]]:
    """Location and message only — never echo inputs (they may be megabytes of base64)."""
    return [
        {"loc": [str(part) for part in err.get("loc", ())], "msg": str(err.get("msg", ""))}
        for err in errors
    ]


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            422,
            "invalid_params",
            "params failed validation",
            details=validation_details(list(exc.errors())),
        )

    @app.exception_handler(ValidationError)
    async def _validation(_: Request, exc: ValidationError) -> JSONResponse:
        return error_response(
            422,
            "invalid_params",
            "params failed validation",
            details=validation_details(list(exc.errors())),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code, retryable = _HTTP_CODES.get(exc.status_code, ("error", exc.status_code >= 500))
        return error_response(exc.status_code, code, str(exc.detail), retryable=retryable)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", exc_info=exc)
        return error_response(500, "internal", "internal server error", retryable=True)
