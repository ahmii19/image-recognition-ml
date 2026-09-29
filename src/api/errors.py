"""Structured Error Handling and Normalized Error Responses for FastAPI."""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger("api.errors")


class APIException(Exception):
    """Base API Exception for domain and validation errors."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id


class ImageValidationError(APIException):
    """Raised when uploaded file fails Phase 3 image validation."""

    def __init__(self, message: str, code: str = "INVALID_IMAGE", request_id: Optional[str] = None) -> None:
        super().__init__(
            status_code=400,
            code=code,
            message=message,
            request_id=request_id,
        )


class PayloadTooLargeError(APIException):
    """Raised when uploaded file exceeds maximum configured size."""

    def __init__(self, message: str, request_id: Optional[str] = None) -> None:
        super().__init__(
            status_code=413,
            code="FILE_TOO_LARGE",
            message=message,
            request_id=request_id,
        )


class InvalidModeError(APIException):
    """Raised when recognition mode is unsupported."""

    def __init__(self, message: str, request_id: Optional[str] = None) -> None:
        super().__init__(
            status_code=422,
            code="INVALID_MODE",
            message=message,
            request_id=request_id,
        )


class ModelUnavailableError(APIException):
    """Raised when ML model is not loaded or ready."""

    def __init__(self, message: str, request_id: Optional[str] = None) -> None:
        super().__init__(
            status_code=503,
            code="MODEL_UNAVAILABLE",
            message=message,
            request_id=request_id,
        )


def register_error_handlers(app: FastAPI) -> None:
    """Register custom exception handlers with FastAPI application."""

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
        req_id = getattr(request.state, "request_id", exc.request_id)
        logger.warning(f"[{req_id}] APIException: {exc.code} - {exc.message} (HTTP {exc.status_code})")
        payload = ErrorResponse(
            success=False,
            error=ErrorDetail(code=exc.code, message=exc.message, request_id=req_id),
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        # Format the validation error cleanly without leaking python internal traces
        first_error = exc.errors()[0] if exc.errors() else {}
        loc = " -> ".join([str(l) for l in first_error.get("loc", []) if l != "body"])
        msg = first_error.get("msg", "Invalid request parameters")
        formatted_message = f"Field '{loc}': {msg}" if loc else msg

        logger.warning(f"[{req_id}] RequestValidationError: {formatted_message}")
        payload = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="INVALID_REQUEST_PARAMETERS",
                message=formatted_message,
                request_id=req_id,
            ),
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump())

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "FILE_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            422: "UNPROCESSABLE_ENTITY",
            500: "INTERNAL_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }
        code = code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
        message = str(exc.detail) if exc.detail else "An HTTP error occurred."

        payload = ErrorResponse(
            success=False,
            error=ErrorDetail(code=code, message=message, request_id=req_id),
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        logger.error(f"[{req_id}] Unhandled Server Error: {str(exc)}", exc_info=True)
        payload = ErrorResponse(
            success=False,
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="An unexpected internal server error occurred while processing the request.",
                request_id=req_id,
            ),
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump())
