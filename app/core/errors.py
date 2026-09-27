"""Application exception hierarchy and FastAPI error handling."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class MSNError(Exception):
    """Base class for all MSN AI application errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "msn_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_payload(self) -> dict:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
                "status": self.status_code,
            }
        }
        if self.details is not None:
            payload["error"]["details"] = self.details
        return payload


class ConfigurationError(MSNError):
    """Raised when application configuration is invalid."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "configuration_error"


class ProviderNotFoundError(MSNError):
    """Raised when an unknown model provider is requested."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "provider_not_found"


class ModelNotLoadedError(MSNError):
    """Raised when a model is required but not loaded."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_loaded"


class InferenceError(MSNError):
    """Raised when model inference fails."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "inference_error"


class SessionNotFoundError(MSNError):
    """Raised when a conversation session does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "session_not_found"


class ValidationError(MSNError):
    """Raised for request-level validation failures."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"


def _error_response(exc: MSNError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


def register_error_handlers(app: FastAPI) -> None:
    """Attach handlers that turn MSNError subclasses into JSON responses."""

    @app.exception_handler(MSNError)
    async def msn_error_handler(request: Request, exc: MSNError) -> JSONResponse:
        logger.error("%s %s -> %s: %s", request.method, request.url.path, exc.code, exc.message)
        return _error_response(exc)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled error on %s %s: %s", request.method, request.url.path, exc
        )
        return _error_response(
            ConfigurationError(
                f"An unexpected error occurred: {exc.__class__.__name__}",
                details=str(exc) if settings_debug() else None,
            )
        )


def settings_debug() -> bool:
    from app.core.config import settings

    return settings.debug