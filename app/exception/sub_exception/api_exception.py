import sys
from http import HTTPStatus

from app.exception.error_codes import ErrorCode
from app.exception.policy_application_exception import PolicyApplicationException


class APIException(PolicyApplicationException):
    """Base for all HTTP / API-layer failures."""
    error_code = ErrorCode.UNKNOWN_ERROR
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR


class BadRequestException(APIException):
    """Raised when the client sends a malformed or semantically invalid request."""
    error_code = ErrorCode.API_BAD_REQUEST
    http_status = HTTPStatus.BAD_REQUEST

    def __init__(self, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Bad request: {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"reason": reason, **(context or {})},
        )


class ValidationException(APIException):
    """Raised when Pydantic or business-rule validation fails."""
    error_code = ErrorCode.API_VALIDATION_ERROR
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY

    def __init__(self, field: str, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Validation error on field '{field}': {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"field": field, "reason": reason, **(context or {})},
        )


class UnauthorizedException(APIException):
    """Raised when a request lacks valid authentication credentials."""
    error_code = ErrorCode.API_UNAUTHORIZED
    http_status = HTTPStatus.UNAUTHORIZED

    def __init__(self, reason: str = "Missing or invalid credentials", error_details=sys, context: dict = None):
        super().__init__(
            error_message=reason,
            error_details=error_details,
            error_code=self.__class__.error_code,
            context=context or {},
        )


class ForbiddenException(APIException):
    """Raised when the authenticated user lacks permission for the resource."""
    error_code = ErrorCode.API_FORBIDDEN
    http_status = HTTPStatus.FORBIDDEN

    def __init__(self, resource: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Access to '{resource}' is forbidden",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"resource": resource, **(context or {})},
        )


class ResourceNotFoundException(APIException):
    """Raised when a REST resource does not exist (404)."""
    error_code = ErrorCode.API_NOT_FOUND
    http_status = HTTPStatus.NOT_FOUND

    def __init__(self, resource: str, identifier: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Resource '{resource}' with id '{identifier}' not found",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"resource": resource, "identifier": identifier, **(context or {})},
        )


class RateLimitException(APIException):
    """Raised when a client exceeds the configured request rate limit."""
    error_code = ErrorCode.API_RATE_LIMITED
    http_status = HTTPStatus.TOO_MANY_REQUESTS

    def __init__(self, limit: int, window_seconds: int, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Rate limit exceeded: {limit} requests per {window_seconds}s",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"limit": limit, "window_seconds": window_seconds, **(context or {})},
        )