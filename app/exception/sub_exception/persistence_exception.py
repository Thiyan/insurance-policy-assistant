import sys
from http import HTTPStatus

from app.exception.error_codes import ErrorCode
from app.exception.policy_application_exception import PolicyApplicationException


class PersistenceException(PolicyApplicationException):
    """Base for all DB / storage layer failures."""
    error_code = ErrorCode.UNKNOWN_ERROR
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR


class RecordNotFoundException(PersistenceException):
    """Raised when a DB lookup by ID or fingerprint returns nothing."""
    error_code = ErrorCode.PERSIST_NOT_FOUND
    http_status = HTTPStatus.NOT_FOUND

    def __init__(self, entity: str, identifier: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"{entity} with identifier '{identifier}' was not found",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"entity": entity, "identifier": identifier, **(context or {})},
        )


class RecordConflictException(PersistenceException):
    """Raised on unique-constraint violations (duplicate inserts)."""
    error_code = ErrorCode.PERSIST_CONFLICT
    http_status = HTTPStatus.CONFLICT

    def __init__(self, entity: str, field: str, value: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"{entity} with {field}='{value}' already exists",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"entity": entity, "field": field, "value": value, **(context or {})},
        )


class DatabaseWriteException(PersistenceException):
    """Raised when an INSERT / UPDATE / DELETE fails."""
    error_code = ErrorCode.PERSIST_WRITE_FAILED
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, operation: str, entity: str, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"DB {operation} on '{entity}' failed: {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"operation": operation, "entity": entity, "reason": reason, **(context or {})},
        )


class DatabaseReadException(PersistenceException):
    """Raised when a SELECT query fails unexpectedly."""
    error_code = ErrorCode.PERSIST_READ_FAILED
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, entity: str, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"DB read on '{entity}' failed: {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"entity": entity, "reason": reason, **(context or {})},
        )


class DatabaseConnectionException(PersistenceException):
    """Raised when the DB connection pool is unavailable or times out."""
    error_code = ErrorCode.PERSIST_CONNECTION_LOST
    http_status = HTTPStatus.SERVICE_UNAVAILABLE

    def __init__(self, db_url_hint: str, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Cannot connect to database '{db_url_hint}': {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"db_url_hint": db_url_hint, "reason": reason, **(context or {})},
        )