import sys
from http import HTTPStatus

from app.exception.error_codes import ErrorCode
from app.exception.policy_application_exception import PolicyApplicationException


class IngestionException(PolicyApplicationException):
    """Base for all ingestion-pipeline failures."""
    error_code  = ErrorCode.UNKNOWN_ERROR
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY


class FileNotFoundException(IngestionException):
    """Raised when the source file does not exist on disk or in remote storage."""
    error_code = ErrorCode.INGEST_FILE_NOT_FOUND
    http_status = HTTPStatus.NOT_FOUND

    def __init__(self, file_path: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"File not found: '{file_path}'",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"file_path": file_path, **(context or {})},
        )


class UnsupportedFileTypeException(IngestionException):
    """Raised when the file extension has no registered loader."""
    error_code = ErrorCode.INGEST_UNSUPPORTED_TYPE
    http_status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE

    def __init__(self, file_type: str, supported: list[str], error_details=sys, context: dict = None):
        super().__init__(
            error_message=(
                f"File type '{file_type}' is not supported. "
                f"Supported types: {supported}"
            ),
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"file_type": file_type, "supported": supported, **(context or {})},
        )


class DocumentParseException(IngestionException):
    """Raised when a loader fails to parse document content."""
    error_code = ErrorCode.INGEST_PARSE_FAILED
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY

    def __init__(self, file_name: str, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Failed to parse '{file_name}': {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"file_name": file_name, "reason": reason, **(context or {})},
        )


class ChunkingException(IngestionException):
    """Raised when the text splitter fails to chunk a document."""
    error_code = ErrorCode.INGEST_CHUNK_FAILED
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY

    def __init__(self, file_name: str, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Chunking failed for '{file_name}': {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"file_name": file_name, "reason": reason, **(context or {})},
        )


class EmbeddingException(IngestionException):
    """Raised when the embedding model fails to encode a chunk."""
    error_code = ErrorCode.INGEST_EMBED_FAILED
    http_status = HTTPStatus.BAD_GATEWAY

    def __init__(self, model: str, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Embedding failed with model '{model}': {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"model": model, "reason": reason, **(context or {})},
        )


class DocumentAlreadyExistsException(IngestionException):
    """Raised when a document with the same fingerprint is already indexed."""
    error_code = ErrorCode.INGEST_ALREADY_EXISTS
    http_status = HTTPStatus.CONFLICT

    def __init__(self, fingerprint: str, file_name: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Document '{file_name}' is already indexed (fingerprint: {fingerprint[:12]}…)",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"fingerprint": fingerprint, "file_name": file_name, **(context or {})},
        )