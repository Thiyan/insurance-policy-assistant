import sys
import traceback
from http import HTTPStatus

from app.exception.error_codes import ErrorCode


class PolicyApplicationException(Exception):
    """
    Base exception for the entire RAG application.

    Captures:
      - A human-readable error message
      - The file name and line number where the exception was raised
      - Full traceback as a string
      - An optional error code for programmatic handling
      - Optional extra context (dict) for structured logging

    Usage:
        try:
            risky_operation()
        except Exception:
            raise RAGApplicationException("Something went wrong", error_code=ErrorCode.UNKNOWN_ERROR)
    """

    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR
    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(
            self,
            error_message: str,
            error_details=sys,
            error_code: ErrorCode = None,
            context: dict = None,
            http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR,
    ):
        super().__init__(str(error_message))

        self.http_status = http_status
        _, _, exc_tb = error_details.exc_info()

        self.error_message = str(error_message)
        self.error_code = error_code or self.__class__.error_code
        self.context = context or {}

        if exc_tb is not None:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.line_number = exc_tb.tb_lineno
        else:
            self.file_name = "Unknown"
            self.line_number = -1

        self.traceback_str = "".join(
            traceback.format_exception(*error_details.exc_info())
        )

    def __str__(self) -> str:
        context_str = f"\nContext: {self.context}" if self.context else ""
        return (
            f"[{self.__class__.__name__}] [{self.error_code}] "
            f"in [{self.file_name}] line [{self.line_number}] | "
            f"Message: {self.error_message}"
            f"{context_str}\n"
            f"Traceback:\n{self.traceback_str}"
        )

    def to_dict(self) -> dict:
        """Serialise for structured logging or API error responses."""
        return {
            "exception": self.__class__.__name__,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "file_name": self.file_name,
            "line_number": self.line_number,
            "context": self.context,
            "http_status": self.http_status,
        }