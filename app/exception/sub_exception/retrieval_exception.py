import sys
from http import HTTPStatus

from app.exception.error_codes import ErrorCode
from app.exception.policy_application_exception import PolicyApplicationException


class RetrievalException(PolicyApplicationException):
    """Base for all retrieval-pipeline failures."""
    error_code = ErrorCode.UNKNOWN_ERROR
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR


class VectorSearchException(RetrievalException):
    """Raised when the vector DB similarity search fails."""
    error_code = ErrorCode.RETRIEVAL_QUERY_FAILED
    http_status = HTTPStatus.SERVICE_UNAVAILABLE

    def __init__(self, query: str, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Vector search failed for query '{query[:80]}…': {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"query_preview": query[:80], "reason": reason, **(context or {})},
        )


class RerankerException(RetrievalException):
    """Raised when the cross-encoder reranker fails."""
    error_code = ErrorCode.RETRIEVAL_RERANK_FAILED
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"Reranker failed: {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"reason": reason, **(context or {})},
        )


class LLMException(RetrievalException):
    """Raised when the language model call fails (timeout, quota, bad response)."""
    error_code = ErrorCode.RETRIEVAL_LLM_FAILED
    http_status = HTTPStatus.BAD_GATEWAY

    def __init__(self, model: str, reason: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"LLM '{model}' call failed: {reason}",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"model": model, "reason": reason, **(context or {})},
        )


class EmptyRetrievalResultException(RetrievalException):
    """Raised when the retriever returns no relevant documents for a query."""
    error_code = ErrorCode.RETRIEVAL_EMPTY_RESULT
    http_status = HTTPStatus.NOT_FOUND

    def __init__(self, query: str, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"No relevant documents found for query: '{query[:80]}'",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"query_preview": query[:80], **(context or {})},
        )


class RetrievalTimeoutException(RetrievalException):
    """Raised when the full RAG pipeline exceeds the configured timeout."""
    error_code = ErrorCode.RETRIEVAL_TIMEOUT
    http_status = HTTPStatus.GATEWAY_TIMEOUT

    def __init__(self, timeout_seconds: float, error_details=sys, context: dict = None):
        super().__init__(
            error_message=f"RAG pipeline timed out after {timeout_seconds}s",
            error_details=error_details,
            error_code=self.__class__.error_code,
            context={"timeout_seconds": timeout_seconds, **(context or {})},
        )