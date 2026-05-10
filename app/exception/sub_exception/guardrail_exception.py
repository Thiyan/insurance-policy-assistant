from http import HTTPStatus

from app.exception.error_codes import ErrorCode
from app.exception.sub_exception.retrieval_exception import RetrievalException


class GuardrailViolation(RetrievalException):
    """Base for all guardrail failures in the generation pipeline.

    Prefer the concrete subclasses (InputGuardrailViolation,
    ContextGuardrailViolation, OutputGuardrailViolation) over raising this
    directly — they carry the correct HTTP status and error code for their
    stage.
    """

    error_code = ErrorCode.GUARDRAIL_VIOLATION
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class InputGuardrailViolation(GuardrailViolation):
    """Raised when the user's question fails an input guardrail.

    Examples: empty question, excessive length, prompt-injection attempt.
    Maps to 400 Bad Request — the problem is with the client's input.
    """

    error_code = ErrorCode.GUARDRAIL_INVALID_INPUT
    http_status = HTTPStatus.BAD_REQUEST


class ContextGuardrailViolation(GuardrailViolation):
    """Raised when the retrieved context fails a guardrail.

    Examples: no chunks retrieved, context budget exceeded with no usable
    content.  Maps to 422 Unprocessable Entity — the request was valid but
    the pipeline cannot produce a grounded answer from what was retrieved.
    """

    error_code = ErrorCode.GUARDRAIL_VIOLATION
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY


class OutputGuardrailViolation(GuardrailViolation):
    """Raised when the LLM's response fails an output guardrail.

    Examples: empty answer, soft refusal detected, grounding check failed.
    Maps to 502 Bad Gateway — the upstream model returned an unusable
    response.
    """

    error_code = ErrorCode.GUARDRAIL_VIOLATION
    http_status = HTTPStatus.BAD_GATEWAY
