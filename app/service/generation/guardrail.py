import logging

from app.config.config import DEFAULT_GUARDRAIL_CONFIG
from app.config.constants import LLM_REFUSAL_PHRASES, INJECTION_PATTERNS
from app.exception.sub_exception.guardrail_exception import InputGuardrailViolation, ContextGuardrailViolation, \
    OutputGuardrailViolation
from app.model.query_match import QueryMatch

logger = logging.getLogger(__name__)


def check_input(question: str) -> None:
    """Validate and sanitise the user question before it reaches the LLM."""
    if not question or not question.strip():
        raise InputGuardrailViolation("Question must not be empty.")

    stripped = question.strip()

    if len(stripped) < DEFAULT_GUARDRAIL_CONFIG.MIN_QUESTION_CHARS:
        raise InputGuardrailViolation(
            f"Question is too short (minimum {DEFAULT_GUARDRAIL_CONFIG.MIN_QUESTION_CHARS} characters).",
        )

    if len(stripped) > DEFAULT_GUARDRAIL_CONFIG.MAX_QUESTION_CHARS:
        raise InputGuardrailViolation(
            f"Question exceeds the {DEFAULT_GUARDRAIL_CONFIG.MAX_QUESTION_CHARS}-character limit "
            f"({len(stripped)} chars).",
        )

    if INJECTION_PATTERNS.search(stripped):
        logger.warning("Prompt injection attempt detected: %.120s", stripped)
        raise InputGuardrailViolation(
            "Question contains disallowed instruction-override patterns.",
        )


def check_context(matches: list[QueryMatch]) -> list[QueryMatch]:
    """Ensure the retrieval result is usable and fits within the context budget.

    Returns a (possibly truncated) list of matches safe to send to the LLM.
    """
    if not matches or len(matches) < DEFAULT_GUARDRAIL_CONFIG.MIN_MATCHES_REQUIRED:
        raise ContextGuardrailViolation(
            f"At least {DEFAULT_GUARDRAIL_CONFIG.MIN_MATCHES_REQUIRED} retrieved chunk(s) are required "
            "to generate a grounded answer.",
        )

    # Greedily include matches until the char budget is exhausted.
    selected: list[QueryMatch] = []
    total = 0
    for match in matches:
        chunk_len = len(match.text or "")
        if total + chunk_len > DEFAULT_GUARDRAIL_CONFIG.MAX_CONTEXT_CHARS:
            logger.debug(
                "Context budget reached — using %d/%d matches.", len(selected), len(matches)
            )
            break
        selected.append(match)
        total += chunk_len

    if not selected:
        # The very first chunk already exceeds the budget — take it truncated.
        first = matches[0]
        truncated_text = (first.text or "")[:DEFAULT_GUARDRAIL_CONFIG.MAX_CONTEXT_CHARS]
        logger.warning(
            "Single chunk exceeds context budget; truncating to %d chars.", DEFAULT_GUARDRAIL_CONFIG.MAX_CONTEXT_CHARS
        )
        # Return a shallow copy with the text replaced so we don't mutate the original.
        truncated = QueryMatch(
            text=truncated_text,
            source=first.source,
            page_number=first.page_number,
        )
        selected = [truncated]

    return selected


def check_output(answer: str, matches: list[QueryMatch]) -> None:
    """Validate the LLM's response before returning it to the caller."""
    if not answer or not answer.strip():
        raise OutputGuardrailViolation("LLM returned an empty response.")

    if len(answer) > DEFAULT_GUARDRAIL_CONFIG.MAX_ANSWER_CHARS:
        raise OutputGuardrailViolation(
            f"LLM response exceeds the {DEFAULT_GUARDRAIL_CONFIG.MAX_ANSWER_CHARS}-character safety limit.",
        )

    lower_answer = answer.lower()

    # Detect soft refusals the LLM sometimes produces.
    for phrase in LLM_REFUSAL_PHRASES:
        if phrase in lower_answer:
            logger.warning("Possible LLM refusal detected in answer.")
            raise OutputGuardrailViolation(
                "LLM declined to answer. The question may be outside its permitted scope.",
            )

    # Grounding check — at least one source must be referenced in the answer.
    # sources = {m.source for m in matches if m.source}
    # if sources and not any(src.lower() in lower_answer for src in sources):
    #     logger.warning(
    #         "Answer does not reference any retrieved source. "
    #         "Possible hallucination. Sources: %s", sources
    #     )
    #     raise OutputGuardrailViolation(
    #         "The generated answer could not be grounded in the retrieved sources.",
    #     )
