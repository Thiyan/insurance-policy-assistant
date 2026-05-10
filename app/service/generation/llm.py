import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph_sdk.errors import AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError

from app.config.config import APP_CONFIG
from app.config.constants import SYSTEM_PROMPT
from app.exception.sub_exception.retrieval_exception import LLMException
from app.model.query_match import QueryMatch
from app.model.query_result import QueryResult
from app.service.generation.guardrail import check_input, check_output, check_context

logger = logging.getLogger(__name__)


def generate_answer(
    question: str,
    retrieval: QueryResult,
    model: str = APP_CONFIG.LLM_MODEL,
) -> str:
    """Generate a grounded answer from retrieved context using an LLM.

    Args:
        question:  The original user question.
        retrieval: Retrieved chunks to use as context.
        model:     ChatOpenAI model name.

    Returns:
        The LLM's answer string.

    Raises:
        GuardrailViolation: An input or output policy was violated.
        LLMException:       The LLM call failed for a technical reason.
    """
    check_input(question)

    safe_matches = check_context(retrieval.matches)
    context = _build_context(safe_matches)

    answer = _call_llm(question, context, model)

    check_output(answer, safe_matches)

    logger.debug("Answer generated using model '%s'", model)
    return answer



# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(question: str, context: str, model: str) -> str:
    """Invoke the LLM and map provider errors to LLMException."""
    llm = ChatOpenAI(model=model, temperature=APP_CONFIG.TEMPERATURE)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    try:
        response = llm.invoke(messages)
    except AuthenticationError as exc:
        raise LLMException(model=model, reason=f"Invalid API key: {exc}") from exc
    except RateLimitError as exc:
        raise LLMException(model=model, reason=f"Rate limit exceeded: {exc}") from exc
    except APITimeoutError as exc:
        raise LLMException(model=model, reason=f"Request timed out: {exc}") from exc
    except APIConnectionError as exc:
        raise LLMException(model=model, reason=f"Connection error: {exc}") from exc
    except Exception as exc:
        raise LLMException(model=model, reason=f"Unexpected error during generation: {exc}") from exc

    return response.content


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(matches: list[QueryMatch]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    return "\n\n".join(
        f"[{i + 1}] Source: {m.source}, Page {m.page_number}\n{m.text}"
        for i, m in enumerate(matches)
    )