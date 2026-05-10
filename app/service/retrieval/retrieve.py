import logging

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from app.config.config import APP_CONFIG
from app.config.constants import SYSTEM_PROMPT
from app.exception.sub_exception.retrieval_exception import VectorSearchException, EmptyRetrievalResultException, \
    LLMException
from app.model.query_match import QueryMatch
from app.model.query_result import QueryResult
from app.model.rag_response import RAGResponse
from app.persistence.db import query_collection

logger = logging.getLogger(__name__)


def retrieve(
    collection: chromadb.Collection,
    question: str,
    n_results: int = APP_CONFIG.N_RESULTS,
) -> QueryResult:
    """Embed a question and retrieve the top-n matching chunks from ChromaDB.

    Args:
        collection: The ChromaDB collection to search.
        question:   Natural language query string.
        n_results:  Number of chunks to retrieve.

    Returns:
        QueryResult with ranked QueryMatch objects.

    Raises:
        EmptyRetrievalResultException: No matching chunks found for the question.
        VectorSearchException:         ChromaDB query failed.
    """
    if not question or not question.strip():
        raise VectorSearchException(
            query=question,
            reason="Question must be a non-empty string",
        )

    try:
        result = query_collection(collection, question, n_results=n_results)
    except Exception as exc:
        raise VectorSearchException(
            query=question,
            reason=f"ChromaDB query failed: {exc}",
        ) from exc

    if not result.matches:
        raise EmptyRetrievalResultException(query=question)

    logger.debug("Retrieved %d chunk(s) for question: %r", len(result.matches), question)
    return result


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
        LLMException: LLM call fails for any reason.
    """
    context = _build_context(retrieval.matches)
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

    logger.debug("Answer generated using model '%s'", model)
    return response.content


def rag_query(
    collection: chromadb.Collection,
    question: str,
    n_results: int = APP_CONFIG.N_RESULTS,
) -> RAGResponse:
    """Run a full retrieval-augmented generation query.

    Args:
        collection: ChromaDB collection to retrieve from.
        question:   User question to answer.
        n_results:  Number of chunks to retrieve and ground the answer on.

    Returns:
        RAGResponse containing the answer, question, and full retrieval trace.

    Raises:
        EmptyRetrievalResultException: No relevant chunks found.
        VectorSearchException:         Retrieval step failed.
        LLMException:                  Generation step failed.
    """
    logger.info("RAG query started | question: %r", question)

    retrieval = retrieve(collection, question, n_results)
    answer = generate_answer(question, retrieval)

    response = RAGResponse(question=question, answer=answer, retrieval=retrieval)
    logger.info(
        "RAG query complete | chunks=%d | pages=%s | question: %r | answer: %r",
        len(retrieval.matches), response.source_pages, question, response.answer
    )
    return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(matches: list[QueryMatch]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    return "\n\n".join(
        f"[{i + 1}] Source: {m.source}, Page {m.page_number}\n{m.text}"
        for i, m in enumerate(matches)
    )