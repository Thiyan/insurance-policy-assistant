import logging

import chromadb

from app.config.config import APP_CONFIG
from app.model.rag_response import RAGResponse
from app.service.generation.llm import generate_answer
from app.service.retrieval.context_retrieve import retrieve

logger = logging.getLogger(__name__)


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
