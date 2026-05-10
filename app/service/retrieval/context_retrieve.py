import logging

import chromadb

from app.config.config import APP_CONFIG
from app.exception.sub_exception.retrieval_exception import VectorSearchException, EmptyRetrievalResultException
from app.model.query_result import QueryResult
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
