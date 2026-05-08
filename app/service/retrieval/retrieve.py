import logging

import chromadb
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.config.config import N_RESULTS, LLM_MODEL, SYSTEM_PROMPT, TEMPERATURE
from app.model.query_match import QueryMatch
from app.model.query_result import QueryResult
from app.model.rag_response import RAGResponse
from app.persistence.db import query_collection

logger = logging.getLogger(__name__)

def retrieve(
    collection: chromadb.Collection,
    question: str,
    n_results: int = N_RESULTS,
) -> QueryResult:
    """Embed a question and retrieve the top-n matching chunks from ChromaDB.

    Args:
        collection: The ChromaDB collection to search.
        question: Natural language query string.
        n_results: Number of chunks to retrieve.

    Returns:
        QueryResult with ranked QueryMatch objects.
    """
    result = query_collection(collection, question, n_results=n_results)
    logger.debug("Retrieved %d chunks for question: %r", len(result.matches), question)
    return result


def generate_answer(
    question: str,
    retrieval: QueryResult,
    model: str = LLM_MODEL,
) -> str:
    """Generate a grounded answer from retrieved context using an LLM.

    Args:
        question: The original user question.
        retrieval: Retrieved chunks to use as context.
        model: ChatOpenAI model name.

    Returns:
        The LLM's answer string.
    """
    context = _build_context(retrieval.matches)
    llm = ChatOpenAI(model=model, temperature=TEMPERATURE)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)
    return response.content


def _build_context(matches: list[QueryMatch]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    parts = [
        f"[{i + 1}] Source: {m.source}, Page {m.page_number}\n{m.text}"
        for i, m in enumerate(matches)
    ]
    return "\n\n".join(parts)


def rag_query(
    collection: chromadb.Collection,
    question: str,
    n_results: int = N_RESULTS,
) -> RAGResponse:
    """Run a full retrieval-augmented generation query.

    Args:
        collection: ChromaDB collection to retrieve from.
        question: User question to answer.
        n_results: Number of chunks to retrieve and ground the answer on.

    Returns:
        RAGResponse containing the answer, question, and full retrieval trace.
    """
    retrieval = retrieve(collection, question, n_results)
    answer = generate_answer(question, retrieval)

    response = RAGResponse(question=question, answer=answer, retrieval=retrieval)
    # logger.info(
    #     "RAG query complete | pages=%s | question=%r",
    #     response.source_pages, question,
    # )
    print(
        "RAG query complete | pages=%s | question=%r",
        response.source_pages, question,
        response.answer
    )
    return response