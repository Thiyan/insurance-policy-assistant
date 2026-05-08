import logging
import os
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings

from app.config.config import EMBEDDING_MODEL, N_RESULTS, HNSW_SPACE
from app.model.chunk import Chunk
from app.model.embedded_chunk import EmbeddedChunk
from app.model.query_match import QueryMatch
from app.model.query_result import QueryResult
from app.model.stored_result import StoreResult

logger = logging.getLogger(__name__)

def store_in_chromadb(
    embedded_chunks: list[EmbeddedChunk],
    db_path: str,
    collection_name: str,
    replace: bool = True,
) -> StoreResult:
    """Persist embedded chunks into a ChromaDB collection.

    Args:
        embedded_chunks: Paired chunks and their embedding vectors.
        db_path: Directory path for the ChromaDB persistent store.
        collection_name: Target collection name.
        replace: If True, drop and recreate the collection on re-runs.
                 If False, raise an error when the collection already exists.

    Returns:
        StoreResult summarising what was written.

    Raises:
        ValueError: If embedded_chunks is empty.
        FileExistsError: If replace=False and the collection already exists.
    """
    if not embedded_chunks:
        raise ValueError("No embedded chunks provided — nothing to store.")

    Path(db_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    was_replaced = _prepare_collection(client, collection_name, replace)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": HNSW_SPACE},
    )

    collection.add(
        ids=[ec.chunk.id for ec in embedded_chunks],
        documents=[ec.chunk.text for ec in embedded_chunks],
        metadatas=[_chunk_metadata(ec.chunk) for ec in embedded_chunks],
        embeddings=[ec.embedding for ec in embedded_chunks],
    )

    result = StoreResult(
        collection_name=collection_name,
        db_path=str(Path(db_path).resolve()),
        chunks_stored=len(embedded_chunks),
        was_replaced=was_replaced,
    )
    logger.info(
        "Stored %d chunks into '%s' (collection: '%s', replaced: %s)",
        result.chunks_stored, result.db_path, result.collection_name, result.was_replaced,
    )
    return result


def _prepare_collection(
    client: chromadb.ClientAPI,
    collection_name: str,
    replace: bool,
) -> bool:
    """Delete an existing collection if replace=True, or raise if replace=False.

    Returns:
        True if an existing collection was dropped, False if none existed.
    """
    exists = collection_name in {c.name for c in client.list_collections()}
    if not exists:
        return False
    if not replace:
        raise FileExistsError(
            f"Collection '{collection_name}' already exists and replace=False."
        )
    client.delete_collection(collection_name)
    return True


def _chunk_metadata(chunk: Chunk) -> dict:
    """Serialise a Chunk's metadata fields for ChromaDB storage."""
    return {
        "source": chunk.doc_metadata.source,
        "title": chunk.doc_metadata.title,
        "author": chunk.doc_metadata.author,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "total_pages": chunk.doc_metadata.total_pages,
    }


def query_collection(
    collection: chromadb.Collection,
    question: str,
    n_results: int = N_RESULTS,
    model: str = EMBEDDING_MODEL,
) -> QueryResult:
    """Query a ChromaDB collection using semantic similarity.

    Args:
        collection: The ChromaDB collection to search.
        question: Natural language query string.
        n_results: Number of top matches to return.
        model: OpenAI embedding model — must match the one used at index time.

    Returns:
        QueryResult containing the question and ranked matches.
    """
    embeddings_model = OpenAIEmbeddings(model=model)
    query_embedding = embeddings_model.embed_query(question)

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    matches = [
        _parse_match(doc, raw)
        for doc, meta in zip(raw["documents"][0], raw["metadatas"][0])
    ]

    result = QueryResult(question=question, matches=matches)
    logger.debug("Query '%s' returned %d matches", question, len(matches))
    return result


def _parse_match(document: str, metadata: dict) -> QueryMatch:
    """Deserialize a raw ChromaDB result row into a QueryMatch."""
    return QueryMatch(
        text=document,
        source=metadata["source"],
        title=metadata["title"],
        author=metadata["author"],
        page_number=metadata["page_number"],
        total_pages=metadata["total_pages"],
        chunk_index=metadata["chunk_index"],
    )

def load_collection(db_path: str, collection_name: str) -> chromadb.Collection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"ChromaDB not found at '{db_path}'. Run the ingestion pipeline first."
        )

    client = chromadb.PersistentClient(path=db_path)

    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        raise ValueError(
            f"Collection '{collection_name}' not found. Available: {existing}"
        )

    collection = client.get_collection(name=collection_name)
    print(f"✅ Loaded collection '{collection_name}' with {collection.count()} chunks")
    return collection