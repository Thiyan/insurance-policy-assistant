import logging
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings

from app.config.config import APP_CONFIG
from app.exception.sub_exception.ingestion_exception import EmbeddingException
from app.exception.sub_exception.persistence_exception import DatabaseWriteException, DatabaseReadException, \
    RecordNotFoundException, DatabaseConnectionException, RecordConflictException
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
        db_path:         Directory path for the ChromaDB persistent store.
        collection_name: Target collection name.
        replace:         If True, drop and recreate the collection on re-runs.
                         If False, raise when the collection already exists.

    Returns:
        StoreResult summarising what was written.

    Raises:
        DatabaseWriteException:   Store operation failed.
        RecordConflictException:  replace=False and collection already exists.
        DatabaseConnectionException: ChromaDB client could not be initialised.
    """
    if not embedded_chunks:
        raise DatabaseWriteException(
            operation="add",
            entity=collection_name,
            reason="No embedded chunks provided — nothing to store",
        )

    client = _get_client(db_path)
    was_replaced = _prepare_collection(client, collection_name, replace)

    try:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": APP_CONFIG.HNSW_SPACE},
        )
        collection.add(
            ids=[ec.chunk.id for ec in embedded_chunks],
            documents=[ec.chunk.text for ec in embedded_chunks],
            metadatas=[_chunk_metadata(ec.chunk) for ec in embedded_chunks],
            embeddings=[ec.embedding for ec in embedded_chunks],
        )
    except Exception as exc:
        raise DatabaseWriteException(
            operation="add",
            entity=collection_name,
            reason=f"ChromaDB write failed: {exc}",
        ) from exc

    result = StoreResult(
        collection_name=collection_name,
        db_path=str(Path(db_path).resolve()),
        chunks_stored=len(embedded_chunks),
        was_replaced=was_replaced,
    )
    logger.info(
        "Stored %d chunk(s) into collection '%s' at '%s' (replaced: %s)",
        result.chunks_stored, result.collection_name, result.db_path, result.was_replaced,
    )
    return result


def query_collection(
    collection: chromadb.Collection,
    question: str,
    n_results: int = APP_CONFIG.N_RESULTS,
    model: str = APP_CONFIG.EMBEDDING_MODEL,
) -> QueryResult:
    """Query a ChromaDB collection using semantic similarity.

    Args:
        collection: The ChromaDB collection to search.
        question:   Natural language query string.
        n_results:  Number of top matches to return.
        model:      OpenAI embedding model — must match the one used at index time.

    Returns:
        QueryResult containing the question and ranked matches.

    Raises:
        EmbeddingException:    Query embedding failed.
        DatabaseReadException: ChromaDB query failed.
    """
    try:
        query_embedding = OpenAIEmbeddings(model=model).embed_query(question)
    except Exception as exc:
        raise EmbeddingException(
            model=model,
            reason=f"Failed to embed query: {exc}",
        ) from exc

    try:
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
    except Exception as exc:
        raise DatabaseReadException(
            entity=collection.name,
            reason=f"ChromaDB query failed: {exc}",
        ) from exc

    matches = [
        _parse_match(doc, meta)
        for doc, meta in zip(raw["documents"][0], raw["metadatas"][0])
    ]

    logger.debug("Query returned %d match(es) for: %r", len(matches), question)
    return QueryResult(question=question, matches=matches)


def load_collection(db_path: str, collection_name: str) -> chromadb.Collection:
    """Load an existing ChromaDB collection.

    Args:
        db_path:         Path to the ChromaDB persistent store directory.
        collection_name: Name of the collection to load.

    Returns:
        The ChromaDB Collection object.

    Raises:
        DatabaseConnectionException: db_path does not exist.
        RecordNotFoundException:     Collection not found in the store.
        DatabaseReadException:       get_collection call failed unexpectedly.
    """
    client = _get_client(db_path)

    existing = {c.name for c in client.list_collections()}
    if collection_name not in existing:
        raise RecordNotFoundException(
            entity="ChromaDB collection",
            identifier=collection_name,
            context={"available_collections": sorted(existing)},
        )

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise DatabaseReadException(
            entity=collection_name,
            reason=f"Failed to load collection: {exc}",
        ) from exc

    logger.info(
        "Loaded collection '%s' with %d chunk(s) from '%s'",
        collection_name, collection.count(), db_path,
    )
    return collection


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_client(db_path: str) -> chromadb.ClientAPI:
    """Initialise a persistent ChromaDB client, creating the directory if needed.

    Raises:
        DatabaseConnectionException: Path is invalid or client init fails.
    """
    try:
        Path(db_path).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=db_path)
    except Exception as exc:
        raise DatabaseConnectionException(
            db_url_hint=db_path,
            reason=f"Failed to initialise ChromaDB client: {exc}",
        ) from exc


def _prepare_collection(
    client: chromadb.ClientAPI,
    collection_name: str,
    replace: bool,
) -> bool:
    """Delete an existing collection if replace=True, or raise if replace=False.

    Returns:
        True if an existing collection was dropped, False if none existed.

    Raises:
        RecordConflictException: Collection exists and replace=False.
        DatabaseWriteException:  Deletion failed unexpectedly.
    """
    exists = collection_name in {c.name for c in client.list_collections()}
    if not exists:
        return False

    if not replace:
        raise RecordConflictException(
            entity="ChromaDB collection",
            field="name",
            value=collection_name,
        )

    try:
        client.delete_collection(collection_name)
    except Exception as exc:
        raise DatabaseWriteException(
            operation="delete",
            entity=collection_name,
            reason=f"Failed to drop existing collection: {exc}",
        ) from exc

    logger.debug("Dropped existing collection '%s'", collection_name)
    return True


def _chunk_metadata(chunk: Chunk) -> dict:
    """Serialise a Chunk's metadata fields for ChromaDB storage."""
    return {
        "source":      chunk.doc_metadata.source,
        "title":       chunk.doc_metadata.title,
        "author":      chunk.doc_metadata.author,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "total_pages": chunk.doc_metadata.total_pages,
    }


def _parse_match(document: str, metadata: dict) -> QueryMatch:
    """Deserialise a raw ChromaDB result row into a QueryMatch."""
    return QueryMatch(
        text=document,
        source=metadata["source"],
        title=metadata["title"],
        author=metadata["author"],
        page_number=metadata["page_number"],
        total_pages=metadata["total_pages"],
        chunk_index=metadata["chunk_index"],
    )