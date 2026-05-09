import logging
import time

from app.config.config import APP_CONFIG
from app.exception.sub_exception.ingestion_exception import DocumentParseException, ChunkingException, \
    IngestionException, FileNotFoundException, UnsupportedFileTypeException, EmbeddingException
from app.exception.sub_exception.persistence_exception import PersistenceException
from app.model.ingestion_result import IngestionResult
from app.persistence.db import store_in_chromadb
from app.service.ingestion.chunking import chunk_pages
from app.service.ingestion.embedding import embed_chunks
from app.service.ingestion.pre_processing import extract_doc_metadata, extract_page_markdowns

logger = logging.getLogger(__name__)


def execute_ingestion_pipeline(
    pdf_path: str,
    db_path: str = APP_CONFIG.DB_PATH,
    collection_name: str = APP_CONFIG.COLLECTION_NAME,
) -> IngestionResult:
    """Run the full PDF ingestion pipeline: extract → chunk → embed → store.

    Args:
        pdf_path:        Path to the source PDF file.
        db_path:         Directory for the ChromaDB persistent store.
        collection_name: Target ChromaDB collection.

    Returns:
        IngestionResult summarising what was processed and stored.

    Raises:
        FileNotFoundException:        PDF does not exist at the given path.
        UnsupportedFileTypeException: File is not a PDF.
        DocumentParseException:       PDF is corrupted, encrypted, or unreadable.
        ChunkingException:            Text splitting failed.
        EmbeddingException:           OpenAI embedding call failed.
        PersistenceException:         ChromaDB write failed.
    """
    started_at = time.monotonic()
    logger.info("Starting ingestion pipeline for '%s'", pdf_path)

    doc_metadata = _step("metadata extraction", lambda: extract_doc_metadata(pdf_path))
    logger.info("  Title: '%s' | Author: '%s' | Pages: %d",
                doc_metadata.title, doc_metadata.author, doc_metadata.total_pages)

    pages = _step("page extraction", lambda: extract_page_markdowns(pdf_path))
    logger.info("  Extracted %d non-empty page(s)", len(pages))

    if not pages:
        raise DocumentParseException(
            file_name=pdf_path,
            reason="No extractable text found — PDF may be image-only",
        )

    chunks = _step("chunking", lambda: chunk_pages(pages, doc_metadata))
    logger.info("  Created %d chunk(s)", len(chunks))

    if not chunks:
        raise ChunkingException(
            file_name=pdf_path,
            reason="Chunking produced no output — all page content may be whitespace",
        )

    embedded_chunks = _step("embedding", lambda: embed_chunks(chunks))

    store_result = _step(
        "persistence",
        lambda: store_in_chromadb(embedded_chunks, db_path, collection_name),
    )

    result = IngestionResult(
        doc_metadata=doc_metadata,
        pages_extracted=len(pages),
        chunks_created=len(chunks),
        store_result=store_result,
        elapsed_seconds=round(time.monotonic() - started_at, 3),
    )

    logger.info("Ingestion complete — %s", result.summary)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _step(name: str, fn):
    """Execute a pipeline step, time it, and surface any exception with context.

    Known custom exceptions are re-raised as-is so callers get the precise
    type. Any unexpected exception is wrapped in IngestionException.
    """
    t = time.monotonic()
    try:
        result = fn()
        logger.debug("Step '%s' completed in %.3fs", name, time.monotonic() - t)
        return result
    except (
        FileNotFoundException,
        UnsupportedFileTypeException,
        DocumentParseException,
        ChunkingException,
        EmbeddingException,
        PersistenceException,
    ):
        raise
    except Exception as exc:
        raise IngestionException(
            error_message=f"Unexpected error in '{name}' step: {exc}",
            context={"step": name},
        ) from exc