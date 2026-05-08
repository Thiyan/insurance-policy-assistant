
import logging
import time

from app.config.config import DB_PATH, COLLECTION_NAME
from app.model.ingestion_result import IngestionResult
from app.persistence.db import store_in_chromadb
from app.service.ingestion.chunking import chunk_pages
from app.service.ingestion.embedding import embed_chunks
from app.service.ingestion.pre_processing import extract_doc_metadata, extract_page_markdowns

logger = logging.getLogger(__name__)

def execute_ingestion_pipeline(
    pdf_path: str,
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION_NAME,
) -> IngestionResult:
    """Run the full PDF ingestion pipeline: extract → chunk → embed → store.

    Args:
        pdf_path: Path to the source PDF file.
        db_path: Directory for the ChromaDB persistent store.
        collection_name: Target ChromaDB collection.

    Returns:
        IngestionResult summarising what was processed and stored.

    Raises:
        FileNotFoundError: If the PDF does not exist.
    """
    started_at = time.monotonic()

    logger.info("Extracting metadata from '%s'", pdf_path)
    doc_metadata = extract_doc_metadata(pdf_path)
    logger.info("  Title: %s | Author: %s | Pages: %d",
                doc_metadata.title, doc_metadata.author, doc_metadata.total_pages)

    logger.info("Extracting pages as Markdown...")
    pages = extract_page_markdowns(pdf_path)
    logger.info("  Extracted %d non-empty pages", len(pages))

    logger.info("Chunking text...")
    chunks = chunk_pages(pages, doc_metadata)
    logger.info("  Created %d chunks", len(chunks))

    logger.info("Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)

    logger.info("Persisting to ChromaDB...")
    store_result = store_in_chromadb(embedded_chunks, db_path, collection_name)

    result = IngestionResult(
        doc_metadata=doc_metadata,
        pages_extracted=len(pages),
        chunks_created=len(chunks),
        store_result=store_result,
        elapsed_seconds=time.monotonic() - started_at,
    )
    logger.info(result.summary)
    return result