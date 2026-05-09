import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.exception.sub_exception.ingestion_exception import ChunkingException
from app.model.chunk import Chunk
from app.model.doc_metadata import DocMetadata
from app.model.page_content import PageContent

logger = logging.getLogger(__name__)


def chunk_pages(
    pages: list[PageContent],
    doc_metadata: DocMetadata,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split pages into overlapping text chunks enriched with metadata.

    Args:
        pages:          Extracted page contents to chunk.
        doc_metadata:   Document-level metadata attached to every chunk.
        chunk_size:     Maximum character length of each chunk.
        chunk_overlap:  Number of overlapping characters between chunks.

    Returns:
        Ordered list of Chunk objects across all pages.

    Raises:
        ChunkingException: Invalid config values, or splitting fails.
    """
    if not pages:
        logger.debug("chunk_pages received empty page list — returning []")
        return []

    _validate_config(chunk_size, chunk_overlap, doc_metadata.source)

    logger.info(
        "Chunking %d page(s) from '%s' (chunk_size=%d, overlap=%d)",
        len(pages), doc_metadata.source, chunk_size, chunk_overlap,
    )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = [
            _make_chunk(sub_chunk, page.page_number, chunk_index, doc_metadata)
            for chunk_index, (page, sub_chunk) in enumerate(
                (page, sub_chunk)
                for page in pages
                for sub_chunk in splitter.split_text(page.text)
                if sub_chunk.strip()
            )
        ]

    except ChunkingException:
        raise
    except Exception as exc:
        raise ChunkingException(
            file_name=doc_metadata.source,
            reason=f"Unexpected error during text splitting: {exc}",
        ) from exc

    if not chunks:
        logger.warning("No chunks produced from '%s' — all pages may be whitespace-only", doc_metadata.source)
    else:
        logger.info("Produced %d chunk(s) from '%s'", len(chunks), doc_metadata.source)

    return chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_config(chunk_size: int, chunk_overlap: int, source: str) -> None:
    if chunk_size < 1:
        raise ChunkingException(
            file_name=source,
            reason=f"chunk_size must be >= 1, got {chunk_size}",
        )
    if chunk_overlap < 0:
        raise ChunkingException(
            file_name=source,
            reason=f"chunk_overlap must be >= 0, got {chunk_overlap}",
        )
    if chunk_overlap >= chunk_size:
        raise ChunkingException(
            file_name=source,
            reason=f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})",
        )


def _make_chunk(
    text: str,
    page_number: int,
    chunk_index: int,
    doc_metadata: DocMetadata,
) -> Chunk:
    return Chunk(
        id=f"chunk_{chunk_index}",
        text=text,
        page_number=page_number,
        chunk_index=chunk_index,
        doc_metadata=doc_metadata,
    )